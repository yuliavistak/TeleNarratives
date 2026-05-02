"""
https://github.com/MuMiN-dataset/mumin-baseline/blob/main/src/train_graph_model.py

Training utilities for the heterogeneous GraphSAGE message-classification model."""

import copy
import csv
import datetime as dt
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm

from src.disinfograph.gnn.dgl import load_dgl_graph, save_dgl_graph


logger = logging.getLogger(__name__)


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    """Write *rows* to a CSV file at *path*.

    Args:
        path: Destination file path (parent directory must exist).
        fieldnames: Ordered list of column names.
        rows: Sequence of dicts mapping field names to values.
    """
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalise_value(value):
    """Coerce whole-number floats to int for cleaner CSV output.

    Args:
        value: Any scalar value.

    Returns:
        An int when *value* is a float with no fractional part, otherwise the
        original value unchanged.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _get_node_field_values(graph, task: str, node_ids: torch.Tensor, field_names, default):
    """Look up per-node scalar values from graph node data.

    Tries each name in *field_names* in order and returns the first found.
    Falls back to *default* (called if callable, repeated otherwise) when none
    of the field names are present.

    Args:
        graph: DGL heterogeneous graph.
        task: Node type name to query.
        node_ids: 1-D tensor of node indices.
        field_names: Ordered sequence of data keys to try.
        default: Fallback — either a callable that receives ``cpu_node_ids``
            or a scalar value repeated for every node.

    Returns:
        A list of scalar values, one per node in *node_ids*.
    """
    node_data = graph.nodes[task].data
    cpu_node_ids = node_ids.detach().cpu().long()

    for field_name in field_names:
        if field_name not in node_data:
            continue
        values = node_data[field_name][cpu_node_ids].detach().cpu().view(-1).tolist()
        return [_normalise_value(value) for value in values]

    if callable(default):
        return default(cpu_node_ids)

    return [default for _ in range(len(cpu_node_ids))]


class BinaryF1Tracker:
    """
    Small replacement for binary classification metrics on binary tasks.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(2, dtype=torch.float64)
        self.fp = torch.zeros(2, dtype=torch.float64)
        self.fn = torch.zeros(2, dtype=torch.float64)
        self.score_chunks = []
        self.target_chunks = []

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, scores: Optional[torch.Tensor] = None) -> None:
        preds = preds.detach().view(-1).to(torch.int64).cpu()
        targets = targets.detach().view(-1).to(torch.int64).cpu()
        if scores is not None:
            self.score_chunks.append(scores.detach().view(-1).to(torch.float64).cpu())
            self.target_chunks.append(targets.to(torch.float64))

        for cls in (0, 1):
            pred_is_cls = preds == cls
            target_is_cls = targets == cls
            self.tp[cls] += (pred_is_cls & target_is_cls).sum().item()
            self.fp[cls] += (pred_is_cls & ~target_is_cls).sum().item()
            self.fn[cls] += (~pred_is_cls & target_is_cls).sum().item()

    def compute(self) -> torch.Tensor:
        precision = self.tp / torch.clamp(self.tp + self.fp, min=1.0)
        recall = self.tp / torch.clamp(self.tp + self.fn, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1.0)
        return f1.to(torch.float32)

    def compute_general_f1(self) -> torch.Tensor:
        precision = self.tp.sum() / torch.clamp(self.tp.sum() + self.fp.sum(), min=1.0)
        recall = self.tp.sum() / torch.clamp(self.tp.sum() + self.fn.sum(), min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1.0)
        return f1.to(torch.float32)

    def _stack_scores_and_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.score_chunks:
            return torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)
        return torch.cat(self.score_chunks), torch.cat(self.target_chunks)

    def compute_roc_auc(self) -> float:
        scores, targets = self._stack_scores_and_targets()
        if scores.numel() == 0:
            return 0.0

        positives = targets.sum().item()
        negatives = targets.numel() - positives
        if positives == 0 or negatives == 0:
            return 0.0

        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order]
        sorted_targets = targets[order]

        distinct = torch.where(sorted_scores[1:] != sorted_scores[:-1])[0]
        threshold_idxs = torch.cat(
            [distinct, torch.tensor([sorted_scores.numel() - 1], dtype=torch.int64)]
        )

        true_positives = torch.cumsum(sorted_targets, dim=0)[threshold_idxs]
        false_positives = (threshold_idxs.to(torch.float64) + 1.0) - true_positives

        tpr = torch.cat(
            [
                torch.tensor([0.0], dtype=torch.float64),
                true_positives / positives,
                torch.tensor([1.0], dtype=torch.float64),
            ]
        )
        fpr = torch.cat(
            [
                torch.tensor([0.0], dtype=torch.float64),
                false_positives / negatives,
                torch.tensor([1.0], dtype=torch.float64),
            ]
        )
        return float(torch.trapz(tpr, fpr).item())

    def compute_pr_auc(self) -> float:
        scores, targets = self._stack_scores_and_targets()
        if scores.numel() == 0:
            return 0.0

        positives = targets.sum().item()
        if positives == 0:
            return 0.0

        order = torch.argsort(scores, descending=True)
        sorted_targets = targets[order]
        true_positives = torch.cumsum(sorted_targets, dim=0)
        false_positives = torch.cumsum(1.0 - sorted_targets, dim=0)

        precision = true_positives / torch.clamp(true_positives + false_positives, min=1.0)
        recall = true_positives / positives
        recall_prev = torch.cat([torch.tensor([0.0], dtype=torch.float64), recall[:-1]])
        return float(((recall - recall_prev) * precision).sum().item())

    def compute_binary_metrics(self) -> Dict[str, float]:
        class_f1 = self.compute()
        # Scalar binary metrics use class 1 as the positive class.
        tn = float(self.tp[0].item())
        fp = float(self.fp[1].item())
        fn = float(self.fn[1].item())
        tp = float(self.tp[1].item())

        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        mcc_denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
        mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0.0 else 0.0

        return {
            'general_f1': float(self.compute_general_f1().item()),
            'class_0_f1': float(class_f1[0].item()),
            'class_1_f1': float(class_f1[1].item()),
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'mcc': mcc,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'pr_auc': self.compute_pr_auc(),
            'roc_auc': self.compute_roc_auc(),
        }


def _summarise_split_metrics(loss: float, scorer: BinaryF1Tracker) -> Dict[str, float]:
    metrics = scorer.compute_binary_metrics()
    metrics['loss'] = loss
    return metrics


def _compute_pos_weight(graph, task: str, train_node_ids: torch.Tensor, override: Optional[float] = None) -> float:
    """Compute BCE positive-class weight from the train split, unless overridden."""
    if override is not None:
        return float(override)

    labels = graph.nodes[task].data['label'][train_node_ids].detach().view(-1).to(torch.int64).cpu()
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())

    if positives == 0 or negatives == 0:
        return 1.0

    return negatives / positives


def _evaluate_binary_split(model,
                           dataloader,
                           task: str,
                           device: torch.device,
                           pos_weight_tensor: torch.Tensor,
                           scorer: BinaryF1Tracker,
                           split_name: str) -> Dict[str, float]:
    """Evaluate one split with fresh logits, fresh metric accumulators, and eval/no_grad semantics."""
    model.eval()
    scorer.reset()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for _, _, blocks in dataloader:
            blocks = [block.to(device) for block in blocks]

            input_feats = {n: f.float() for n, f in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['label'][task].to(device)

            logits = model(blocks, input_feats).squeeze()
            loss = F.binary_cross_entropy_with_logits(
                input=logits,
                target=output_labels.float(),
                pos_weight=pos_weight_tensor,
            )

            scorer(logits.ge(0), output_labels, torch.sigmoid(logits))
            total_loss += float(loss)
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError(f'No batches were produced for the {split_name} split.')

    return _summarise_split_metrics(total_loss / num_batches, scorer)


def subgraph_by_similar_to(graph):
    """Node-induced subgraph: only messages in a SIMILAR_TO edge + their posting channels.

    All edge types between the retained nodes are preserved. Use ``edge_types``
    in :func:`train_graph_model` to further restrict what the model sees.
    """
    import dgl
    import torch

    src, dst = graph.edges(etype=("message", "similar_to", "message"))
    msg_ids = torch.unique(torch.cat([src, dst]))

    posted_src, posted_dst = graph.edges(etype=("channel", "posted", "message"))
    mask = torch.isin(posted_dst, msg_ids)
    ch_ids = torch.unique(posted_src[mask])

    sub = dgl.node_subgraph(graph, {"message": msg_ids, "channel": ch_ids})
    print(
        f"subgraph_by_similar_to: {sub.num_nodes('message'):,} messages, "
        f"{sub.num_nodes('channel'):,} channels"
    )
    return sub


def train_graph_model(task: str,
                      size: Optional[str] = None,
                      num_epochs: int = 300,
                      random_split: bool = False,
                      graph=None,
                      graph_path: Optional[Path] = None,
                      edge_types: Optional[list] = None,
                      **_) -> Dict[str, Dict[str, float]]:
    '''
    Train a heterogeneous GraphConv model on a DGL graph.

    Args:
        task (str):
            The task to consider (for now only 'message' is supported).
        size (Optional[str]):
            The dataset size to use when loading the built-in graph.
        num_epochs (int, optional):
            The number of epochs to train for. Defaults to 300.
        random_split (bool, optional):
            Whether a random train/val/test split of the data should be
            performed. Defaults to False.
        graph (optional):
            A DGL heterograph to train on directly.
        graph_path (Optional[Path]):
            Path to a saved DGL graph to load or save.
        edge_types (Optional[list]):
            Canonical edge triplets to include, e.g.
            [("message", "forward_from", "message"),
             ("message", "similar_to", "message")].
            When None all edge types in the graph are used.

    Returns:
        dict:
            The results of the training, with keys 'train', 'val' and 'split',
            with dictionaries with the split scores as values.
    '''
    # Set random seeds
    try:
        import dgl
        from dgl.dataloading import DataLoader as NodeDataLoader
        from dgl.dataloading import MultiLayerNeighborSampler
        from src.disinfograph.gnn.model import HeteroGraphSAGE
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise RuntimeError(
            "The graph training dependencies could not be imported. "
            "This environment likely needs compatible DGL, torchmetrics, or related packages."
        ) from exc

    # Set random seeds
    torch.manual_seed(4242)
    dgl.seed(4242)

    # Set config
    config = dict(hidden_dim=1024,
                  input_dropout=0.2,
                  dropout=0.2,
                  size=size,
                  task=task,
                  lr=3e-4,
                  betas=(0.9, 0.999),
                  pos_weight='auto',
                  early_stopping_metric='pr_auc',
                  early_stopping_patience=20,
                  early_stopping_min_delta=1e-4)

    # Set up PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    resolved_graph_path = Path(graph_path) if graph_path is not None else None

    if graph is None and resolved_graph_path is not None and resolved_graph_path.exists():
        graph = load_dgl_graph(resolved_graph_path)

    if graph is None:
        if size is None:
            raise ValueError("Either 'graph' or 'size' must be provided.")
        if resolved_graph_path is not None:
            save_dgl_graph(graph, resolved_graph_path)
    elif resolved_graph_path is not None and not resolved_graph_path.exists():
        save_dgl_graph(graph, resolved_graph_path)

    if edge_types is not None:
        graph = dgl.edge_type_subgraph(graph, edge_types)

    print('Graph canonical edge types:')
    for etype in graph.canonical_etypes:
        print(f'  {etype}: {graph.num_edges(etype)} edges')

    # Store labels and masks
    train_mask = graph.nodes[task].data['train_mask'].bool()
    val_mask = graph.nodes[task].data['val_mask'].bool()
    test_mask = graph.nodes[task].data['test_mask'].bool()

    # Initialise dictionary with feature dimensions
    dims = {ntype: graph.nodes[ntype].data['feat'].shape[-1]
            for ntype in graph.ntypes}
    feat_dict = {rel: (dims[rel[0]], dims[rel[2]])
                 for rel in graph.canonical_etypes}

    # Initialise model
    model = HeteroGraphSAGE(input_dropout=0.2,
                            dropout=0.2,
                            hidden_dim=1024,
                            feat_dict=feat_dict,
                            task=task)
    model.to(device)
    model.train()

    # Enumerate the nodes with the labels, for performing train/val/test splits
    node_enum = torch.arange(graph.num_nodes(task))

    # If we are performing a random split then split the dataset into a
    # 80/10/10 train/val/test split, with a fixed random seed
    if random_split:

        # Set a random seed through a PyTorch Generator
        torch_gen = torch.Generator().manual_seed(4242)

        # Compute the number of train/val/test samples
        num_train = int(0.8 * graph.num_nodes(task))
        num_val = int(0.1 * graph.num_nodes(task))
        num_test = graph.num_nodes(task) - (num_train + num_val)
        nums = [num_train, num_val, num_test]

        # Split the data, using the PyTorch generator for reproducibility
        train_subset, val_subset, test_subset = D.random_split(dataset=node_enum,
                                                               lengths=nums,
                                                               generator=torch_gen)

        # Store the resulting node IDs
        train_nids = {task: node_enum[train_subset.indices].long()}
        val_nids = {task: node_enum[val_subset.indices].long()}
        test_nids = {task: node_enum[test_subset.indices].long()}

    # If we are not performing a random split we're performing a split based on
    # the claim clusters of the data. This means that the different splits will
    # belong to different events, thus making the task harder.
    else:
        train_nids = {task: node_enum[train_mask].long()}
        val_nids = {task: node_enum[val_mask].long()}
        test_nids = {task: node_enum[test_mask].long()}

    manual_pos_weight = _.get('pos_weight')
    effective_pos_weight = _compute_pos_weight(
        graph=graph,
        task=task,
        train_node_ids=train_nids[task],
        override=manual_pos_weight,
    )
    config['pos_weight'] = effective_pos_weight

    early_stopping_metric = _.get('early_stopping_metric', config['early_stopping_metric'])
    if early_stopping_metric not in {'pr_auc', 'class_1_f1'}:
        raise ValueError("early_stopping_metric must be either 'pr_auc' or 'class_1_f1'.")

    early_stopping_patience = _.get('early_stopping_patience', config['early_stopping_patience'])
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience < 1:
            raise ValueError('early_stopping_patience must be >= 1 or None.')

    early_stopping_min_delta = float(
        _.get('early_stopping_min_delta', config['early_stopping_min_delta'])
    )
    config['early_stopping_metric'] = early_stopping_metric
    config['early_stopping_patience'] = early_stopping_patience
    config['early_stopping_min_delta'] = early_stopping_min_delta

    # Set up the sampler
    sampler = MultiLayerNeighborSampler([100, 100], replace=False)

    # Set up the dataloaders
    train_dataloader = NodeDataLoader(graph,
                                      train_nids,
                                      sampler,
                                      batch_size=32,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=1)
    val_dataloader = NodeDataLoader(graph,
                                    val_nids,
                                    sampler,
                                    batch_size=1000000,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=1)
    test_dataloader = NodeDataLoader(graph,
                                     test_nids,
                                     sampler,
                                     batch_size=1000000,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=1)

    # Weight the positive class using the actual train-split imbalance by default.
    pos_weight_tensor = torch.tensor(effective_pos_weight, dtype=torch.float32, device=device)

    # Set up path to state dict
    run_timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    Path("models").mkdir(exist_ok=True)
    model_dir = Path("models") / f"{run_timestamp}-{task}-model-{num_epochs}"
    model_dir.mkdir(exist_ok=True)
    best_model_path = model_dir / 'best_model.pt'
    print(f'Starting training for task "{task}" at {run_timestamp}. Model directory: {model_dir}')
    print(f'Using pos_weight={effective_pos_weight:.6f} for class 1 on the train split.')
    if early_stopping_patience is not None:
        print(
            f'Using early stopping on val_{early_stopping_metric} '
            f'(patience={early_stopping_patience}, min_delta={early_stopping_min_delta:g}).'
        )

    # Initialise optimiser
    opt = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    # Initialise learning rate scheduler
    scheduler = LinearLR(optimizer=opt,
                         start_factor=1.,
                         end_factor=1e-7 / 3e-4,
                         total_iters=100)

    # Initialise scorer
    scorer = BinaryF1Tracker()
    metrics_rows = []
    best_epoch = 0
    best_metric_value = float('-inf')
    best_state_dict = None
    best_train_metrics = None
    best_val_metrics = None
    no_improvement_epochs = 0

    # Initialise progress bar
    epoch_pbar = tqdm(range(num_epochs))
    # epoch_pbar = tqdm(range(num_epochs), desc='Training epochs', unit='epoch')

    for epoch in epoch_pbar:

        # Reset metrics
        train_loss = 0.0
        val_loss = 0.0
        train_metrics = {}
        val_metrics = {}

        # Reset metrics
        scorer.reset()

        # Train model
        model.train()
        for _, _, blocks in train_dataloader:

            # Reset the gradients
            opt.zero_grad()

            # Ensure that `blocks` are on the correct device
            blocks = [block.to(device) for block in blocks]

            # Get the input features and the output labels
            input_feats = {n: feat.float()
                           for n, feat in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['label'][task].to(device)

            # Forward propagation
            logits = model(blocks, input_feats).squeeze()

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(
                input=logits,
                target=output_labels.float(),
                pos_weight=pos_weight_tensor,
            )

            # Compute training metrics
            scorer(logits.ge(0), output_labels, torch.sigmoid(logits))

            # Backward propagation
            loss.backward()

            # Update gradients
            opt.step()

            # Store the training loss
            train_loss += float(loss)

        # Divide the training loss by the number of batches
        train_loss /= len(train_dataloader)

        # Compute the training metrics
        train_metrics = _summarise_split_metrics(train_loss, scorer)

        # Evaluate validation split with fresh accumulators and eval/no_grad semantics.
        val_metrics = _evaluate_binary_split(
            model=model,
            dataloader=val_dataloader,
            task=task,
            device=device,
            pos_weight_tensor=pos_weight_tensor,
            scorer=scorer,
            split_name='validation',
        )
        val_loss = val_metrics['loss']

        # Gather statistics to be logged
        stats = [
            ('train_loss', train_loss),
            ('train_general_f1', train_metrics['general_f1']),
            ('train_class_0_f1', train_metrics['class_0_f1']),
            ('train_class_1_f1', train_metrics['class_1_f1']),
            ('train_precision', train_metrics['precision']),
            ('train_recall', train_metrics['recall']),
            ('train_specificity', train_metrics['specificity']),
            ('train_mcc', train_metrics['mcc']),
            ('train_pr_auc', train_metrics['pr_auc']),
            ('train_roc_auc', train_metrics['roc_auc']),
            ('train_tn', train_metrics['tn']),
            ('train_fp', train_metrics['fp']),
            ('train_fn', train_metrics['fn']),
            ('train_tp', train_metrics['tp']),
            ('val_loss', val_loss),
            ('val_general_f1', val_metrics['general_f1']),
            ('val_class_0_f1', val_metrics['class_0_f1']),
            ('val_class_1_f1', val_metrics['class_1_f1']),
            ('val_precision', val_metrics['precision']),
            ('val_recall', val_metrics['recall']),
            ('val_specificity', val_metrics['specificity']),
            ('val_mcc', val_metrics['mcc']),
            ('val_pr_auc', val_metrics['pr_auc']),
            ('val_roc_auc', val_metrics['roc_auc']),
            ('val_tn', val_metrics['tn']),
            ('val_fp', val_metrics['fp']),
            ('val_fn', val_metrics['fn']),
            ('val_tp', val_metrics['tp']),
            ('learning_rate', opt.param_groups[0]['lr'])
        ]

        # Report and log statistics
        config['epoch'] = epoch
        for statistic, value in stats:
            config[statistic] = value

        metrics_rows.extend([
            {
                'phase': 'epoch',
                'epoch': epoch + 1,
                'split': 'train',
                **train_metrics,
                'learning_rate': opt.param_groups[0]['lr']
            },
            {
                'phase': 'epoch',
                'epoch': epoch + 1,
                'split': 'val',
                **val_metrics,
                'learning_rate': opt.param_groups[0]['lr']
            }
        ])

        epoch_pbar.write(
            f'Epoch {epoch + 1}/{num_epochs} | '
            f'train_loss={train_loss:.3f} | '
            f'train_general_f1={train_metrics["general_f1"]:.3f} | '
            f'train_class_0_f1={train_metrics["class_0_f1"]:.3f} | '
            f'train_class_1_f1={train_metrics["class_1_f1"]:.3f} | '
            f'train_mcc={train_metrics["mcc"]:.3f} | '
            f'val_loss={val_loss:.3f} | '
            f'val_general_f1={val_metrics["general_f1"]:.3f} | '
            f'val_class_0_f1={val_metrics["class_0_f1"]:.3f} | '
            f'val_class_1_f1={val_metrics["class_1_f1"]:.3f} | '
            f'val_mcc={val_metrics["mcc"]:.3f} | '
            f'lr={opt.param_groups[0]["lr"]:.7f}'
        )

        current_metric_value = float(val_metrics[early_stopping_metric])
        improved = current_metric_value > (best_metric_value + early_stopping_min_delta)
        if improved:
            best_epoch = epoch + 1
            best_metric_value = current_metric_value
            best_state_dict = copy.deepcopy(model.state_dict())
            best_train_metrics = dict(train_metrics)
            best_val_metrics = dict(val_metrics)
            no_improvement_epochs = 0
            torch.save(
                {
                    'epoch': best_epoch,
                    'monitor_metric': early_stopping_metric,
                    'monitor_value': best_metric_value,
                    'model_state_dict': best_state_dict,
                },
                best_model_path,
            )
        else:
            no_improvement_epochs += 1

        # Update learning rate
        scheduler.step()

        if early_stopping_patience is not None and no_improvement_epochs >= early_stopping_patience:
            epoch_pbar.write(
                f'Early stopping at epoch {epoch + 1}/{num_epochs}; '
                f'best epoch was {best_epoch} with '
                f'val_{early_stopping_metric}={best_metric_value:.4f}.'
            )
            break

    # Close progress bar
    epoch_pbar.close()

    if best_state_dict is None:
        raise RuntimeError('Training completed without producing a best model checkpoint.')

    model.load_state_dict(best_state_dict)
    train_metrics = best_train_metrics if best_train_metrics is not None else train_metrics
    val_metrics = best_val_metrics if best_val_metrics is not None else val_metrics
    print(
        f'Restored best model from epoch {best_epoch} with '
        f'val_{early_stopping_metric}={best_metric_value:.4f}.'
    )

    # Reset loss
    val_loss = 0.0
    test_loss = 0.0

    val_metrics = _evaluate_binary_split(
        model=model,
        dataloader=val_dataloader,
        task=task,
        device=device,
        pos_weight_tensor=pos_weight_tensor,
        scorer=scorer,
        split_name='validation',
    )
    val_loss = val_metrics['loss']

    # Final evaluation on the test set
    model.eval()
    scorer.reset()
    test_inference_rows = []
    label_names = {
        0: "doesn't contain narrative",
        1: 'contains narrative'
    }
    for _, output_nodes, blocks in tqdm(test_dataloader, desc='Test', unit='batch'):
        with torch.no_grad():

            # Ensure that `blocks` are on the correct device
            blocks = [block.to(device) for block in blocks]

            # Get the input features and the output labels
            input_feats = {n: f.float()
                           for n, f in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['label'][task].to(device)

            # Forward propagation
            logits = model(blocks, input_feats).squeeze()

            # Compute test loss
            loss = F.binary_cross_entropy_with_logits(
                input=logits,
                target=output_labels.float(),
                pos_weight=pos_weight_tensor,
            )

            # Compute test metrics
            pred_labels = logits.ge(0).to(torch.int64)
            pred_scores = torch.sigmoid(logits)
            scorer(pred_labels, output_labels, pred_scores)

            # Store the test loss
            test_loss += float(loss)

            test_node_ids = output_nodes[task].detach().cpu().long()
            message_ids = _get_node_field_values(
                graph=graph,
                task=task,
                node_ids=test_node_ids,
                field_names=['message_id', 'channel_id'],
                default=lambda ids: ids.tolist()
            )
            channel_ids = _get_node_field_values(
                graph=graph,
                task=task,
                node_ids=test_node_ids,
                field_names=['channel_id'],
                default=''
            )

            test_inference_rows.extend([
                {
                    'message_id': message_id,
                    'channel_id': channel_id,
                    'label': label_names[int(pred_label)]
                }
                for message_id, channel_id, pred_label in zip(
                    message_ids,
                    channel_ids,
                    pred_labels.detach().cpu().view(-1).tolist()
                )
            ])

    # Divide the test loss by the number of batches
    test_loss /= len(test_dataloader)

    # Compute the test metrics
    test_metrics = _summarise_split_metrics(test_loss, scorer)

    # Gather statistics to be logged
    results = {
        'train': {
            **train_metrics
        },
        'val': {
            **val_metrics
        },
        'test': {
            **test_metrics
        }
    }

    metrics_rows.extend([
        {
            'phase': 'final',
            'epoch': '',
            'split': 'train',
            **train_metrics,
            'learning_rate': opt.param_groups[0]['lr']
        },
        {
            'phase': 'final',
            'epoch': '',
            'split': 'val',
            **val_metrics,
            'learning_rate': opt.param_groups[0]['lr']
        },
        {
            'phase': 'final',
            'epoch': '',
            'split': 'test',
            **test_metrics,
            'learning_rate': opt.param_groups[0]['lr']
        }
    ])

    _write_csv(
        model_dir / 'metrics.csv',
        fieldnames=[
            'phase',
            'epoch',
            'split',
            'loss',
            'general_f1',
            'class_0_f1',
            'class_1_f1',
            'precision',
            'recall',
            'specificity',
            'mcc',
            'tn',
            'fp',
            'fn',
            'tp',
            'pr_auc',
            'roc_auc',
            'learning_rate'
        ],
        rows=metrics_rows
    )
    _write_csv(
        model_dir / 'test_inference.csv',
        fieldnames=['message_id', 'channel_id', 'label'],
        rows=test_inference_rows
    )

    return results
