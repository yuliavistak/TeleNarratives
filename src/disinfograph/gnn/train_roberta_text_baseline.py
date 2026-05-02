"""Fine-tune a RoBERTa-style text classifier for the binary message task."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import importlib.metadata as importlib_metadata
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
GNN_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, GNN_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


METRIC_FIELDNAMES = [
    "phase",
    "epoch",
    "split",
    "loss",
    "general_f1",
    "class_0_f1",
    "class_1_f1",
    "precision",
    "recall",
    "specificity",
    "mcc",
    "tn",
    "fp",
    "fn",
    "tp",
    "pr_auc",
    "roc_auc",
    "learning_rate",
]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_training_libs():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torch.utils.data as torch_data
        from torch.optim.lr_scheduler import LinearLR
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RoBERTa text training requires torch in the notebook environment."
        ) from exc

    try:
        transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "RoBERTa text training requires transformers in the notebook environment."
        ) from exc

    torch_version_text = torch.__version__.split("+", 1)[0]
    torch_version_parts = tuple(int(part) for part in torch_version_text.split(".")[:2])
    transformers_major = int(transformers_version.split(".", 1)[0])
    if transformers_major >= 5 and torch_version_parts < (2, 4):
        raise RuntimeError(
            "The installed transformers package will disable PyTorch because this notebook "
            f"has torch {torch.__version__} and transformers {transformers_version}. "
            "Either install torch >= 2.4 or install transformers<5, then restart the notebook kernel."
        )

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RoBERTa text training requires transformers in the notebook environment."
        ) from exc

    return torch, nn, optim, torch_data, LinearLR, AutoModelForSequenceClassification, AutoTokenizer


def _iter_progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, **kwargs)


class BinaryMetricTracker:
    """Binary metrics with the same output columns as the GNN trainer."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.tp = np.zeros(2, dtype=np.float64)
        self.fp = np.zeros(2, dtype=np.float64)
        self.fn = np.zeros(2, dtype=np.float64)
        self.score_chunks = []
        self.target_chunks = []

    def update(self, preds, targets, scores=None) -> None:
        preds = np.asarray(preds).reshape(-1).astype(np.int64)
        targets = np.asarray(targets).reshape(-1).astype(np.int64)
        if scores is not None:
            self.score_chunks.append(np.asarray(scores).reshape(-1).astype(np.float64))
            self.target_chunks.append(targets.astype(np.float64))

        for cls in (0, 1):
            pred_is_cls = preds == cls
            target_is_cls = targets == cls
            self.tp[cls] += np.logical_and(pred_is_cls, target_is_cls).sum()
            self.fp[cls] += np.logical_and(pred_is_cls, ~target_is_cls).sum()
            self.fn[cls] += np.logical_and(~pred_is_cls, target_is_cls).sum()

    def _class_f1(self) -> np.ndarray:
        precision = self.tp / np.clip(self.tp + self.fp, 1.0, None)
        recall = self.tp / np.clip(self.tp + self.fn, 1.0, None)
        return 2 * precision * recall / np.clip(precision + recall, 1.0, None)

    def _general_f1(self) -> float:
        precision = self.tp.sum() / max(self.tp.sum() + self.fp.sum(), 1.0)
        recall = self.tp.sum() / max(self.tp.sum() + self.fn.sum(), 1.0)
        return float(2 * precision * recall / max(precision + recall, 1.0))

    def _scores_and_targets(self):
        if not self.score_chunks:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        return np.concatenate(self.score_chunks), np.concatenate(self.target_chunks)

    def _roc_auc(self) -> float:
        scores, targets = self._scores_and_targets()
        if scores.size == 0:
            return 0.0
        positives = targets.sum()
        negatives = targets.size - positives
        if positives == 0 or negatives == 0:
            return 0.0

        order = np.argsort(-scores)
        sorted_scores = scores[order]
        sorted_targets = targets[order]
        distinct = np.where(sorted_scores[1:] != sorted_scores[:-1])[0]
        threshold_idxs = np.concatenate([distinct, np.asarray([sorted_scores.size - 1])])
        true_positives = np.cumsum(sorted_targets)[threshold_idxs]
        false_positives = (threshold_idxs.astype(np.float64) + 1.0) - true_positives
        tpr = np.concatenate([[0.0], true_positives / positives, [1.0]])
        fpr = np.concatenate([[0.0], false_positives / negatives, [1.0]])
        return float(np.trapz(tpr, fpr))

    def _pr_auc(self) -> float:
        scores, targets = self._scores_and_targets()
        if scores.size == 0:
            return 0.0
        positives = targets.sum()
        if positives == 0:
            return 0.0

        order = np.argsort(-scores)
        sorted_targets = targets[order]
        true_positives = np.cumsum(sorted_targets)
        false_positives = np.cumsum(1.0 - sorted_targets)
        precision = true_positives / np.clip(true_positives + false_positives, 1.0, None)
        recall = true_positives / positives
        recall_prev = np.concatenate([[0.0], recall[:-1]])
        return float(((recall - recall_prev) * precision).sum())

    def compute(self, loss: float) -> Dict[str, float]:
        class_f1 = self._class_f1()
        tn = float(self.tp[0])
        fp = float(self.fp[1])
        fn = float(self.fn[1])
        tp = float(self.tp[1])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        mcc_denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
        mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0.0 else 0.0

        return {
            "loss": float(loss),
            "general_f1": self._general_f1(),
            "class_0_f1": float(class_f1[0]),
            "class_1_f1": float(class_f1[1]),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "mcc": mcc,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "pr_auc": self._pr_auc(),
            "roc_auc": self._roc_auc(),
        }


def _resolve_label_column(df: pd.DataFrame, label_col: str) -> str:
    candidates = [label_col, "message_label", "binary_label", "target"]
    for candidate in candidates:
        if candidate in df.columns and df[candidate].notna().any():
            return candidate
    raise ValueError(
        f"Could not find a usable label column. Tried: {', '.join(dict.fromkeys(candidates))}."
    )


def _resolve_channel_column(df: pd.DataFrame, channel_col: Optional[str]) -> Optional[str]:
    candidates = [channel_col, "channel_id", "channel_username", "username", "channel"]
    for candidate in candidates:
        if candidate and candidate in df.columns:
            return candidate
    return None


def _normalise_training_frame(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    date_col: str,
    channel_col: Optional[str],
    message_id_col: str,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' is missing.")

    resolved_label_col = _resolve_label_column(df, label_col)
    resolved_channel_col = _resolve_channel_column(df, channel_col)
    rows = []

    for row_idx, row in df.reset_index(drop=True).iterrows():
        raw_label = row.get(resolved_label_col)
        if pd.isna(raw_label):
            continue

        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue

        text_value = row.get(text_col)
        text = "" if pd.isna(text_value) else str(text_value)
        channel_value = row.get(resolved_channel_col) if resolved_channel_col else "unknown"
        if pd.isna(channel_value):
            channel_value = "unknown"

        rows.append(
            {
                "row_id": row_idx,
                "text": text,
                "label": label,
                "channel": channel_value,
                "date_utc": row.get(date_col) if date_col in df.columns else pd.NaT,
                "message_id": row.get(message_id_col) if message_id_col in df.columns else row_idx,
                "channel_id": row.get("channel_id") if "channel_id" in df.columns else channel_value,
            }
        )

    if not rows:
        raise ValueError("No usable rows remained after filtering for text and binary labels.")

    prepared = pd.DataFrame(rows)
    labels = set(prepared["label"].astype(int).unique().tolist())
    if labels != {0, 1}:
        raise ValueError(f"Both binary classes are required for training; found labels {sorted(labels)}.")

    timestamps = pd.to_datetime(prepared["date_utc"], errors="coerce", utc=True)
    fallback_timestamp = (
        timestamps.dropna().min()
        if timestamps.notna().any()
        else pd.Timestamp("1970-01-01", tz="UTC")
    )
    timestamps = timestamps.fillna(fallback_timestamp)
    iso_calendar = timestamps.dt.isocalendar()
    prepared["year"] = iso_calendar.year.astype("int64")
    prepared["week"] = iso_calendar.week.astype("int64")
    prepared["date_utc"] = timestamps
    return prepared


def _choose_split_cut(
    labels: np.ndarray,
    time_buckets: np.ndarray,
    target_count: float,
    target_pos_rate: float,
    rng: np.random.Generator,
) -> int:
    """Return the number of items to assign to the current split.

    Mirrors ``_choose_split_cut`` from ``gnn/dgl_graph/neo4j_export.py``.
    Prefers ISO-week boundaries, minimises deviation from the target count
    and from the global positive rate.
    """
    n_items = len(labels)
    if n_items == 0:
        return 0

    cuts = np.arange(n_items + 1, dtype=np.int64)
    week_boundary = np.zeros(n_items + 1, dtype=bool)
    week_boundary[[0, n_items]] = True
    if n_items > 1:
        week_boundary[np.where(time_buckets[1:] != time_buckets[:-1])[0] + 1] = True

    prefix_pos = np.concatenate(([0], np.cumsum(labels, dtype=np.int64)))
    prefix_rates = np.full(n_items + 1, target_pos_rate, dtype=np.float64)
    non_empty = cuts > 0
    prefix_rates[non_empty] = prefix_pos[non_empty] / cuts[non_empty]

    count_score = np.abs(cuts - target_count)
    label_weight = min(max(n_items * 0.1, 1.0), 10.0)
    label_score = np.abs(prefix_rates - target_pos_rate) * label_weight
    boundary_penalty = np.where(week_boundary, 0.0, 0.25)
    tie_break = rng.random(n_items + 1) * 1e-6
    total_score = count_score + label_score + boundary_penalty + tie_break
    return int(cuts[np.argmin(total_score)])


def add_chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 4242,
) -> pd.DataFrame:
    """Add a ``split`` column using the same temporal strategy as the GNN pipeline.

    Mirrors ``_build_stratified_masks`` from ``gnn/dgl_graph/neo4j_export.py``:
    messages are grouped by channel, sorted chronologically within each group,
    and split at ISO-week-aligned boundaries that approximate the target
    fractions while preserving the global positive rate.

    The input DataFrame is expected to have already been deduplicated (e.g. with
    ``deduplicate_forwarded_messages``) so that forwarded copies do not appear
    in a different split from their source.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0.")

    split_df = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(random_seed)

    # Reuse pre-computed ISO year/week from _normalise_training_frame; derive
    # them here if they are absent.
    if "year" in split_df.columns and "week" in split_df.columns:
        time_bucket = (
            split_df["year"].to_numpy(dtype=np.int64) * 100
            + split_df["week"].to_numpy(dtype=np.int64)
        ).astype(np.int32)
    else:
        ts = pd.to_datetime(split_df["date_utc"], errors="coerce", utc=True)
        iso = ts.dt.isocalendar()
        time_bucket = (
            iso.year.to_numpy(dtype=np.int64) * 100
            + iso.week.to_numpy(dtype=np.int64)
        ).astype(np.int32)

    split_df["_time_bucket"] = time_bucket
    split_df["_tie_break"] = rng.random(len(split_df))

    split_values = np.full(len(split_df), "", dtype=object)
    target_pos_rate = float(split_df["label"].mean()) if len(split_df) > 0 else 0.0
    remaining_frac = val_frac + test_frac
    val_share_of_remaining = val_frac / remaining_frac if remaining_frac > 0.0 else 0.0

    for _, channel_group in split_df.groupby("channel_id", sort=False):
        ordered = channel_group.sort_values(
            by=["_time_bucket", "date_utc", "_tie_break", "message_id"],
            kind="stable",
        )
        positions = ordered.index.to_numpy(dtype=np.int64)
        channel_labels = split_df.loc[positions, "label"].to_numpy(dtype=np.int64)
        channel_buckets = split_df.loc[positions, "_time_bucket"].to_numpy(dtype=np.int32)
        n_channel = len(positions)

        train_cut = _choose_split_cut(
            labels=channel_labels,
            time_buckets=channel_buckets,
            target_count=train_frac * n_channel,
            target_pos_rate=target_pos_rate,
            rng=rng,
        )
        val_cut_rel = _choose_split_cut(
            labels=channel_labels[train_cut:],
            time_buckets=channel_buckets[train_cut:],
            target_count=val_share_of_remaining * max(n_channel - train_cut, 0),
            target_pos_rate=target_pos_rate,
            rng=rng,
        )
        val_cut = train_cut + val_cut_rel

        split_values[positions[:train_cut]] = "train"
        split_values[positions[train_cut:val_cut]] = "val"
        split_values[positions[val_cut:]] = "test"

    split_df["split"] = split_values
    if (split_df["split"] == "").any():
        raise RuntimeError("Split assignment failed: some rows were not assigned a split.")

    split_df = split_df.drop(columns=["_time_bucket", "_tie_break"])
    return split_df


def _make_loader(torch, torch_data, tokenizer, texts, labels, max_length: int, batch_size: int, shuffle: bool):
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    labels_tensor = torch.tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
    dataset = torch_data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        labels_tensor,
    )
    return torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _evaluate_split(torch, model, dataloader, criterion, device) -> Dict[str, float]:
    model.eval()
    tracker = BinaryMetricTracker()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.view(-1)
            loss = criterion(logits.float(), labels.float())
            scores = torch.sigmoid(logits)
            preds = logits.ge(0).to(torch.int64)
            n_items = int(labels.numel())
            total_loss += float(loss.item()) * n_items
            total_items += n_items
            tracker.update(
                preds.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                scores.detach().cpu().numpy(),
            )

    if total_items == 0:
        raise RuntimeError("No batches were produced for evaluation.")
    return tracker.compute(total_loss / total_items)


def train_roberta_baseline(
    df: pd.DataFrame,
    task: str = "message",
    model_name: str = "xlm-roberta-base",
    text_col: str = "text",
    label_col: str = "label",
    date_col: str = "date_utc",
    channel_col: Optional[str] = None,
    message_id_col: str = "message_id",
    output_dir: Path | str = "models",
    num_epochs: int = 3,
    batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    max_length: int = 256,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    # use_mixed_precision: bool = True,
    gradient_checkpointing: bool = False,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 4242,
    pos_weight: Optional[float] = None,
    early_stopping_metric: str = "pr_auc",
    early_stopping_patience: Optional[int] = 2,
    early_stopping_min_delta: float = 1e-4,
    freeze_encoder: bool = False,
    **_,
) -> Dict[str, Dict[str, float]]:
    """Fine-tune RoBERTa using tokenized text and binary labels from a dataframe."""
    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if eval_batch_size is None:
        eval_batch_size = batch_size
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be >= 1.")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if early_stopping_metric not in {"pr_auc", "class_1_f1"}:
        raise ValueError("early_stopping_metric must be either 'pr_auc' or 'class_1_f1'.")

    (
        torch,
        nn,
        optim,
        torch_data,
        LinearLR,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    ) = _require_training_libs()

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    prepared = _normalise_training_frame(
        df=df,
        text_col=text_col,
        label_col=label_col,
        date_col=date_col,
        channel_col=channel_col,
        message_id_col=message_id_col,
    )
    prepared = add_chronological_split(
        prepared,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_seed=random_seed,
    ).reset_index(drop=True)

    split_indices = {
        split_name: prepared.index[prepared["split"] == split_name].to_numpy(dtype=np.int64)
        for split_name in ("train", "val", "test")
    }
    for split_name, indices in split_indices.items():
        if len(indices) == 0:
            raise ValueError(f"The {split_name} split is empty; adjust the split fractions.")

    train_labels = prepared.loc[split_indices["train"], "label"].to_numpy(dtype=np.int64)
    positives = int((train_labels == 1).sum())
    negatives = int((train_labels == 0).sum())
    effective_pos_weight = (
        float(pos_weight)
        if pos_weight is not None
        else (negatives / positives if positives and negatives else 1.0)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    if gradient_checkpointing and not freeze_encoder:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if freeze_encoder:
        base_model = getattr(model, model.base_model_prefix, None)
        if base_model is not None:
            for parameter in base_model.parameters():
                parameter.requires_grad = False

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(effective_pos_weight, dtype=torch.float32, device=device)
    )
    opt = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = LinearLR(
        optimizer=opt,
        start_factor=1.0,
        end_factor=1e-7 / lr,
        total_iters=max(1, min(100, num_epochs)),
    )

    dataloaders = {}
    for split_name in ("train", "val", "test"):
        split_frame = prepared.loc[split_indices[split_name]]
        dataloaders[split_name] = _make_loader(
            torch=torch,
            torch_data=torch_data,
            tokenizer=tokenizer,
            texts=split_frame["text"].tolist(),
            labels=split_frame["label"].tolist(),
            max_length=max_length,
            batch_size=batch_size if split_name == "train" else eval_batch_size,
            shuffle=(split_name == "train"),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datetime = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_dir = output_dir / f"{datetime}-{task}-roberta-text-baseline-{num_epochs}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_model.pt"
    best_transformer_dir = model_dir / "best_model"

    split_export = prepared[["message_id", "channel_id", "date_utc", "label", "split"]].copy()
    split_export.to_csv(model_dir / "split_samples.csv", index=False)

    config = {
        "task": task,
        "model_name": model_name,
        "text_col": text_col,
        "label_col": label_col,
        "max_length": max_length,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_mixed_precision": False,
        "gradient_checkpointing": gradient_checkpointing,
        "num_epochs": num_epochs,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "random_seed": random_seed,
        "pos_weight": effective_pos_weight,
        "early_stopping_metric": early_stopping_metric,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "freeze_encoder": freeze_encoder,
        "split_counts": {name: int(len(indices)) for name, indices in split_indices.items()},
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Using device: {device}")
    print(f"Starting RoBERTa text baseline at {datetime}. Model directory: {model_dir}")
    print(f"Using tokenizer/model: {model_name}")
    if gradient_accumulation_steps > 1:
        print(f"Using gradient_accumulation_steps={gradient_accumulation_steps}.")
    if gradient_checkpointing and not freeze_encoder:
        print("Using gradient checkpointing.")
    print(f"Using pos_weight={effective_pos_weight:.6f} for class 1 on the train split.")
    print(f"Split counts: {config['split_counts']}")

    metrics_rows = []
    best_epoch = 0
    best_metric_value = float("-inf")
    best_state_dict = None
    best_train_metrics = None
    best_val_metrics = None
    no_improvement_epochs = 0

    for epoch in _iter_progress(range(num_epochs), desc="Training epochs", unit="epoch"):
        model.train()
        train_tracker = BinaryMetricTracker()
        train_loss = 0.0
        train_items = 0
        opt.zero_grad(set_to_none=True)

        for batch_index, (input_ids, attention_mask, labels) in enumerate(dataloaders["train"]):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.view(-1)
            loss = criterion(logits.float(), labels.float())
            (loss / gradient_accumulation_steps).backward()

            should_step = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or (batch_index + 1) == len(dataloaders["train"])
            )
            if should_step:
                opt.step()
                opt.zero_grad(set_to_none=True)

            scores = torch.sigmoid(logits)
            preds = logits.ge(0).to(torch.int64)
            n_items = int(labels.numel())
            train_loss += float(loss.item()) * n_items
            train_items += n_items
            train_tracker.update(
                preds.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                scores.detach().cpu().numpy(),
            )

        train_metrics = train_tracker.compute(train_loss / train_items)
        val_metrics = _evaluate_split(torch, model, dataloaders["val"], criterion, device)
        current_lr = float(opt.param_groups[0]["lr"])

        metrics_rows.extend([
            {
                "phase": "epoch",
                "epoch": epoch + 1,
                "split": "train",
                **train_metrics,
                "learning_rate": current_lr,
            },
            {
                "phase": "epoch",
                "epoch": epoch + 1,
                "split": "val",
                **val_metrics,
                "learning_rate": current_lr,
            },
        ])

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_metrics['loss']:.3f} | "
            f"train_class_1_f1={train_metrics['class_1_f1']:.3f} | "
            f"train_mcc={train_metrics['mcc']:.3f} | "
            f"val_loss={val_metrics['loss']:.3f} | "
            f"val_class_1_f1={val_metrics['class_1_f1']:.3f} | "
            f"val_mcc={val_metrics['mcc']:.3f} | "
            f"lr={current_lr:.7f}"
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
                    "epoch": best_epoch,
                    "monitor_metric": early_stopping_metric,
                    "monitor_value": best_metric_value,
                    "model_state_dict": best_state_dict,
                    "config": config,
                },
                best_model_path,
            )
            model.save_pretrained(best_transformer_dir)
            tokenizer.save_pretrained(best_transformer_dir)
        else:
            no_improvement_epochs += 1

        scheduler.step()

        if early_stopping_patience is not None and no_improvement_epochs >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1}/{num_epochs}; "
                f"best epoch was {best_epoch} with "
                f"val_{early_stopping_metric}={best_metric_value:.4f}."
            )
            break

    if best_state_dict is None:
        raise RuntimeError("Training completed without producing a best model checkpoint.")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.load_state_dict(best_state_dict)
    train_metrics = best_train_metrics or _evaluate_split(
        torch, model, dataloaders["train"], criterion, device
    )
    val_metrics = best_val_metrics or _evaluate_split(
        torch, model, dataloaders["val"], criterion, device
    )
    test_metrics = _evaluate_split(torch, model, dataloaders["test"], criterion, device)
    final_lr = float(opt.param_groups[0]["lr"])

    metrics_rows.extend([
        {"phase": "final", "epoch": "", "split": "train", **train_metrics, "learning_rate": final_lr},
        {"phase": "final", "epoch": "", "split": "val", **val_metrics, "learning_rate": final_lr},
        {"phase": "final", "epoch": "", "split": "test", **test_metrics, "learning_rate": final_lr},
    ])
    _write_csv(model_dir / "metrics.csv", METRIC_FIELDNAMES, metrics_rows)

    label_names = {0: "doesn't contain narrative", 1: "contains narrative"}
    test_inference_rows = []
    model.eval()
    with torch.no_grad():
        test_frame = prepared.loc[split_indices["test"]].reset_index(drop=True)
        for batch_idx, (input_ids, attention_mask, _) in enumerate(
            _iter_progress(dataloaders["test"], desc="Test", unit="batch")
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.view(-1)
            pred_labels = logits.ge(0).to(torch.int64).detach().cpu().numpy().tolist()
            start = batch_idx * eval_batch_size
            rows = test_frame.iloc[start:start + len(pred_labels)]
            for (_, row), pred_label in zip(rows.iterrows(), pred_labels):
                test_inference_rows.append(
                    {
                        "message_id": row["message_id"],
                        "channel_id": row["channel_id"],
                        "label": label_names[int(pred_label)],
                    }
                )

    _write_csv(
        model_dir / "test_inference.csv",
        fieldnames=["message_id", "channel_id", "label"],
        rows=test_inference_rows,
    )

    print(
        f"Restored best model from epoch {best_epoch} with "
        f"val_{early_stopping_metric}={best_metric_value:.4f}."
    )
    print(f"Wrote metrics to {model_dir / 'metrics.csv'}")
    print(f"Wrote test inference to {model_dir / 'test_inference.csv'}")

    return {"train": train_metrics, "val": val_metrics, "test": test_metrics}


def load_neo4j_message_frame(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
) -> pd.DataFrame:
    """Return the Neo4j Message frame accepted by train_roberta_baseline."""
    from disinfograph.gnn.neo4j_export import load_training_frames_from_neo4j
    from src.disinfograph.config import get_neo4j_config

    config = get_neo4j_config()
    frames = load_training_frames_from_neo4j(
        uri=uri or config["uri"],
        username=username or config["username"],
        password=password or config["password"],
        database=database or config["database"],
    )
    return frames["messages"]
