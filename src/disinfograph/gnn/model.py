"""Heterogeneous GraphSAGE model for node classification on multigraphs."""

import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.disinfograph.gnn.heterographconv import HeteroGraphConv


class HeteroGraphSAGE(nn.Module):
    """Two-layer heterogeneous GraphSAGE with a linear classification head.

    Each convolution layer aggregates neighbour features via LSTM pooling and
    applies GELU activation.  A LayerNorm is applied between the two conv layers
    and again after the second.  The classification head consists of BatchNorm,
    two linear layers with GELU, and a final projection to a single logit.

    Args:
        input_dropout: Dropout probability applied to input features in the
            first conv layer.
        dropout: Dropout probability applied to hidden representations in the
            second conv layer and the classification head.
        hidden_dim: Dimensionality of the hidden representations.
        feat_dict: Mapping from canonical edge type
            ``(src_type, rel_type, dst_type)`` to a pair
            ``(src_feat_dim, dst_feat_dim)`` describing the input feature sizes
            for that relation.
        task: Node type whose final representation is passed to the classifier.
            Defaults to ``'message'``.
    """

    def __init__(
        self,
        input_dropout: float,
        dropout: float,
        hidden_dim: int,
        feat_dict: Dict[Tuple[str, str, str], Tuple[int, int]],
        task: str = "message",
    ):
        super().__init__()
        self.feat_dict = feat_dict
        self.hidden_dim = hidden_dim
        self.task = task

        self.conv1 = HeteroGraphConv(
            {
                rel: dglnn.SAGEConv(
                    in_feats=(feats[0], feats[1]),
                    out_feats=hidden_dim,
                    aggregator_type="lstm",
                    feat_drop=input_dropout,
                    activation=nn.GELU(),
                )
                for rel, feats in feat_dict.items()
            },
            aggregate="sum",
        )

        self.conv2 = HeteroGraphConv(
            {
                rel: dglnn.SAGEConv(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim,
                    aggregator_type="lstm",
                    feat_drop=dropout,
                    activation=nn.GELU(),
                )
                for rel, _ in feat_dict.items()
            },
            aggregate="sum",
        )

        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, blocks: List, h_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run two GraphSAGE convolutions and return task-node logits.

        Args:
            blocks: A list of two DGL message-flow graphs (MFGs) produced by
                a neighbour sampler — one per conv layer.
            h_dict: Dictionary mapping node type names to their feature tensors
                (shape ``[num_nodes, feat_dim]``).

        Returns:
            A tensor of shape ``[num_task_nodes, 1]`` containing raw logits for
            the task node type.
        """
        h_dict = self.conv1(blocks[0], h_dict)
        h_dict = {k: self.norm(v) for k, v in h_dict.items()}
        h_dict = self.conv2(blocks[1], h_dict)
        h_dict = {k: self.norm(v) for k, v in h_dict.items()}
        return self.clf(h_dict[self.task])
