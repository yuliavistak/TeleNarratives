"""
Adapted from mumin-build by Dan Saattrup Nielsen and Ryan McConville
Original: https://github.com/MuMiN-dataset/mumin-baseline/blob/main/src/heterographconv.py
License: MIT

Custom HeteroGraphConv with canonical edge-type key compatibility.
"""

from functools import partial

import torch
import torch.nn as nn
from dgl import DGLError


class HeteroGraphConv(nn.Module):
    """Heterogeneous graph convolution with canonical edge-type key support.

    Adapted from ``dglnn.HeteroGraphConv`` to accept canonical edge-type
    tuples ``(src_type, rel_type, dst_type)`` as dictionary keys when
    constructing the per-relation module map.

    Args:
        mods: Mapping from canonical edge type to the per-relation convolution
            module.
        aggregate: Aggregation strategy for combining representations produced
            by different edge types into the same destination node type.
            One of ``'sum'``, ``'max'``, ``'min'``, ``'mean'``, or ``'stack'``.
            Defaults to ``'sum'``.
    """

    def __init__(self, mods, aggregate="sum"):
        super().__init__()
        self.mods = nn.ModuleDict(
            {"_".join(etype): val for etype, val in mods.items()}
        )
        # Allow zero-in-degree nodes — there is no general self-loop rule for
        # heterogeneous graphs.
        for _, module in self.mods.items():
            set_allow_zero = getattr(module, "set_allow_zero_in_degree", None)
            if callable(set_allow_zero):
                set_allow_zero(True)

        if isinstance(aggregate, str):
            self.agg_fn = self.get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, graph, inputs, mod_args=None, mod_kwargs=None):
        """Apply per-relation convolutions and aggregate destination node representations.

        Args:
            graph: A DGL heterogeneous graph or message-flow graph block.
            inputs: Input node features — either a dict mapping node type to
                feature tensor, or a pair ``(src_dict, dst_dict)`` for blocks.
            mod_args: Optional mapping from relation type to extra positional
                arguments forwarded to the per-relation module.
            mod_kwargs: Optional mapping from relation type to extra keyword
                arguments forwarded to the per-relation module.

        Returns:
            A dict mapping destination node type to the aggregated output tensor.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}

        outputs = {nty: [] for nty in graph.dsttypes}

        if isinstance(inputs, tuple) or graph.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[:graph.number_of_dst_nodes(k)] for k, v in inputs.items()
                }

            for stype, etype, dtype in graph.canonical_etypes:
                rel_graph = graph[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[f"{stype}_{etype}_{dtype}"](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}),
                )
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in graph.canonical_etypes:
                rel_graph = graph[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[f"{stype}_{etype}_{dtype}"](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}),
                )
                outputs[dtype].append(dstdata)

        return {
            nty: self.agg_fn(alist, nty)
            for nty, alist in outputs.items()
            if alist
        }

    def get_aggregate_fn(self, agg: str):
        """Return an aggregation callable for the given strategy name.

        Args:
            agg: One of ``'sum'``, ``'max'``, ``'min'``, ``'mean'``,
                or ``'stack'``.

        Returns:
            A callable that accepts a list of tensors and a destination node
            type string, and returns a single aggregated tensor.

        Raises:
            DGLError: If *agg* is not one of the supported strategies.
        """
        if agg == "sum":
            fn = self._sum_reduce_func
        elif agg == "max":
            fn = self._max_reduce_func
        elif agg == "min":
            fn = self._min_reduce_func
        elif agg == "mean":
            fn = self._mean_reduce_func
        elif agg == "stack":
            return self._stack_agg_func
        else:
            raise DGLError(
                f'Invalid cross type aggregator. Must be one of '
                f'"sum", "max", "min", "mean" or "stack". But got "{agg}"'
            )
        return partial(self._agg_func, fn=fn)

    @staticmethod
    def _max_reduce_func(inputs, dim):
        return torch.max(inputs, dim=dim)[0]

    @staticmethod
    def _min_reduce_func(inputs, dim):
        return torch.min(inputs, dim=dim)[0]

    @staticmethod
    def _sum_reduce_func(inputs, dim):
        return torch.sum(inputs, dim=dim)

    @staticmethod
    def _mean_reduce_func(inputs, dim):
        return torch.mean(inputs, dim=dim)

    @staticmethod
    def _stack_agg_func(inputs, dsttype):  # pylint: disable=unused-argument
        if not inputs:
            return None
        return torch.stack(inputs, dim=1)

    @staticmethod
    def _agg_func(inputs, dsttype, fn):  # pylint: disable=unused-argument
        if not inputs:
            return None
        stacked = torch.stack(inputs, dim=0)
        return fn(stacked, dim=0)
