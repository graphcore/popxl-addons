# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Union, Mapping, Dict, TYPE_CHECKING

import popxl
from popxl.transforms.autodiff import GradGraphInfo

if TYPE_CHECKING:
    from popxl_addons.graph import GraphWithNamedArgs, BoundGraph


GraphLike = Union[popxl.Graph, "GraphWithNamedArgs", "BoundGraph"]
GradGraphInfoLike = Union[GradGraphInfo, "GraphWithNamedArgs"]

CalledGraphsGradInfoLike = Mapping[GraphLike, GradGraphInfoLike]
CalledGraphsGradInfo = Mapping[popxl.Graph, GradGraphInfo]


def _normalise_called_graphs_grad_info(
    fwd_graph: GraphLike, grad_graph: GradGraphInfoLike
) -> Tuple[popxl.Graph, GradGraphInfo]:
    """Normalise a (GraphLike, GradGraphInfoLike) association to (popxl.Graph, GradGraphInfo)"""
    fwd_graph = fwd_graph if isinstance(fwd_graph, popxl.Graph) else fwd_graph.graph
    grad_graph = grad_graph if isinstance(grad_graph, GradGraphInfo) else grad_graph.grad_graph_info
    return fwd_graph, grad_graph


def _normalise_called_graphs_grad_info_dict(
    called_graphs_grad_info: Optional[CalledGraphsGradInfoLike] = None,
) -> CalledGraphsGradInfo:
    """Normalise a (GraphLike, GradGraphInfoLike) dictionary to (popxl.Graph, GradGraphInfo)"""
    called_graphs_grad_info = called_graphs_grad_info if called_graphs_grad_info is not None else {}
    return dict(
        [
            _normalise_called_graphs_grad_info(fwd_graph, grad_graph)
            for fwd_graph, grad_graph in called_graphs_grad_info.items()
        ]
    )
