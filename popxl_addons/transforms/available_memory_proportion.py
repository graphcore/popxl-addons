# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable

import popxl
from popxl.ops.utils import convert_optional_float
import popxl_addons as addons
from popxl_addons.graph import GraphWithNamedArgs

__all__ = ["set_available_memory_proportion_by_ipu", "set_graph_available_memory_proportion_by_ipu"]


def set_available_memory_proportion_by_ipu(ir: popxl.Ir, proportions: Iterable[float]):
    """For all ops in the `ir`, if `availableMemoryProportion` can be set, set it as
        specified by `proportions`.
        
        If the available memory proportion has been set on a op site it will be overridden.

    Args:
        ir (popxl.Ir)
        proportions (List[float]): The availableMemoryProportion to be set on each ipu
                                   proportions[N] == 'proportion for ipu N'
    """
    ipu_to_prop = dict(enumerate(proportions))
    for g in ir._pb_ir.getAllGraphs():
        for op in g.getOps():
            if hasattr(op, "setAvailableMemoryProportion"):
                prop = ipu_to_prop.get(op.getVirtualGraphId(), ipu_to_prop[0])
                op.setAvailableMemoryProportion(convert_optional_float(prop))


def set_graph_available_memory_proportion_by_ipu(graph: popxl.Graph, proportions: Iterable[float]):
    """For all ops in the `graph`, if `availableMemoryProportion` can be set, set it as
        specified by `proportions`.
        
        If the available memory proportion has been set on a op site it will be overridden.

    Args:
        graph (popxl.Graph)
        proportions (List[float]): The availableMemoryProportion to be set on each ipu
                                   proportions[N] == 'proportion for ipu N'
    """
    visited = []
    _set_graph_available_memory_proportion_by_ipu(graph._pb_graph, proportions, visited)


def _set_graph_available_memory_proportion_by_ipu(pb_graph, proportions: Iterable[float], visited_graphs):
    ipu_to_prop = dict(enumerate(proportions))
    for op in pb_graph.getOps():
        if hasattr(op, "getCalledGraph"):
            called = op.getCalledGraph()
            if called not in visited_graphs:
                _set_graph_available_memory_proportion_by_ipu(called, proportions, visited_graphs)

        if hasattr(op, "setAvailableMemoryProportion"):
            prop = ipu_to_prop.get(op.getVirtualGraphId(), ipu_to_prop[0])
            op.setAvailableMemoryProportion(convert_optional_float(prop))

    visited_graphs.append(pb_graph)
