# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Iterable, Optional

import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.ops.call import CallInfo
from popart.ir.transforms.autodiff import (
    autodiff as _autodiff,
    GradGraphInfo,
    get_expected_forward_inputs_from_call)

import popart_extensions as pir_ext
from popart_extensions.tuple_map import sanitise

__all__ = ["autodiff", "autodiff_with_accumulation", "connect_activations"]


class ConcreteGradGraph(pir_ext.ConcreteGraph):
    def __init__(self):
        super().__init__()
        self.grad_info: GradGraphInfo


def autodiff(
        graph: pir_ext.ConcreteGraph,
        *args,
        **kwargs) -> ConcreteGradGraph:
    """Extension Autodiff.
        This method calls pir.transforms.autodiff and then some required patterns after to ensure the returned
        grad graph is lowerable.

    Args:
        graph (pir.Graph)
        gradsProvided (Optional[Iterable[pir.Tensor]], optional). Defaults to all outputs of the provided graph.
        gradsRequired (Optional[Iterable[pir.Tensor]], optional). Defaults to all inputs of the provided graph.
        calledGraphsGradInfo (Optional[Mapping[pir.Graph, GradGraphInfo]], optional). Defaults to None.
        returnAllGradGraphs (bool, optional). Defaults to False.

    Returns:
        GradGraphInfo: result of pir.transforms.autodiff
    """
    grad_info: GradGraphInfo = _autodiff(graph, *args, **kwargs)  # type: ignore

    ir = grad_info.graph.ir()._pb_ir
    # TODO: Only run required patterns
    ir.setPatterns(_ir.patterns.Patterns(_ir.patterns.PatternsLevel.Default))
    ir.applyPreAliasPatterns(grad_info.graph._pb_graph)
    # TODO: Should inplacing be run?
    #       If not, we can end up with lots of outplace identities
    ir.applyInplacePattern(grad_info.graph._pb_graph)

    grad_graph = ConcreteGradGraph._from_pb(grad_info.graph._pb_graph)
    grad_graph.grad_info = grad_info
    return grad_graph


def connect_activations(forward_call_info: CallInfo,
                        grad_info: GradGraphInfo,  # TODO: try to combine grad_info and callable_grad_graph somehow.
                        callable_grad_graph: pir_ext.CallableGraph):
    activations = get_expected_forward_inputs_from_call(forward_call_info, grad_info)
    for sg_tensor, act in activations.items():
        callable_grad_graph[sanitise(act.name)] = (sg_tensor, act)


def autodiff_with_accumulation(concrete_graph: pir_ext.ConcreteGraph,
                               tensors_to_accumulate_grads: Iterable[pir.Tensor]) -> ConcreteGradGraph:
    # Autodiff the graph.
    grad_graph = autodiff(concrete_graph)

    # Modify the graph to have accumulator inputs
    accumulate_gradients_in_graph(
        grad_graph,
        tensors_to_accumulate_grads)

    return grad_graph


def accumulate_gradients_in_graph(graph: ConcreteGradGraph,
                                  tensors_to_accumulate_grads: Iterable[pir.Tensor],
                                  accum_type: Optional[pir.dtype] = None) -> Dict[pir.Tensor, pir.Tensor]:
    """Replace the outputs in grad graph that represent a gradient of a tensor in 'tensors_to_accumulate_grads'
        Adds a new input to the grad graph and an accumulate op in the grad graph.

       Returns a mapping from tensors in 'tensors_to_accumulate_grads' to the new subgraph_input"""
    grad_info = graph.grad_info

    expected_outputs = grad_info.get_output_tensors()

    variables: Dict[pir.Tensor, pir.Tensor] = {}

    for tensor in tensors_to_accumulate_grads:
        idx = expected_outputs.index(tensor)
        subgraph_tensor = pir.Tensor._from_pb_tensor(graph._pb_graph.getOutputTensor(idx))
        graph._pb_graph.removeOutput(idx)

        accum_type = tensor.dtype if accum_type is None else accum_type

        with graph:
            accum = graph.add_input_tensor(
                lambda: np.zeros(tensor.shape, accum_type.as_numpy()), "Accum__" + tensor.name)
            ops.accumulate(accum, subgraph_tensor)

        variables[tensor] = accum

    return variables
