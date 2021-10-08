# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Mapping, Optional
import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
from popart.ir.ops.call import CallInfo
from popart.ir.transforms.autodiff import (
    autodiff as _autodiff,
    GradGraphInfo,
    get_expected_forward_inputs_from_call)

import popart_extensions as pir_ext
from .gradient_accumulation import accumulate_gradients_in_graph


__all__ = ["autodiff", "autodiff_with_accumulation", "connect_activations"]


class ConcreteGradGraph(pir_ext.ConcreteGraph):
    def __init__(self, grad_info: GradGraphInfo, graph: pir.Graph):
        super().__init__(graph)
        self.grad_info = grad_info


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
    grad_info: GradGraphInfo = _autodiff(graph.graph, *args, **kwargs)  # type: ignore

    ir = grad_info.graph.ir()._pb_ir
    # TODO: Only run required patterns
    ir.setPatterns(_ir.patterns.Patterns(_ir.patterns.PatternsLevel.Default))
    ir.applyPreAliasPatterns(grad_info.graph._pb_graph)
    # TODO: Should inplacing be run?
    #       If not, we can end up with lots of outplace identities
    ir.applyInplacePattern(grad_info.graph._pb_graph)

    return ConcreteGradGraph(grad_info, grad_info.graph)


def sanitise_name(name: str) -> str:
    return name.replace(".", "_")


def autodiff_with_accumulation(concrete_graph: pir_ext.ConcreteGraph,
                               tensors_to_accumulate_grads: Iterable[pir.Tensor]) -> ConcreteGradGraph:

    # Autodiff the graph.
    grad_graph = autodiff(concrete_graph)

    # Modify the graph to have accumulator inputs
    accumulators = accumulate_gradients_in_graph(
        grad_graph.grad_info,
        tensors_to_accumulate_grads)

    # Add variableDefs for each of the new accumulator inputs
    for accumulator in accumulators.values():
        # TODO: accumulator.name isn't going to be safe... need to sanitize.
        grad_graph[sanitise_name(accumulator.name)] = (
            pir_ext.variable_def(np.zeros(accumulator.shape, accumulator.dtype.as_numpy()), name=accumulator.name),  # type: ignore
            accumulator)

    return grad_graph


def connect_activations(forward_call_info: CallInfo,
                        grad_info: GradGraphInfo,  # TODO: try to combine grad_info and callable_grad_graph somehow.
                        callable_grad_graph: pir_ext.CallableGraph):
    activations = get_expected_forward_inputs_from_call(forward_call_info, grad_info)
    for sg_tensor, act in activations.items():
        callable_grad_graph[sanitise_name(act.name)] = (sg_tensor, act)
