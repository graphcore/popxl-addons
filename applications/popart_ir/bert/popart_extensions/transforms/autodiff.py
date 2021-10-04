# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Optional
import numpy as np
import popart.ir as pir
from popart.ir.ops.call import CallInfo
from popart.ir.transforms.autodiff import (
    autodiff,
    GradGraphInfo,
    get_expected_forward_inputs_from_call)

import popart_extensions as pir_ext
from .gradient_accumulation import accumulate_gradients_in_graph


__all__ = ["autodiff_with_accumulation", "connect_activations"]


class ConcreteGradGraph(pir_ext.ConcreteGraph):
    def __init__(self, grad_info: GradGraphInfo, graph: pir.Graph):
        super().__init__(graph)
        self.grad_info = grad_info


def autodiff_with_accumulation(concrete_graph: pir_ext.ConcreteGraph,
                               tensors_to_accumulate_grads: Iterable[pir.Tensor]) -> ConcreteGradGraph:

    # Autodiff the graph.
    grad_info: GradGraphInfo = autodiff(concrete_graph.graph)  # type: ignore

    # Create a concrete graph for the grad graph.
    grad_graph = ConcreteGradGraph(grad_info, grad_info.graph)

    # Modify the graph to have accumulator inputs
    accumulators = accumulate_gradients_in_graph(
        grad_info,
        tensors_to_accumulate_grads)

    # Add variableDefs for each of the new accumulator inputs
    for accumulator in accumulators.values():
        # TODO: accumulator.name isn't going to be safe... need to sanitize.
        grad_graph[accumulator.name] = (
            pir_ext.variable_def(np.zeros(accumulator.shape, accumulator.dtype.as_numpy()), name=accumulator.name),  # type: ignore
            accumulator)

    return grad_graph


def connect_activations(forward_call_info: CallInfo,
                        grad_info: GradGraphInfo,  # TODO: try to combine grad_info and callable_grad_graph somehow.
                        callable_grad_graph: pir_ext.CallableGraph):
    activations = get_expected_forward_inputs_from_call(forward_call_info, grad_info)
    for sg_tensor, act in activations.items():
        callable_grad_graph[act.name] = (sg_tensor, act)
