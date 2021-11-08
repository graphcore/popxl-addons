# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Iterable, Optional, Mapping, Union

import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.ops.call import CallInfo
from popart.ir.transforms.autodiff import (autodiff as _autodiff, GradGraphInfo, get_expected_forward_inputs_from_call)

import popart_ir_extensions as pir_ext
from popart_ir_extensions.tuple_map import sanitise

__all__ = ["autodiff", "autodiff_with_accumulation", "connect_activations"]


class ConcreteGradGraph(pir_ext.ConcreteGraph):
    def __init__(self):
        super().__init__()
        self.grad_info: GradGraphInfo

    def to_callable_with_mapping(self, *args, **kwargs):
        graph = super().to_callable_with_mapping(*args, **kwargs)
        grad_graph = CallableGradGraph(self.grad_info, graph._graph, graph._input_defs)
        grad_graph.insert_all(graph)
        return grad_graph

    @classmethod
    def _from_pb(cls, graph: _ir.Graph, grad_info: Optional[GradGraphInfo] = None, **kwargs) -> 'ConcreteGradGraph':
        self = super()._from_pb(graph, **kwargs)
        if grad_info is not None:
            self.grad_info = grad_info
        return self


class CallableGradGraph(pir_ext.CallableGraph):
    def __init__(self, grad_info: GradGraphInfo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_info = grad_info


def autodiff(graph: pir_ext.ConcreteGraph,
             grads_provided: Optional[Iterable[pir.Tensor]] = None,
             grads_required: Optional[Iterable[pir.Tensor]] = None,
             called_graphs_grad_info: Optional[Mapping[pir.Graph, GradGraphInfo]] = None,
             return_all_grad_graphs: bool = False) -> Union[ConcreteGradGraph, Dict[pir.Graph, ConcreteGradGraph]]:
    """
    Extension Autodiff.
    This method calls pir.transforms.autodiff and then some required patterns after to ensure the returned
    grad graph is lowerable.

    Args:
        graph (pir.Graph)
        grads_provided (Optional[Iterable[pir.Tensor]], optional). Defaults to all outputs of the provided graph.
        grads_required (Optional[Iterable[pir.Tensor]], optional). Defaults to all inputs of the provided graph.
        called_graphs_grad_info (Optional[Mapping[pir.Graph, GradGraphInfo]], optional). Defaults to None.
        return_all_grad_graphs (bool, optional). Defaults to False.

    Returns:
        ConcreteGradGrad
    """
    grad_info_all: Dict[pir.Graph, GradGraphInfo] = _autodiff(graph,
                                                              grads_provided=grads_provided,
                                                              grads_required=grads_required,
                                                              called_graphs_grad_info=called_graphs_grad_info,
                                                              return_all_grad_graphs=True)  # type: ignore
    grad_info = grad_info_all[graph]

    ir = grad_info.graph.ir()._pb_ir
    # TODO: Only run required patterns
    ir.setPatterns(_ir.patterns.Patterns(_ir.patterns.PatternsLevel.Default))

    for grad_info_i in grad_info_all.values():
        ir.applyPreAliasPatterns(grad_info_i.graph._pb_graph)
        # TODO: Should inplacing be run?
        #       If not, we can end up with lots of outplace identities
        ir.applyInplacePattern(grad_info_i.graph._pb_graph)

    if not return_all_grad_graphs:
        grad_graph = ConcreteGradGraph._from_pb(grad_info.graph._pb_graph, grad_info)
        return grad_graph
    else:
        grad_graph_all = {
            graph: ConcreteGradGraph._from_pb(grad_info_i.graph._pb_graph, grad_info_i)
            for graph, grad_info_i in grad_info_all.items()
        }
        return grad_graph_all


def connect_activations(forward_call_info: CallInfo, callable_grad_graph: CallableGradGraph):
    """Connect the activations from a callsite of a forward graph to a CallableGraph of the associated gradient graph.

    Args:
        forward_call_info (CallInfo): From `call_with_info` of calling a forward graph.
        callable_grad_graph: result of converting a ConcreteGradGraph into CallableGradGraph
    """
    activations = get_expected_forward_inputs_from_call(forward_call_info, callable_grad_graph.grad_info)
    for sg_tensor, act in activations.items():
        callable_grad_graph[sanitise(act.name)] = (sg_tensor, act)


def autodiff_with_accumulation(concrete_graph: pir_ext.ConcreteGraph,
                               tensors_to_accumulate_grads: Iterable[pir.Tensor]) -> ConcreteGradGraph:
    """Calls `pir_ext.autodiff` then `pir_ext.accumulate_gradients_in_graph`."""
    # Autodiff the graph.
    grad_graph = autodiff(concrete_graph)

    # Modify the graph to have accumulator inputs
    accumulate_gradients_in_graph(grad_graph, tensors_to_accumulate_grads)

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
            accum = graph.add_input_tensor(lambda: np.zeros(tensor.shape, accum_type.as_numpy()),
                                           "Accum__" + tensor.name)
            ops.accumulate(accum, subgraph_tensor)

        variables[tensor] = accum

    return variables
