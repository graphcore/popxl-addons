# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import wraps, partial
from typing import Dict, Iterable, List, Optional, Mapping, Union, overload
from typing_extensions import Literal

import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.ops.call import CallInfo
from popart.ir.transforms.autodiff import (autodiff as _autodiff, GradGraphInfo)

import popart_ir_extensions as pir_ext
from popart_ir_extensions.tuple_map import sanitise

__all__ = ["autodiff", "autodiff_with_accumulation", "connect_activations"]


class ConcreteGradGraph(pir_ext.ConcreteGraph):
    def __init__(self):
        super().__init__()
        self.grad_info: GradGraphInfo
        self.grad_accumulators: Dict[pir.Tensor, pir.Tensor]

    @wraps(pir_ext.ConcreteGraph.to_callable)
    def to_callable(self, *args, **kwargs) -> 'CallableGradGraph':
        # super().to_callable calls `self.to_callable_with_mapping`
        return super().to_callable(*args, **kwargs)  # type: ignore

    @wraps(pir_ext.ConcreteGraph.to_callable_with_mapping)
    def to_callable_with_mapping(self, *args, **kwargs) -> 'CallableGradGraph':
        graph = super().to_callable_with_mapping(*args, **kwargs)
        grad_graph = CallableGradGraph(self.grad_info, graph._graph, graph._input_defs)
        grad_graph.insert_all(graph)
        return grad_graph

    @classmethod
    def _create_from_pir(cls,
                         graph: pir.Graph,
                         grad_info: Optional[GradGraphInfo] = None,
                         **kwargs) -> 'ConcreteGradGraph':
        self = super()._create_from_pir(graph, **kwargs)
        if grad_info is not None:
            self.grad_info = grad_info
        self.grad_accumulators = {}
        return self

    def get_grad_tensor_for_fwd_input(self, t: pir.Tensor):
        """Returns the gradient tensor in the graph for a tensor in the forward graph.

        Args:
            t (pir.Tensor): Tensor in the forward graph.

        Returns:
            pir.Tensor: Output Tensor in the gradient graph.
        """
        if t not in self.grad_info.forward_graph:
            raise ValueError(f"{t} not in forward graph {self.grad_info.forward_graph.id}.")
        for idx, fwd_tensor in enumerate(self.grad_info.get_output_tensors()):
            if t == fwd_tensor:
                return self.get_output_tensors()[idx]
        raise ValueError(f"{t} does not have an associated gradient in Grad Graph {self.id}. "
                         "Was it included in autodiff's `grads_required`?")

    def get_grad_accumulator_for_fwd_input(self, t: pir.Tensor):
        """Returns the gradient accumulator input tensor in the graph for a tensor in the forward graph.

        Args:
            t (pir.Tensor): Tensor in the forward graph.

        Returns:
            pir.Tensor: Input Tensor in the gradient graph.
        """
        if t not in self.grad_info.forward_graph:
            raise ValueError(f"{t} not in forward graph {self.grad_info.forward_graph.id}.")
        if t not in self.grad_accumulators:
            raise ValueError(f"t={t} does not have an associated gradient accumulator. "
                             "Has gradient accumulation been used? "
                             "Was 't' included in `tensors_to_accumulate_grads`?")
        return self.grad_accumulators[t]


class CallableGradGraph(pir_ext.CallableGraph):
    def __init__(self, grad_info: GradGraphInfo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_info = grad_info

    def get_grad_accumulator_for_fwd_input(self, t: pir.Tensor):
        """Returns the gradient accumulator connected to the CallableGraph for a tensor in the forward graph.

        Args:
            t (pir.Tensor): Tensor in the forward graph.

        Returns:
            pir.Tensor: Connected Input Tensor to the CallableGraph.
        """
        subgraph_g = self.grad_info.graph.get_grad_accumulator_for_fwd_input(t)  # type: ignore
        return self.call_input()[subgraph_g]


@overload
def autodiff(graph: pir_ext.ConcreteGraph,
             grads_provided: Optional[Iterable[pir.Tensor]] = None,
             grads_required: Optional[Iterable[pir.Tensor]] = None,
             called_graphs_grad_info: Optional[Mapping[pir.Graph, GradGraphInfo]] = None,
             return_all_grad_graphs: Literal[False] = False) -> ConcreteGradGraph:
    ...


@overload
def autodiff(graph: pir_ext.ConcreteGraph,
             grads_provided: Optional[Iterable[pir.Tensor]] = None,
             grads_required: Optional[Iterable[pir.Tensor]] = None,
             called_graphs_grad_info: Optional[Mapping[pir.Graph, GradGraphInfo]] = None,
             return_all_grad_graphs: Literal[True] = True) -> Dict[pir.Graph, ConcreteGradGraph]:
    ...


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
        grad_graph = ConcreteGradGraph._create_from_pir(grad_info.graph, grad_info)
        return grad_graph
    else:
        grad_graph_all = {
            graph: ConcreteGradGraph._create_from_pir(grad_info_i.graph, grad_info_i)
            for graph, grad_info_i in grad_info_all.items()
        }
        return grad_graph_all


def connect_activations(forward_call_info: CallInfo, callable_grad_graph: CallableGradGraph):
    """Connect the activations from a callsite of a forward graph to a CallableGraph of the associated gradient graph.

    Args:
        forward_call_info (CallInfo): From `call_with_info` of calling a forward graph.
        callable_grad_graph: result of converting a ConcreteGradGraph into CallableGradGraph
    """
    activations = callable_grad_graph.grad_info.get_inputs_from_forward_call_info(forward_call_info)
    for sg_tensor, act in activations.items():
        callable_grad_graph[sanitise(act.name)] = (sg_tensor, act)


def autodiff_with_accumulation(graph: pir_ext.ConcreteGraph,
                               tensors_to_accumulate_grads: Iterable[pir.Tensor],
                               grads_required: Optional[Iterable[pir.Tensor]] = None) -> ConcreteGradGraph:
    """Calls `pir_ext.autodiff` then `pir_ext.accumulate_gradients_in_graph`.

    Args:
        graph (pir_ext.ConcreteGraph): graph to autodiff
        tensors_to_accumulate_grads (Iterable[pir.Tensor]): Input tensors to `graph` for which accumulators should be added.
        grads_required (Optional[Iterable[pir.Tensor]], optional): Grads required for `autodiff`. Tensor in `tensors_to_accumulate_grads` will be added to this.
    """
    grads_required = list(grads_required or [])
    grads_required += tensors_to_accumulate_grads

    # Autodiff the graph.
    grad_graph = autodiff(graph, grads_required=grads_required)

    # Modify the graph to have accumulator inputs
    accumulate_gradients_in_graph(grad_graph, tensors_to_accumulate_grads)

    return grad_graph


def accumulate_gradients_in_graph(graph: ConcreteGradGraph,
                                  tensors_to_accumulate_grads: Iterable[pir.Tensor],
                                  accum_type: Optional[pir.dtype] = None):
    """Replace the outputs in grad graph that represent a gradient of a tensor in 'tensors_to_accumulate_grads'
        Adds a new input to the grad graph and an accumulate op in the grad graph.

       Returns a mapping from tensors in 'tensors_to_accumulate_grads' to the new subgraph_input"""
    grad_info = graph.grad_info

    expected_outputs = grad_info.get_output_tensors()

    indices_to_remove: List[int] = []

    with graph, pir.in_sequence(True):
        counter = graph.add_input_tensor(partial(np.zeros, shape=(), dtype=np.float32), "accum_counter", by_ref=True)
        with pir.in_sequence(False):
            for tensor in tensors_to_accumulate_grads:
                idx = expected_outputs.index(tensor)
                subgraph_tensor = pir.Tensor._from_pb_tensor(graph._pb_graph.getOutputTensor(idx))
                indices_to_remove.append(idx)

                accum_type = tensor.dtype if accum_type is None else accum_type

                accum = graph.add_input_tensor(partial(np.zeros, shape=tensor.shape, dtype=accum_type.as_numpy()),
                                               sanitise("accum_" + tensor.name),
                                               by_ref=True)
                ops.var_updates.accumulate_mean_(accum, subgraph_tensor, counter)
                graph.grad_accumulators[tensor] = accum

        ops.var_updates.accumulate_(counter, pir.constant(1, pir.float32))

    for idx in indices_to_remove:
        graph._pb_graph.removeOutput(idx)
