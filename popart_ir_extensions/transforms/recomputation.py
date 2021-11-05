# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Optional
import popart._internal.ir as _ir
import popart.ir as pir
from popart.ir.transforms.autodiff import (ExpectedConnection, GradGraphInfo, ExpectedConnectionType)
from popart_ir_extensions.graphs import ConcreteGraph
from popart_ir_extensions.transforms.autodiff import ConcreteGradGraph, connect_activations


def add_recompute_inputs(graph: ConcreteGraph, grad_graph: ConcreteGradGraph):
    """Adds inputs required for recompute to the current graph. Important that inputs
        are added in the same order as they are placed in expected_inputs"""
    fwd_inputs = graph.get_input_tensors()
    fwd_input_mapping: Dict[pir.Tensor, pir.Tensor] = {}
    grad_inputs = grad_graph.get_input_tensors()
    grad_input_mapping: Dict[pir.Tensor, pir.Tensor] = {}

    expected_inputs = []

    # Add inputs that are required for grad_graph
    for idx, ec in enumerate(grad_graph.grad_info.expected_inputs):
        if ec.connection_type == ExpectedConnectionType.FwdGrad:
            tensor = grad_inputs[idx]
            grad_input_mapping[tensor] = pir.subgraph_input(tensor.shape, tensor.dtype, tensor.name)
            expected_inputs.append(ec)
        elif ec.connection_type == ExpectedConnectionType.Fwd and ec.fwd_tensor in fwd_inputs:
            tensor = ec.fwd_tensor
            fwd_input_mapping[tensor] = pir.subgraph_input(tensor.shape, tensor.dtype, tensor.name)
            expected_inputs.append(ec)

    # Finally add any additional inputs to graph that haven't been created
    for tensor in set(fwd_inputs) - set(fwd_input_mapping.keys()):
        fwd_input_mapping[tensor] = pir.subgraph_input(tensor.shape, tensor.dtype, tensor.name)
        expected_inputs.append(
            ExpectedConnection._from_pb(graph._pb_graph,
                                        _ir.ExpectedConnection(tensor.id, _ir.ExpectedConnectionType.Fwd)))

    return fwd_input_mapping, grad_input_mapping, expected_inputs


def recompute_graph(graph: ConcreteGraph, grad_graph: ConcreteGradGraph) -> ConcreteGradGraph:
    """Recompute a Graph.
        Takes a forward and gradient graph.
        Returns a gradient graph that produces the same outputs as gradient graph
        but requires fewer inputs from the forward graph. It achieves this by creating a
        new graph that first calls the forward graph and then calls the gradient graph.

    Args:
        graph (ConcreteGraph): Forward Graph
        grad_graph (ConcreteGradGraph): Gradient Graph

    Returns:
        ConcreteGradGraph: Recompute Graph
    """
    graph = graph
    grad_graph = grad_graph

    r_graph = ConcreteGradGraph._from_pb(graph.ir().create_empty_graph(grad_graph.name + "_recomp")._pb_graph)

    with r_graph:
        fwd_recomp_inputs, grad_inputs, expected_inputs = \
            add_recompute_inputs(graph, grad_graph)

        # This need to use the inputs in not_connected_fwd
        fwd = graph.to_callable_with_mapping(fwd_recomp_inputs)

        fwd_call_info = fwd.call_with_info()

        # This needs to use the inputs in not_connected_bwd
        grad = grad_graph.to_callable_with_mapping(grad_inputs)
        # New recompute_graph should inherit the input_defs
        r_graph.input_defs.insert_all(grad._input_defs)

        connect_activations(fwd_call_info, grad)

        grad_call_info = grad.call_with_info()

        for output in grad_call_info.get_output_tensors():
            pir.subgraph_output(output)

        _r_grad_info = _ir.BwdGraphInfo(r_graph._pb_graph.id, [ec._pb_ec for ec in expected_inputs],
                                        [ec._pb_ec for ec in grad_graph.grad_info.expected_outputs])

        r_graph.grad_info = GradGraphInfo._from_pb(graph.ir()._pb_ir, graph._pb_graph, _r_grad_info)

    return r_graph
