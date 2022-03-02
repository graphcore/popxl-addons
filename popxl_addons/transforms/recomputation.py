# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Tuple
import popart._internal.ir as _ir
import popxl
from popxl.transforms.autodiff import (ExpectedConnection, GradGraphInfo, ExpectedConnectionType)
from popxl_addons.graph import BoundGraph, GraphWithNamedArgs


def add_recompute_inputs(grad_info: GradGraphInfo):
    """Adds inputs required for recompute to the current graph. Important that inputs
        are added in the same order as they are placed in expected_inputs"""
    fwd_input_mapping: Dict[popxl.Tensor, popxl.Tensor] = {}
    grad_input_mapping: Dict[popxl.Tensor, popxl.Tensor] = {}

    expected_inputs = []

    fwd_inputs = grad_info.forward_graph.inputs
    grad_inputs = grad_info.graph.inputs

    # Add inputs that are required for grad_graph
    for idx, ec in enumerate(grad_info.expected_inputs):
        if ec.connection_type == ExpectedConnectionType.FwdGrad:
            tensor = grad_inputs[idx]
            grad_input_mapping[tensor] = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name)
            expected_inputs.append(ec)
        elif ec.connection_type == ExpectedConnectionType.Fwd and ec.fwd_tensor in fwd_inputs:
            tensor = ec.fwd_tensor
            fwd_input_mapping[tensor] = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name)
            expected_inputs.append(ec)

    # Finally add any additional inputs to graph that haven't been created
    for tensor in set(fwd_inputs) - set(fwd_input_mapping.keys()):
        fwd_input_mapping[tensor] = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name)
        expected_inputs.append(
            ExpectedConnection._from_pb(grad_info.forward_graph._pb_graph,
                                        _ir.ExpectedConnection(tensor.id, _ir.ExpectedConnectionType.Fwd)))

    return fwd_input_mapping, grad_input_mapping, expected_inputs


def recompute_graph(grad_graph: GraphWithNamedArgs) -> GraphWithNamedArgs:
    """
    Add recomputation to gradient graph

    Args:
        grad_graph (GraphWithNamedArgs) A gradient graph

    Returns:
        GraphWithNamedArgs: gradient graph with recomputation
    """
    ir = grad_graph.graph.ir
    r_graph = ir.create_empty_graph(grad_graph.graph.name + "_recomp")

    grad_info = grad_graph.grad_graph_info

    with r_graph:
        # Add Required Inputs to the Recompute graph from the GradGraphInfo
        fwd_recomp_inputs, grad_inputs, expected_inputs = add_recompute_inputs(grad_info)

        fgraph = BoundGraph(grad_info.forward_graph, fwd_recomp_inputs)
        # Call Forward Graph
        call_info = fgraph.call_with_info()

        # Include activations
        activations = grad_info.inputs_dict(call_info)
        grad_inputs.update(activations)

        # These are inputs to the grad graph that aren't from `GradGraphInfo`. Such as gradient accumulators.
        for tensor in set(grad_graph.graph.inputs) - set(grad_inputs.keys()):
            grad_inputs[tensor] = popxl.graph_input(tensor.shape,
                                                    tensor.dtype,
                                                    tensor.name,
                                                    by_ref=tensor in grad_graph.graph._by_ref_inputs)

        ggraph = BoundGraph(grad_graph.graph, grad_inputs)
        # Call Gradient Graph
        call_info = ggraph.call_with_info()

        # Add outputs in the Recompute Graph for each output in the Gradient Graph
        for output in call_info.outputs:
            popxl.graph_output(output)

        # Remap NamedArgs from the Gradient Graph to the Recompute Graph
        remapped_args = grad_graph.args.remap(grad_inputs)

        # Construct the new GradGraphInfo
        _r_grad_info = _ir.BwdGraphInfo(r_graph._pb_graph.id, [ec._pb_ec for ec in expected_inputs],
                                        [ec._pb_ec for ec in grad_info.expected_outputs])
        r_grad_info = GradGraphInfo._from_pb(ir._pb_ir, grad_info.forward_graph._pb_graph, _r_grad_info)

    return GraphWithNamedArgs(r_graph, remapped_args, r_grad_info)
