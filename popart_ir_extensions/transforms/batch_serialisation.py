# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Tuple
from numpy.lib.function_base import gradient
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.transforms.autodiff import GradGraphInfo
from popart_ir_extensions.graph import GraphWithNamedArgs
from popart_ir_extensions.route_tensor import route_tensor_into_graph
from popart_ir_extensions.transforms.autodiff import remap_grad_info

__all__ = ["batch_serialise"]


def batch_serialise(graph: GraphWithNamedArgs, steps: int, batched_inputs: Iterable[pir.Tensor]) -> GraphWithNamedArgs:
    """Returns a new graph that wraps `graph`:

        BatchSerialGraph:
            Repeat(MicroBatchGraph, `steps`)
        
        MicroBatchGraph:
            ins = DynamicSlice(batched_inputs)
            outs = Call(ComputeBatchGraph, ins)
            DynamicUpdate(batched_outs, outs)

        ComputeBatchGraph = `graph`

    Args:
        graph (GraphWithNamedArgs): Graph to be batch serialised.
        steps (int): Number of batch serialised steps that should be executed.
        batched_inputs (Iterable[pir.Tensor]): Input tensors to `graph` that should be batched.

    Returns:
        GraphWithNamedArgs: Batch serialised graph
    """

    micro_batch_graph = graph.graph.ir.create_empty_graph(graph.graph.name + "_micro_batch")
    batch_serial_graph = graph.graph.ir.create_empty_graph(graph.graph.name + "_batch_serial")
    with batch_serial_graph:
        index_t = ops.init((), pir.uint32, "batch_serial_index")
        ops.repeat(micro_batch_graph, steps)

    compute_to_micro_batch_map = {}
    micro_batch_to_batch_serial_map = {}

    def loop_input(compute_t, index):
        batched_input = compute_t in batched_inputs

        shape = (steps, *compute_t.shape) if batched_input else compute_t.shape
        with batch_serial_graph:
            batch_serial_input = pir.graph_input(shape, compute_t.dtype, compute_t.name,
                                                 compute_t in graph.graph._by_ref_inputs)

        t = route_tensor_into_graph(batch_serial_input,
                                    modified=compute_t._pb_tensor.modifiedRegionsByOps(graph.graph._pb_graph.getOps()))

        micro_batch_to_batch_serial_map[t] = batch_serial_input

        if batched_input:
            t = ops.dynamic_slice(t, index, [0], [1], True).reshape(compute_t.shape)

        compute_to_micro_batch_map[compute_t] = t
        return t

    def loop_output(compute_out, index):
        with batch_serial_graph:
            shape = (steps, *compute_out.shape)
            batch_serial_output = ops.init(shape, compute_out.dtype, compute_out.name)
            pir.graph_output(batch_serial_output)

        out = route_tensor_into_graph(batch_serial_output, modified=True)
        ops.dynamic_update_(out, index, compute_out.reshape((1, *compute_out.shape)), [0], [1], True)

    with micro_batch_graph, pir.in_sequence():
        index_t = route_tensor_into_graph(index_t, modified=True)

        for compute_t in graph.graph.inputs:
            loop_input(compute_t, index_t)

        compute_batch_outputs = graph.call(args=compute_to_micro_batch_map)

        for out in compute_batch_outputs:
            loop_output(out, index_t)

        ops.var_updates.accumulate_(index_t, pir.constant(1, index_t.dtype))

    # Remap NamedTensors
    named_args = graph.args.remap(compute_to_micro_batch_map).remap(micro_batch_to_batch_serial_map)

    return GraphWithNamedArgs(batch_serial_graph, named_args)


def batch_serialise_forward_and_grad(
        forward_graph: GraphWithNamedArgs, gradient_graph: GraphWithNamedArgs, steps: int,
        batched_inputs: Iterable[pir.Tensor]) -> Tuple[GraphWithNamedArgs, GraphWithNamedArgs]:
    """
    Batch serialise a forward and gradient graph.

    Args:
        forward_graph (GraphWithNamedArgs): Forward graph to be batch serialised.
        gradient_graph (GraphWithNamedArgs): Gradient graph to be batch serialised.
        steps (int): Number of batch serialised steps that should be executed.
        batched_inputs (Iterable[pir.Tensor]): Input tensors to `graph` that should be batched.

    Returns:
        GraphWithNamedArgs: batch serialised forward graph
        GraphWithNamedArgs: batch serialised gradient graph
    """

    grad_batched_inputs = []
    for idx, t in enumerate(gradient_graph.grad_graph_info.inputs):
        if t in batched_inputs or t in forward_graph.graph.outputs:
            grad_batched_inputs.append(gradient_graph.graph.inputs[idx])

    forward_graph = batch_serialise(forward_graph, steps, batched_inputs)

    grad_graph_info_prev = gradient_graph.grad_graph_info
    gradient_graph = batch_serialise(gradient_graph, steps, grad_batched_inputs)

    gradient_graph.grad_graph_info = remap_grad_info(grad_graph_info_prev, forward_graph.graph, gradient_graph.graph)

    return forward_graph, gradient_graph
