# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import popxl
from popxl import ops
from popxl.ops.call import CallSiteInfo
from popxl.transforms.autodiff import ExpectedConnectionType
from popxl_addons import GraphWithNamedArgs, route_tensor_into_graph, NamedTensors
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.route_tensor import route_tensor_into_graph
from popxl_addons.transforms.autodiff import remap_grad_info
from popxl_addons.utils import suffix_graph_name

__all__ = ["batch_serialise", "batch_serialise_fwd_and_grad", "batch_serial_buffer"]

RemoteBufferAndOffset = Tuple[popxl.RemoteBuffer, Optional[int]]


def _is_remote_buffer_and_offset(value: Any) -> bool:
    return len(value) == 2 and isinstance(value[0], popxl.RemoteBuffer) and (isinstance(value[1], int)
                                                                             or value[1] is None)


@dataclass
class BatchSerialResult:
    """Result of executing the a BatchSerialisation transform.
    `graph` is the transformed graph.
    `stored_buffers` is a map from tensors in the original graph to Buffers and offsets of values stored as a result of the transform
    """
    graph: GraphWithNamedArgs
    stored_buffers: Dict[popxl.Tensor, RemoteBufferAndOffset]

    _compute_to_micro: Dict[popxl.Tensor, popxl.Tensor]
    _micro_to_repeat: Dict[popxl.Tensor, popxl.Tensor]

    def remap_tensors(self, ts: NamedTensors) -> NamedTensors:
        """Remap tensors in `ts` from the original graph to the transformed graph.

        Args:
            ts (NamedTensors): Tensors to remap

        Returns:
            NamedTensors: Remapped tensors.
        """
        return ts.remap(self._compute_to_micro).remap(self._micro_to_repeat)


def batch_serial_buffer(t: popxl.Tensor, entries: int = 1) -> RemoteBufferAndOffset:
    """Create a RemoteBuffer and offset tuple from that matches a Tensor `t`

    Args:
        t (popxl.Tensor): Tensor to make a RemoteBuffer for.
        entries (int, optional): the size of the buffer. Defaults to 1

    Returns:
        RemoteBufferAndOffset: (buffer, offset)
    """
    offset = 0
    return (popxl.remote_buffer(t.shape, t.dtype, entries), offset)


def _graphs(ir: popxl.Ir, steps: int, name: str,
            require_seed: bool) -> Tuple[popxl.Graph, popxl.Graph, popxl.Tensor, Optional[popxl.Tensor]]:
    """Create the graphs required for the batch_serial Transformation.

    Args:
        ir (popxl.Ir): Ir to add graphs to.
        steps (int): Number of batch serial steps
        name (str): Graph name prefix
        require_seed (bool): Requires a random seed input
    """
    micro_graph = ir.create_empty_graph(suffix_graph_name(name, "_micro_batch"))
    repeat_graph = ir.create_empty_graph(suffix_graph_name(name, "_repeat_graph"))

    seed = None

    with micro_graph:
        # Loop Carried inputs
        index = popxl.graph_input([], popxl.int32, "batch_serial_loop_index")
        popxl.graph_output(index + 1)
        if require_seed:
            seed = popxl.graph_input((2, ), popxl.uint32, "batch_serial_random_seed")
            seed, next_seed = ops.split_random_seed(seed)
            popxl.graph_output(next_seed)

    with repeat_graph:
        index_t = popxl.graph_input([], popxl.int32, "batch_serial_loop_index")
        index_t = index_t * steps
        inputs = [index_t]
        if seed is not None:
            seed_t = popxl.graph_input((2, ), popxl.uint32, "batch_serial_random_seed")
            inputs.append(seed_t)
        ops.repeat(micro_graph, steps, *inputs)

    return micro_graph, repeat_graph, index, seed


@popxl.transforms.merge_exchange()
def load_inputs(compute_graph: GraphWithNamedArgs,
                load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferAndOffset]],
                repeat_graph: popxl.Graph, index_t: popxl.Tensor, steps: int, seed_t: Optional[popxl.Tensor],
                entries: int):
    """Add inputs to the repeated Graph.
        Each input can either be loaded from a HostToDeviceStream, or from a RemoteBuffer.
        Any input without an entry in load_handles will be added as a new input to the final Graph.

    Args:
        compute_graph (GraphWithNamedArgs): Users graph that contains the repeated computation.
        load_handles (Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferAndOffset]]): Description of how to load each input.
        repeat_graph (GraphWithNamedArgs): The final Graph, contains a single LoopOp.
        index_t (popxl.Tensor): Index tensor with the current batch serial step.
        steps (int): Number of batch serialisation steps
        entries (int): Number of serial for each of the remote buffers.

    Returns:
        Tuple[Dict[popxl.Tensor, popxl.Tensor],
              Dict[popxl.Tensor, popxl.Tensor]]: Dict from tensors in the users graph to the micro batch graph,
                                                 Dict from tensors in the micro batch graph to the final graph.
    """
    compute_to_micro = {}
    micro_to_repeat = {}
    for t in compute_graph.graph.inputs:
        # Seed input tensor is handled seperately
        if t == seed_t:
            continue

        handle = load_handles.get(t, None)
        if handle is None:
            # No handle provided for `t`.
            # Create a input in the final graph and route it through the LoopOp
            with repeat_graph:
                by_ref = t in compute_graph.graph._by_ref_inputs
                repeat_t = popxl.graph_input(t.shape, t.dtype, t.name, by_ref, t.meta_shape)
            modified_regions = t._pb_tensor.modifiedRegionsByOps(compute_graph.graph._pb_graph.getOps())
            micro_t = route_tensor_into_graph(repeat_t, modified=modified_regions)
            compute_to_micro[t] = micro_t
            micro_to_repeat[micro_t] = repeat_t
        elif isinstance(handle, popxl.HostToDeviceStream):
            # HostToDevice handle. HostLoad.
            compute_to_micro[t] = ops.host_load(handle, t.name)
        elif _is_remote_buffer_and_offset(handle):
            # (RemoteBuffer, int) handle.
            # RemoteLoad.
            # Int represents offset into RemoteBuffer.entries from which to start loading from.
            # offset of None will always read from the first entry
            buffer, offset = handle
            index_with_offset = index_t
            if offset is None:
                index_with_offset = index_with_offset % steps
                buffer_size = steps
            elif offset != 0:
                index_with_offset = index_with_offset + (offset * steps)
                buffer_size = steps * (offset + entries)
            else:
                buffer_size = steps * entries
            buffer.entries = max(buffer.entries, buffer_size)
            compute_to_micro[t] = ops.remote_load(buffer, index_with_offset, t.name)
        else:
            raise TypeError(f"Incompatible handle in `load_handles` for tensor {t} : {handle}")
    return compute_to_micro, micro_to_repeat


@popxl.transforms.merge_exchange()
def store_outputs(compute_info: CallSiteInfo,
                  load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferAndOffset]],
                  store_streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
                  store_buffers: Dict[popxl.Tensor, RemoteBufferAndOffset], index_t: popxl.Tensor, steps: int,
                  entries: int) -> Dict[popxl.Tensor, RemoteBufferAndOffset]:
    """Store outputs from the repeated Graph.
        All Tensors in `store_streams` available will be host stored.
        All Tensors in `store_buffers` will be remote stored.

    Args:
        compute_info (CallSiteInfo): CallSiteInfo from calling the repeated graph.
        load_handles (Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferAndOffset]]): Tensors to `ops.host_store`
        store_streams (Dict[popxl.Tensor, popxl.DeviceToHostStream]): Tensors to `ops.host_store`
        store_buffers (Dict[popxl.Tensor, RemoteBufferAndOffset]): Tensors to `ops.remote_store`
        index_t (popxl.Tensor): Index tensor with the current batch serial step.
        steps (int): Number of batch serialisation steps
        entries (int): Number of serial for each of the remote buffers.

    Returns:
        Dict[popxl.Tensor, RemoteBufferAndOffset]: Tensors and their remote locations stored by this function.
    """
    # Filter out store_streams not in the called_graph
    store_streams_in_graph = filter(lambda t: t[0] in compute_info.called_graph, store_streams.items())
    for t, stream in store_streams_in_graph:
        micro_t = compute_info.graph_to_parent(t)
        ops.host_store(stream, micro_t)

    stored = {}
    store_buffers_in_graph = filter(lambda t: t[0] in compute_info.called_graph, store_buffers.items())
    for t, (buffer, offset) in store_buffers_in_graph:
        if t in load_handles.keys() and not isinstance(load_handles[t], popxl.HostToDeviceStream):
            # If the tensor is already in a RemoteBuffer don't store it again.
            stored[t] = load_handles[t]
        else:
            index_with_offset = index_t
            # Store the Tensor
            if offset is None:
                index_with_offset = index_with_offset % steps
                buffer_size = steps
            elif offset != 0:
                index_with_offset = index_with_offset + (offset * steps)
                buffer_size = steps * (offset + entries)
            else:
                buffer_size = steps * entries
            buffer.entries = max(buffer.entries, buffer_size)
            micro_t = compute_info.graph_to_parent(t)
            ops.remote_store(buffer, index_with_offset, micro_t)
            stored[t] = (buffer, offset)

    return stored


def batch_serialise(graph: GraphWithNamedArgs,
                    steps: int,
                    load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferAndOffset]],
                    store_streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
                    store_buffers: Dict[popxl.Tensor, RemoteBufferAndOffset],
                    seed_input: Optional[popxl.Tensor] = None,
                    entries: int = 1) -> BatchSerialResult:
    """Transform a Graph to repeat the computation `steps` times.

        Each iteration:
         * loads inputs as specified by `load_handles`.
         * calls `graph`
         * store outputs (and inputs) as specified by `store_streams` and `store_buffers`

        When specifying a RemoteBufferAndOffset a value as an offset. This offset will adjust the remote load/store index
        by `+(offset*steps)`. If the value of offset is None, the remote load/store index will be adjusted by `index % steps`.

        Inputs that are not specified in `load_handles` will be added as inputs to the returned Graph.
        Outputs that are not specified in `store_streams` or `store_buffers` are not output from the returned Graph.

        See `docs/phased_execution.md` for more details.

    Args:
        graph (GraphWithNamedArgs): Graph to transform
        steps (int): Number of batch serialise steps
        load_handles (Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferAndOffset]]): Handles to load inputs before computation.
        store_streams (Dict[popxl.Tensor, popxl.DeviceToHostStream]): Streams to store outputs after computation.
        store_buffers (Dict[popxl.Tensor, RemoteBufferAndOffset]): Buffers to store outputs (or inputs) after computation.
        seed_input (Optional[popxl.Tensor], optional): Input tensor of a random seed. Defaults to None.
        entries (int, optional): Increases the size of the remote buffers to allow for the returned Graph to be used with multiple input values. Defaults to 1.

    Returns:
        BatchSerialResult
    """
    if entries < 1:
        raise ValueError("entries must be >0")

    micro_graph, repeat_graph, index_t, seed_t = _graphs(graph.graph.ir, steps, graph.graph.name,
                                                         seed_input is not None)

    compute_to_micro = {}
    micro_to_repeat = {}

    if seed_input is not None:
        compute_to_micro[seed_input] = seed_t

    with micro_graph, popxl.in_sequence(False):
        c2m, m2r = load_inputs(graph, load_handles, repeat_graph, index_t, steps, seed_input, entries)
        compute_to_micro.update(c2m)
        micro_to_repeat.update(m2r)

        info = graph.call_with_info(args=compute_to_micro)

        stored_buffers = store_outputs(info, load_handles, store_streams, store_buffers, index_t, steps, entries)

    # Remap NamedTensors
    named_args = graph.args \
        .remap(compute_to_micro) \
        .remap(micro_to_repeat)

    named_graph = GraphWithNamedArgs(repeat_graph, named_args)

    return BatchSerialResult(named_graph, stored_buffers, compute_to_micro, micro_to_repeat)


def batch_serialise_fwd_and_grad(
        forward_graph: GraphWithNamedArgs,
        gradient_graph: GraphWithNamedArgs,
        steps: int,
        load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferAndOffset]],
        store_streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
        store_buffers: Dict[popxl.Tensor, RemoteBufferAndOffset],
        seed_input: Optional[popxl.Tensor] = None,
        entries: int = 1) -> Tuple[BatchSerialResult, BatchSerialResult, NamedTensors]:
    """Transform a matching forward and gradient Graphs that the computation is `steps` times.

        Tensors required for autodiff will be stored automatically.

        See `popxl_addons.transforms.batch_serialise` and `docs/phased_execution.md` for more details.

    Args:
        forward_graph (GraphWithNamedArgs): Forward Graph to transform
        gradient_graph (GraphWithNamedArgs): Gradient Graph to transform
        steps (int): Number of batch serialise steps
        load_handles (Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferAndOffset]]): Handles to load inputs before computation.
        store_streams (Dict[popxl.Tensor, popxl.DeviceToHostStream]): Streams to store outputs after computation.
        store_buffers (Dict[popxl.Tensor, RemoteBufferAndOffset]): Buffers to store outputs (or inputs) after computation.
        seed_input (Optional[popxl.Tensor], optional): Input tensor of a random seed. Defaults to None. Defaults to None.
        entries (int, optional): Increases the size of the remote buffers to allow for the returned Graph to be used with multiple input values. Defaults to 1.

    Returns:
        Tuple[BatchSerialResult, BatchSerialResult, NamedTensors]:
            result of forward_graph, result of gradient_graph, NamedArgs that must be provided at the gradient call.
    """

    grad_inputs = gradient_graph.graph.inputs
    grad_graph_info = gradient_graph.grad_graph_info

    named_expected_grad_inputs = {}
    named_expected_fwd_inputs = set()
    fwd_input_names, named_fwd_inputs = forward_graph.args.unpack()

    # Handle storing of activations required for autodiff.
    activations = [
        i_ec for i_ec in enumerate(grad_graph_info.expected_inputs)
        if i_ec[1].connection_type == ExpectedConnectionType.Fwd
    ]
    for grad_idx, ec in activations:
        t = ec.fwd_tensor
        if t in store_buffers.keys():
            # store buffer already specified.
            pass
        elif t in named_fwd_inputs:
            # named inputs will not be load/stored by the loop.
            # TODO: Make this an input argument instead.
            fwd_idx = named_fwd_inputs.index(t)
            named_expected_fwd_inputs.add(t)
            named_expected_grad_inputs[fwd_input_names[fwd_idx]] = grad_inputs[grad_idx]
        elif t in load_handles.keys() and not isinstance(load_handles[t], popxl.HostToDeviceStream):
            # activation already stored in the input handles
            store_buffers[t] = load_handles[t]
        else:
            # create a buffer for the activation
            store_buffers[t] = batch_serial_buffer(t)

    forward_result = batch_serialise(forward_graph, steps, load_handles, store_streams, store_buffers, seed_input,
                                     entries)

    # Handle loading of activations required for autodiff
    grad_load_handles = {}
    for idx, ec in activations:
        t = ec.fwd_tensor
        if t in forward_result.stored_buffers.keys():
            grad_load_handles[grad_inputs[idx]] = forward_result.stored_buffers[t]
        elif t not in named_expected_fwd_inputs:
            raise RuntimeError(f"Fwd Connection {t} should have been stored in the forward graph")

    # Handle loading of gradients required for autodiff
    for idx, ec in filter(lambda i_ec: i_ec[1].connection_type == ExpectedConnectionType.FwdGrad,
                          enumerate(grad_graph_info.expected_inputs)):
        grad_t = grad_inputs[idx]
        if grad_t not in load_handles:
            raise ValueError(f"FwdGrad input (idx={idx} {grad_t}) to Gradient graph must have a load_handle specified")
        grad_load_handles[grad_t] = load_handles[grad_t]

    gradient_result = batch_serialise(gradient_graph, steps, grad_load_handles, store_streams, store_buffers, None,
                                      entries)

    gradient_result.graph.grad_graph_info = remap_grad_info(grad_graph_info, forward_graph.graph, gradient_graph.graph)

    named_expected_grad_inputs = gradient_result.remap_tensors(NamedTensors.from_dict(named_expected_grad_inputs))

    return forward_result, gradient_result, named_expected_grad_inputs
