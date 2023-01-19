# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
from contextlib import contextmanager
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import partial
import numpy as np
from typing_extensions import Literal
from dataclasses import dataclass
import popxl
from popxl import HostToDeviceStream, ReplicaGrouping, in_sequence, io_tiles, ops
from popxl.transforms.autodiff import ExpectedConnectionType
from popxl_addons import GraphWithNamedArgs, NamedTensors, VariableFactory, add_variable_input
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.transforms.autodiff import remap_grad_info
from popxl_addons.utils import suffix_graph_name
from popxl_addons.rts import gather_replica_sharded_tensor, replica_sharded_spec
from popxl_addons.remote import create_remote_buffer

__all__ = ["batch_serialise", "batch_serialise_fwd_and_grad", "batch_serial_buffer", "RemoteHandle"]


@dataclass
class RemoteHandle:
    """
    The RemoteHandle class gathers necessary info to deal with remote tensors.
    @param buffer (popxl.RemoteBuffer): the remote buffer for the tensor
    @param row_offset (int): the row offset index to access the tensor in the buffer. See batch_serialise for more on that. Default is 0.
    @param shard_group (Optional[popxl.ReplicaGrouping]): shard group for rts, used to create the buffer. None indicates no-RTS (see remote.create_remote_buffer)
    """

    buffer: popxl.RemoteBuffer
    row_offset: int = 0
    shard_group: Optional[popxl.ReplicaGrouping] = None


# TODO remove when all code changed to use RemoteHandles
RemoteBufferWithOffset = Tuple[popxl.RemoteBuffer, Optional[int]]


# TODO remove when all code changed to use RemoteHandles
def _convert_to_remote_handles(handles):
    for t, h in handles.items():
        if _is_remote_buffer_and_offset(h):
            handles[t] = RemoteHandle(*h)
    return handles


# TODO remove when all code changed to use RemoteHandles
def _is_remote_buffer_and_offset(value: Any) -> bool:
    if isinstance(value, HostToDeviceStream) or isinstance(value, RemoteHandle):
        return False
    else:
        return (
            len(value) >= 2
            and isinstance(value[0], popxl.RemoteBuffer)
            and (isinstance(value[1], int) or value[1] is None)
        )


def _is_remote_handle(value: Any) -> bool:
    return isinstance(value, RemoteHandle)


def _io_copy(ts: List[popxl.Tensor]) -> List[popxl.Tensor]:
    """Copy `ts` to/from IO Tiles"""
    return list(map(ops.io_tile_copy, ts))


def _get_loaded_ts(
    compute_graph: GraphWithNamedArgs, load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]]
) -> List[popxl.Tensor]:
    """Get Tensors that should be loaded to execute `compute_graph`"""
    return list(filter(lambda t: t in load_handles.keys(), compute_graph.graph.inputs))


def _get_stored_ts(
    compute_graph: GraphWithNamedArgs,
    load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]],
    store_streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
    store_buffers: Dict[popxl.Tensor, RemoteHandle],
) -> List[popxl.Tensor]:
    """Get Tensors that should be stored after executing `compute_graph`"""

    def filter_fn(t):
        has_input_buffer = t in load_handles.keys() and not isinstance(load_handles[t], popxl.HostToDeviceStream)
        return not has_input_buffer and (t in store_streams.keys() or t in store_buffers.keys())

    return list(filter(filter_fn, chain(compute_graph.graph.inputs, compute_graph.graph.outputs)))


def _get_passed_through_ts(
    compute_graph: GraphWithNamedArgs,
    load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]],
    seed_t: Optional[popxl.Tensor],
) -> List[popxl.Tensor]:
    """Get input tensors that should be passed_through to each iteration of `compute_graph`.
    Input tensors without an entry in `load_handles` will be marked as `passed_through`."""
    return list(filter(lambda t: t != seed_t and t not in load_handles.keys(), compute_graph.graph.inputs))


def _inputs_dict(
    loaded_ts: List[popxl.Tensor],
    loaded: List[popxl.Tensor],
    passed_through_ts: List[popxl.Tensor],
    passed_through: List[popxl.Tensor],
    seed_input: Optional[popxl.Tensor],
    seed: Optional[popxl.Tensor],
) -> Dict[popxl.Tensor, popxl.Tensor]:
    """Construct the input dictionary from loaded, passed through and seed tensors."""
    inputs = {}
    if seed_input is not None:
        inputs[seed_input] = seed
    inputs.update(dict(zip(loaded_ts, loaded)))
    inputs.update(dict(zip(passed_through_ts, passed_through)))
    return inputs


def _add_passed_through_inputs(
    compute_graph: GraphWithNamedArgs, passed_through_ts: List[popxl.Tensor]
) -> List[popxl.Tensor]:
    """Add graph inputs in the current graph for each of the passed through inputs"""
    passed_through = []
    for t in passed_through_ts:
        by_ref = t in compute_graph.graph._by_ref_inputs
        passed_through.append(popxl.graph_input(t.shape, t.dtype, t.name, by_ref, t.meta_shape))
    return passed_through


def _apply_load_offsets(
    index: popxl.Tensor,
    ts: List[popxl.Tensor],
    handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]],
    steps: int,
    rows: int,
) -> List[Tuple[popxl.Tensor, Union[popxl.HostToDeviceStream, Tuple[popxl.RemoteBuffer, popxl.Tensor]]]]:
    """Apply the offsets defined in `handles` and return the handles in the same order as `ts`."""
    applied = []
    for t in ts:
        handle = handles[t]
        if isinstance(handle, popxl.HostToDeviceStream):
            applied.append((t, handle))
        elif _is_remote_handle(handle):
            # (RemoteBuffer, int) handle.
            # RemoteLoad.
            # Int represents row_offset into RemoteBuffer.entries from which to start loading from.
            # row_offset of None will always read from the first entry
            buffer = handle.buffer
            row_offset = handle.row_offset
            index_with_offset = index
            if row_offset is None:
                index_with_offset = index_with_offset % steps
                buffer_size = steps
            elif row_offset != 0:
                index_with_offset = index_with_offset + (row_offset * steps)
                buffer_size = steps * (row_offset + rows)
            else:
                buffer_size = steps * rows
            buffer.entries = max(buffer.entries, buffer_size)
            applied.append((t, (buffer, index_with_offset)))
        else:
            raise TypeError(f"Incompatible handle in `load_handles` for tensor {t} : {handle}")
    return applied


def _apply_store_offsets(
    index: popxl.Tensor, ts: List[popxl.Tensor], buffers: Dict[popxl.Tensor, RemoteHandle], steps: int, rows: int
) -> Dict[popxl.Tensor, Tuple[popxl.RemoteBuffer, popxl.Tensor]]:
    """Apply the offsets defined in `buffers`"""
    buffers_with_offset = {}
    for t in ts:
        if t in buffers.keys():
            buffer = buffers[t].buffer
            row_offset = buffers[t].row_offset
            index_with_offset = index
            # Store the Tensor
            if row_offset is None:
                index_with_offset = index_with_offset % steps
                buffer_size = steps
            elif row_offset != 0:
                index_with_offset = index_with_offset + (row_offset * steps)
                buffer_size = steps * (row_offset + rows)
            else:
                buffer_size = steps * rows
            buffer.entries = max(buffer.entries, buffer_size)
            buffers_with_offset[t] = (buffer, index_with_offset)
    return buffers_with_offset


def _load_tensors(
    handles: List[Tuple[popxl.Tensor, Union[popxl.HostToDeviceStream, Tuple[popxl.RemoteBuffer, popxl.Tensor]]]]
) -> List[popxl.Tensor]:
    """Load tensors as defined in `handles`. Returns the loaded Tensors in the same order as `handles`."""
    loaded = []
    for t, handle in handles:
        if isinstance(handle, popxl.HostToDeviceStream):
            # HostToDevice handle. HostLoad.
            loaded.append(ops.host_load(handle, t.name))
        else:
            buffer, entry = handle
            loaded.append(ops.remote_load(buffer, entry, t.name))
    return loaded


def _store_tensors(
    ts: List[popxl.Tensor],
    to_store: List[popxl.Tensor],
    streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
    buffers_with_offset: Dict[popxl.Tensor, Tuple[popxl.RemoteBuffer, popxl.Tensor]],
):
    """Store tensors `to_store` as defined in `streams` and `buffers_with_offset`"""
    for t, store in zip(ts, to_store):
        if t in streams.keys():
            ops.host_store(streams[t], store)

        if t in buffers_with_offset.keys():
            buffer, entry = buffers_with_offset[t]
            ops.remote_store(buffer, entry, store)


def _gather_tensors(
    load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]],
    loaded_ts: Dict[popxl.Tensor, popxl.Tensor],
) -> List[popxl.Tensor]:
    """
    Gather replica sharded tensors
    """
    gathered = []
    for t, loaded_t in loaded_ts.items():
        if _is_remote_handle(load_handles[t]):
            rg = load_handles[t].shard_group
            if rg and rg.group_size != 1:
                # io tiles external
                loaded_t = gather_replica_sharded_tensor(loaded_t, use_io_tiles=False, replica_group=rg)
        gathered.append(loaded_t)
    return gathered


def _slice_tensors(
    store_buffers: Dict[popxl.Tensor, RemoteHandle], to_store: Dict[popxl.Tensor, popxl.Tensor]
) -> List[popxl.Tensor]:
    """
    Slice replica sharded tensors
    """
    sliced = []
    for t, parent_t in to_store.items():
        if t in store_buffers:
            rg = store_buffers[t].shard_group
            if rg and rg.group_size != 1:
                parent_t = ops.collectives.replica_sharded_slice(parent_t, group=rg)
        sliced.append(parent_t)
    return sliced


# Backported to python3.6
@contextmanager
def null_context():
    yield


@dataclass
class BatchSerialResult:
    """Result of executing the a BatchSerialisation transform.
    `graph` is the transformed graph.
    `stored_buffers` is a map from tensors in the original graph to RemoteHandle of values stored as a result of the transform
    """

    graph: GraphWithNamedArgs
    stored_buffers: Dict[popxl.Tensor, RemoteHandle]

    _remap_dict: Dict[popxl.Tensor, popxl.Tensor]

    def remap_tensors(self, ts: NamedTensors) -> NamedTensors:
        """Remap tensors in `ts` from the original graph to the transformed graph.

        Args:
            ts (NamedTensors): Tensors to remap

        Returns:
            NamedTensors: Remapped tensors.
        """
        return ts.remap(self._remap_dict)


def batch_serial_buffer(
    t: popxl.Tensor, steps: int = 1, rows: int = 1, shard_group: Optional[ReplicaGrouping] = None
) -> popxl.RemoteBuffer:
    """Create a RemoteBuffer for the Tensor `t`, with a matrix layout.
    The returned buffer can be seen as a matrix with `steps` columns and `rows` rows.
    Each column corresponds to the tensor at a different serial step.
    You want to use multiple rows when you create a buffer for identical layers, so that each layer has its row.
    The tensor at a given batch step is located in the remote buffer at `index = row_index * steps + batch_index`.

    Args:
        t (popxl.Tensor): Tensor to make a RemoteBuffer for.
        steps (int, optional) = 1: number of batch serialisation steps. Defaults to 1.
        rows (int, optional) = 1: number of rows for the buffer. Defaults to 1.
        shard_group (ReplicaGrouping, optional) = None: replica group used for sharding the tensor. If no group is provided, the tensor won't be sharded.
    Returns:
        popxl.remote_buffer : a remote buffer with entries = steps * rows
    """
    shard_over = shard_group.group_size if shard_group else 1
    buffer = create_remote_buffer(t, entries=steps * rows, replica_group=shard_group, shard_over=shard_over)
    return buffer


def batch_serialise_non_overlapped(
    graph: GraphWithNamedArgs,
    steps: int,
    load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]],
    store_streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
    store_buffers: Dict[popxl.Tensor, RemoteHandle],
    seed_input: Optional[popxl.Tensor] = None,
    rows: int = 1,
    use_io_tiles: bool = False,
) -> BatchSerialResult:
    """Batch Serialise `graph` without overlapped IO"""
    if rows < 1:
        raise ValueError("rows must be >0")
    ir = graph.graph.ir
    opts = ir._pb_ir.getSessionOptions()
    if use_io_tiles and opts.numIOTiles < 1:
        raise ValueError("IR must have IO tiles")

    tileset = popxl.io_tiles if use_io_tiles else null_context

    # Fetch tensors to be loaded/stored/passed_through.
    loaded_ts = _get_loaded_ts(graph, load_handles)
    stored_ts = _get_stored_ts(graph, load_handles, store_streams, store_buffers)
    passed_through_ts = _get_passed_through_ts(graph, load_handles, seed_input)

    @popxl.in_sequence(True)
    def micro_graph_fn(index: popxl.Tensor, seed: popxl.Tensor, passed_through: List[popxl.TensorByRef]):
        seed, next_seed = ops.split_random_seed(seed)
        with tileset():
            to_load = _apply_load_offsets(index, loaded_ts, load_handles, steps, rows)
            buffers_with_offset = _apply_store_offsets(index, stored_ts, store_buffers, steps, rows)
        # Load
        with tileset(), popxl.transforms.merge_exchange(), popxl.in_sequence(False):
            loaded = _load_tensors(to_load)  # loaded on io tiles if io

        # RTS Gather
        with tileset():
            loaded = _gather_tensors(load_handles, dict(zip(loaded_ts, loaded)))

        if use_io_tiles:
            loaded = _io_copy(loaded)  # io -> compute tiles

        # Compute
        info = graph.call_with_info(
            args=_inputs_dict(loaded_ts, loaded, passed_through_ts, passed_through, seed_input, seed)
        )
        stored = [info.graph_to_parent(t) for t in stored_ts]
        # Store
        if use_io_tiles:
            stored = _io_copy(stored)  # compute -> io

        # RTS Slice
        with tileset():
            stored = _slice_tensors(store_buffers, dict(zip(stored_ts, stored)))

        with tileset(), popxl.transforms.merge_exchange(), popxl.in_sequence(False):
            _store_tensors(stored_ts, stored, store_streams, buffers_with_offset)

        with tileset():
            index = index + 1
        return index, next_seed

    repeat_graph = ir.create_empty_graph(suffix_graph_name(graph.graph.name, "_repeat_graph"))

    with repeat_graph, popxl.in_sequence():
        # Add inputs to returned graph.
        #   Index, Seed, *Passed Through.
        index = popxl.graph_input([], popxl.int32, "batch_serial_loop_index")
        with tileset():
            if use_io_tiles:
                index = ops.io_tile_copy(index)
            index = index * steps
        if seed_input is not None:
            seed = popxl.graph_input((2,), popxl.uint32, "batch_serial_random_seed")
        else:
            seed = popxl.constant((0, 0), popxl.uint32, "dummy_random_seed")
        passed_through = _add_passed_through_inputs(graph, passed_through_ts)

        # Execute batch serial loop.
        micro_graph = ir.create_graph(micro_graph_fn, index, seed, passed_through)
        ops.repeat(micro_graph, steps, index, seed, *passed_through)

    remap_dict = dict(zip(passed_through_ts, passed_through))

    # Remap NamedTensors
    named_args = graph.args.remap(remap_dict)
    named_graph = GraphWithNamedArgs(repeat_graph, named_args)

    stored_buffers = {}
    for t in filter(lambda t: t in graph.graph, store_buffers.keys()):
        if t in load_handles.keys() and not isinstance(load_handles[t], popxl.HostToDeviceStream):
            stored_buffers[t] = load_handles[t]
        else:
            stored_buffers[t] = store_buffers[t]

    return BatchSerialResult(named_graph, stored_buffers, remap_dict)


def batch_serialise_overlapped(
    graph: GraphWithNamedArgs,
    steps: int,
    load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]],
    store_streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
    store_buffers: Dict[popxl.Tensor, RemoteHandle],
    seed_input: Optional[popxl.Tensor] = None,
    rows: int = 1,
):
    """Batch Serialise `graph` with overlapped IO.

    To be able to overlap the IO with the compute we must decompose the standard batch serialisation loop
    such that load/store/compute are operating on different batches of the loop.

    Graph:
        load0
        load1, compute0
        repeat I=(2 -> N)
          storeI-2, loadI, computeI-1
        storeN-1, computeN
        storeN
    """
    # Validate arguments
    if rows < 1:
        raise ValueError("rows must be >0")
    if steps < 2:
        raise ValueError("steps must be >=2 if using overlapped IO")
    ir = graph.graph.ir
    opts = ir._pb_ir.getSessionOptions()
    if opts.numIOTiles < 1:
        raise ValueError("IR must have IO tiles")

    # Fetch tensors to be loaded/stored/passed_through.
    loaded_ts = _get_loaded_ts(graph, load_handles)
    stored_ts = _get_stored_ts(graph, load_handles, store_streams, store_buffers)
    passed_through_ts = _get_passed_through_ts(graph, load_handles, seed_input)

    @popxl.in_sequence()
    def micro_graph_fn(
        index: popxl.Tensor,
        seed: popxl.Tensor,
        loaded: List[popxl.Tensor],
        stored_io: List[popxl.Tensor],
        passed_through: List[popxl.TensorByRef],
    ):
        seed, next_seed = ops.split_random_seed(seed)
        with popxl.io_tiles():
            load_index = index
            store_index = index - 2
            to_load = _apply_load_offsets(load_index, loaded_ts, load_handles, steps, rows)
            buffers_with_offset = _apply_store_offsets(store_index, stored_ts, store_buffers, steps, rows)
        # Overlapped
        with popxl.transforms.io_tile_exchange():
            loaded_io = _load_tensors(to_load)
            _store_tensors(stored_ts, stored_io, store_streams, buffers_with_offset)

        info = graph.call_with_info(
            args=_inputs_dict(loaded_ts, loaded, passed_through_ts, passed_through, seed_input, seed)
        )

        with popxl.io_tiles():
            loaded_io = _gather_tensors(load_handles, dict(zip(loaded_ts, loaded_io)))

        stored = [info.graph_to_parent(t) for t in stored_ts]
        # ---------
        loaded = _io_copy(loaded_io)

        stored_io = _io_copy(stored)
        with popxl.io_tiles():
            stored_io = _slice_tensors(store_buffers, dict(zip(stored_ts, stored_io)))
            index = index + 1

        return (index, next_seed, *loaded, *stored_io)

    repeat_graph = ir.create_empty_graph(suffix_graph_name(graph.graph.name, "_repeat_graph"))

    with repeat_graph, popxl.in_sequence():
        # Add inputs to returned graph.
        #   index, Seed, *Passed Through.
        index = popxl.graph_input([], popxl.int32, "batch_serial_loop_index")
        with popxl.io_tiles():
            index = ops.io_tile_copy(index)
            index = index * steps
        if seed_input is not None:
            seed = popxl.graph_input((2,), popxl.uint32, "batch_serial_random_seed")
        else:
            seed = popxl.constant((0, 0), popxl.uint32, "dummy_random_seed")
        passed_through = _add_passed_through_inputs(graph, passed_through_ts)

        # Load batch 0. Not overlapped
        with popxl.io_tiles():
            to_load = _apply_load_offsets(index, loaded_ts, load_handles, steps, rows)
        with popxl.transforms.io_tile_exchange():
            loaded_0 = _load_tensors(to_load)

        with popxl.io_tiles():
            loaded_0 = _gather_tensors(load_handles, dict(zip(loaded_ts, loaded_0)))

        loaded_0 = _io_copy(loaded_0)
        with popxl.io_tiles():
            index = index + 1

        seed_0, seed = ops.split_random_seed(seed)
        # Load batch 1. Compute batch 0. Overlapped
        with popxl.io_tiles():
            to_load = _apply_load_offsets(index, loaded_ts, load_handles, steps, rows)
        with popxl.transforms.io_tile_exchange():
            loaded_1 = _load_tensors(to_load)

        info = graph.call_with_info(
            args=_inputs_dict(loaded_ts, loaded_0, passed_through_ts, passed_through, seed_input, seed_0)
        )

        with popxl.io_tiles():
            loaded_1 = _gather_tensors(load_handles, dict(zip(loaded_ts, loaded_1)))

        stored_0 = [info.graph_to_parent(t) for t in stored_ts]

        # ---------
        loaded_1 = _io_copy(loaded_1)
        stored_0 = _io_copy(stored_0)
        with popxl.io_tiles():
            stored_0 = _slice_tensors(store_buffers, dict(zip(stored_ts, stored_0)))
            index = index + 1

        if steps - 2 > 0:
            # Repeat steps-2 batches.
            micro_graph = ir.create_graph(micro_graph_fn, index, seed, loaded_1, stored_0, passed_through)
            index, seed, *ts = ops.repeat(micro_graph, steps - 2, index, seed, *loaded_1, *stored_0, *passed_through)
            loaded_n = ts[: len(loaded_0)]
            stored_n_1 = ts[len(loaded_0) :]
        else:
            # If `steps == 2` then skip creating a 0 iteration loop.
            loaded_n = loaded_1
            stored_n_1 = stored_0

        with popxl.io_tiles():
            store_index = index - 2

        seed_n, seed = ops.split_random_seed(seed)
        # Store batch n-1. Compute batch n. Overlapped
        with popxl.io_tiles():
            buffers_with_offset = _apply_store_offsets(store_index, stored_ts, store_buffers, steps, rows)
        with popxl.transforms.io_tile_exchange():
            _store_tensors(stored_ts, stored_n_1, store_streams, buffers_with_offset)
        info = graph.call_with_info(
            args=_inputs_dict(loaded_ts, loaded_n, passed_through_ts, passed_through, seed_input, seed_n)
        )
        stored_n = [info.graph_to_parent(t) for t in stored_ts]
        # ---------
        stored_n = _io_copy(stored_n)
        with popxl.io_tiles():
            stored_n = _slice_tensors(store_buffers, dict(zip(stored_ts, stored_n)))
            store_index = store_index + 1

        # Store batch n. Not overlapped
        with popxl.io_tiles():
            buffers_with_offset = _apply_store_offsets(store_index, stored_ts, store_buffers, steps, rows)
        with popxl.transforms.io_tile_exchange():
            _store_tensors(stored_ts, stored_n, store_streams, buffers_with_offset)

    remap_dict = dict(zip(passed_through_ts, passed_through))

    # Remap NamedTensors
    named_args = graph.args.remap(remap_dict)
    named_graph = GraphWithNamedArgs(repeat_graph, named_args)

    stored_buffers = {}
    for t in filter(lambda t: t in graph.graph, store_buffers.keys()):
        if t in load_handles.keys() and not isinstance(load_handles[t], popxl.HostToDeviceStream):
            stored_buffers[t] = load_handles[t]
        else:
            stored_buffers[t] = store_buffers[t]

    return BatchSerialResult(named_graph, stored_buffers, remap_dict)


def batch_serialise(
    graph: GraphWithNamedArgs,
    steps: int,
    load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferWithOffset, RemoteHandle]],
    store_streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
    store_buffers: Dict[popxl.Tensor, Union[RemoteBufferWithOffset, RemoteHandle]],
    seed_input: Optional[popxl.Tensor] = None,
    rows: int = 1,
    io_mode: Literal["compute", "io", "io_overlapped"] = "io",
) -> BatchSerialResult:
    """Transform a Graph to repeat the computation `steps` times.

        Each iteration:
         * loads inputs as specified by `load_handles`.
         * calls `graph`
         * store outputs (and inputs) as specified by `store_streams` and `store_buffers`

        You can think to the RemoteBuffer as a matrix having `batch_index = 0 ... steps-1` labeling columns and `row_offset = 0 ... rows-1` labeling rows.
        A specific tensor identified by `(row_offset, batch_index)` is located in the remote buffer at `index = row_index * steps + batch_index`.
        When you specify a RemoteHandle you provide an int value which is used as row_offset: it will adjust the remote load/store index
        by `+(row_offset*steps)`.
        If the value of row_offset is None, the remote load/store index will be adjusted by `index % steps`, which provides the `batch_index` (first row will be accessed).

        Inputs that are not specified in `load_handles` will be added as inputs to the returned Graph.
        Outputs that are not specified in `store_streams` or `store_buffers` are not output from the returned Graph.

        See `docs/phased_execution.md` for more details.

    Args:
        graph (GraphWithNamedArgs): Graph to transform
        steps (int): Number of batch serialise steps
        load_handles (Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]]): Handles to load inputs before computation.
        store_streams (Dict[popxl.Tensor, popxl.DeviceToHostStream]): Streams to store outputs after computation.
        store_buffers (Dict[popxl.Tensor, RemoteHandle]): Buffers to store outputs (or inputs) after computation.
        seed_input (Optional[popxl.Tensor], optional): Input tensor of a random seed. Defaults to None.
        rows (int, optional): Increases the size of the remote buffers to allow for the returned Graph to be used with multiple input values. Defaults to 1.
        io_mode (Literal['compute', 'io', 'io_overlapped']): How to load/store the Tensors during the loop.
                                                             `compute` uses the Compute tiles.
                                                             `io` uses the IO tiles.
                                                             `io_overlapped` uses the io tiles and builds the loop such that Compute and IO execute at the same time.
    Returns:
        BatchSerialResult
    """
    # TODO remove when all code changed to use RemoteHandles
    load_handles = _convert_to_remote_handles(load_handles)
    store_buffers = _convert_to_remote_handles(store_buffers)

    # IO overlapped requires steps>=2. If steps == 1 default back to the non overlapped version with IO tiles.
    if io_mode == "io_overlapped" and steps < 2:
        logging.warning(
            "batch_serialisation with io_mode='io_overlapped' requires at least 3 steps. Falling back to io_mode='io'."
        )
        io_mode = "io"
    if io_mode in ["compute", "io"]:
        return batch_serialise_non_overlapped(
            graph, steps, load_handles, store_streams, store_buffers, seed_input, rows, use_io_tiles=io_mode == "io"
        )
    elif io_mode == "io_overlapped":
        return batch_serialise_overlapped(graph, steps, load_handles, store_streams, store_buffers, seed_input, rows)
    else:
        raise ValueError(f"Unknown 'io_mode' {io_mode}. Supported: (compute, io, io_overlapped)")


def batch_serialise_fwd_and_grad(
    forward_graph: GraphWithNamedArgs,
    gradient_graph: GraphWithNamedArgs,
    named_inputs_for_grad_graph: NamedTensors,
    steps: int,
    load_handles: Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteBufferWithOffset, RemoteHandle]],
    store_streams: Dict[popxl.Tensor, popxl.DeviceToHostStream],
    store_buffers: Dict[popxl.Tensor, Union[RemoteBufferWithOffset, RemoteHandle, popxl.ReplicaGrouping]],
    seed_input: Optional[popxl.Tensor] = None,
    rows: int = 1,
    io_mode: Literal["compute", "io", "io_overlapped"] = "io",
) -> Tuple[BatchSerialResult, BatchSerialResult]:
    """Transform a matching forward and gradient Graphs that the computation is `steps` times.

        Tensors required for autodiff will be stored automatically.

        See `popxl_addons.transforms.batch_serialise` and `docs/phased_execution.md` for more details.

    Args:
        forward_graph (GraphWithNamedArgs): Forward Graph to transform
        gradient_graph (GraphWithNamedArgs): Gradient Graph to transform
        named_inputs_for_grad_graph (NamedTensors): for each tensor provided here, a named input is added to the backward graph. Typically you want them to be the fwd variables, fwd.args
        steps (int): Number of batch serialise steps
        load_handles (Dict[popxl.Tensor, Union[popxl.HostToDeviceStream, RemoteHandle]]): Handles to load inputs before computation.
        store_streams (Dict[popxl.Tensor, popxl.DeviceToHostStream]): Streams to store outputs after computation.
        store_buffers (Dict[popxl.Tensor, RemoteHandle, popxl.ReplicaGrouping]): Dictionary between tensors and RemoteHandles specifying where to store outputs (or inputs) after computation.
                                                                                 If you want an activation to be sharded, you can provide an entry (activation, ReplicaGrouping) in the dictionary,
                                                                                 specifying the shard_group.
        seed_input (Optional[popxl.Tensor], optional): Input tensor of a random seed. Defaults to None.
        rows (int, optional): Increases the size of the remote buffers to allow for the returned Graph to be used with multiple input values. Defaults to 1.
        io_mode (Literal['compute', 'io', 'io_overlapped']): How to load/store the Tensors during the loop.
                                                             `compute` uses the Compute tiles.
                                                             `io` uses the IO tiles.
                                                             `io_overlapped` uses the io tiles and builds the loop such that Compute and IO execute at the same time.
    Returns:
        Tuple[BatchSerialResult, BatchSerialResult]:
            result of forward_graph, result of gradient_graph
    """

    def _filter_store_buffers(buffers: Dict[popxl.Tensor, Union[RemoteHandle, popxl.ReplicaGrouping]]):
        replica_groups = {t: rg for t, rg in buffers.items() if isinstance(rg, popxl.ReplicaGrouping)}
        buffers = {t: b for t, b in buffers.items() if not isinstance(b, popxl.ReplicaGrouping)}
        return buffers, replica_groups

    store_buffers, activations_shard_groups = _filter_store_buffers(store_buffers)
    grad_inputs = gradient_graph.graph.inputs
    grad_graph_info = gradient_graph.grad_graph_info

    # Handle storing of activations required for autodiff.
    activations = [
        i_ec
        for i_ec in enumerate(grad_graph_info.expected_inputs)
        if i_ec[1].connection_type == ExpectedConnectionType.Fwd
    ]
    for grad_idx, ec in activations:
        t = ec.fwd_tensor
        if t in store_buffers.keys() or t in named_inputs_for_grad_graph.tensors:
            # store buffer already specified.
            pass
        elif t in load_handles.keys() and not isinstance(load_handles[t], popxl.HostToDeviceStream):
            # activation already stored in the input handles
            store_buffers[t] = load_handles[t]
        else:
            # create a buffer for the activations
            shard_group = None
            if activations_shard_groups and t in activations_shard_groups:
                shard_group = activations_shard_groups[t]
            buffer = batch_serial_buffer(t, steps=steps, rows=rows, shard_group=shard_group)
            store_buffers[t] = RemoteHandle(buffer, 0, shard_group)

    forward_result = batch_serialise(
        forward_graph, steps, load_handles, store_streams, store_buffers, seed_input, rows, io_mode
    )

    # Handle loading of activations required for autodiff
    grad_load_handles = {}
    named_inputs = {}
    for idx, ec in activations:
        # add it to named inputs
        t = ec.fwd_tensor
        if t in named_inputs_for_grad_graph.tensors:
            name = list(named_inputs_for_grad_graph.named_tensors.keys())[
                list(named_inputs_for_grad_graph.named_tensors.values()).index(ec.fwd_tensor)
            ]
            named_inputs[name] = grad_inputs[idx]
        # load it from forward stored buffers
        elif t in forward_result.stored_buffers.keys():
            grad_load_handles[grad_inputs[idx]] = forward_result.stored_buffers[t]
        else:
            raise RuntimeError(
                f"Fwd Connection {t} missing. You need to provide it either in forward_variables or in load_handles"
            )

    # named args need to be in the same order as the gradient graph inputs
    named_inputs.update(gradient_graph.args.named_tensors)
    new_args = NamedTensors.from_dict(named_inputs)
    gradient_graph.args = new_args

    # Handle loading of gradients required for autodiff
    for idx, ec in filter(
        lambda i_ec: i_ec[1].connection_type == ExpectedConnectionType.FwdGrad,
        enumerate(grad_graph_info.expected_inputs),
    ):
        grad_t = grad_inputs[idx]
        if grad_t not in load_handles:
            raise ValueError(f"FwdGrad input (idx={idx} {grad_t}) to Gradient graph must have a load_handle specified")
        grad_load_handles[grad_t] = load_handles[grad_t]

    gradient_result = batch_serialise(
        gradient_graph, steps, grad_load_handles, store_streams, store_buffers, None, rows, io_mode
    )

    gradient_result.graph.grad_graph_info = remap_grad_info(grad_graph_info, forward_graph.graph, gradient_graph.graph)

    return forward_result, gradient_result
