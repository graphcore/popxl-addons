# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import wraps
from typing import Callable, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union, overload
from weakref import WeakKeyDictionary
from dataclasses import dataclass
from collections import defaultdict

import popxl
from popxl import ops
from popxl.context import debug_context_frame_offset, io_tiles
from popxl.tensor import Variable
from popxl.ops.call import CallSiteInfo
from popxl.remote_buffer import RemoteBuffer
from popxl.transforms.autodiff import ExpectedConnectionType, GradGraphInfo
from popxl.transforms.merge_exchange import io_tile_exchange
from popxl_addons.graph import BoundGraph, GraphWithNamedArgs

from popxl_addons.dot_tree import DotTree, to_mapping
from popxl_addons.graph_cache import GraphCache
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.remote import RemoteBuffers
from popxl_addons.dot_tree import sanitise

__all__ = [
    "remote_variables", "remote_replica_sharded_variables", "all_gather_replica_sharded_tensors",
    "all_gather_replica_sharded_tensors_io", "reduce_scatter_for_replica_sharding",
    "reduce_scatter_for_replica_sharding_io", "load_from_buffers", "load_to_io_tiles", "copy_from_io_tiles",
    "copy_to_io_tiles", "store_to_buffers", "store_from_io_tiles"
]

BufferEntry = Tuple[RemoteBuffer, Union[int, popxl.Tensor]]


class NamedBuffers(DotTree[BufferEntry]):
    pass


def remote_variables(named_tensors: NamedTensors, buffers: RemoteBuffers) -> NamedBuffers:
    remote = {}
    for key, tensor in named_tensors.to_dict().items():
        if isinstance(tensor, Variable):
            remote[key] = buffers.remote_variable(tensor)
    return NamedBuffers.from_dict(remote)


def remote_replica_sharded_variables(named_tensors: NamedTensors, buffers: RemoteBuffers,
                                     threshold: int = 1024) -> NamedBuffers:
    remote = {}
    for key, tensor in named_tensors.to_dict().items():
        if isinstance(tensor, Variable):
            if tensor.nelms >= threshold:
                remote[key] = buffers.replica_sharded_variable(tensor)
            else:
                remote[key] = buffers.remote_variable(tensor)
    return NamedBuffers.from_dict(remote)


@debug_context_frame_offset(1)
def all_gather_replica_sharded_tensors(named_tensors: NamedTensors) -> NamedTensors:
    gathered = {}
    for key, tensor in named_tensors.to_dict().items():
        if tensor.meta_shape:
            tensor = ops.collectives.replicated_all_gather(tensor).reshape_(tensor.meta_shape)
        gathered[key] = tensor
    return NamedTensors.from_dict(gathered)


@debug_context_frame_offset(1)
def all_gather_replica_sharded_tensors_io(named_tensors: NamedTensors) -> NamedTensors:
    with io_tiles():
        return all_gather_replica_sharded_tensors(named_tensors)


@debug_context_frame_offset(1)
def reduce_scatter_for_replica_sharding(named_tensors: NamedTensors, threshold: int = 1024) -> NamedTensors:
    reduced = {}
    for key, tensor in named_tensors.to_dict().items():
        if tensor.nelms >= threshold:
            reduced[key] = ops.collectives.replicated_reduce_scatter(tensor, 'mean', None, True)
        else:
            reduced[key] = ops.collectives.replicated_all_reduce_(tensor, 'mean')
    return NamedTensors.from_dict(reduced)


@debug_context_frame_offset(1)
def reduce_scatter_for_replica_sharding_io(named_tensors: NamedTensors, threshold: int = 1024) -> NamedTensors:
    with io_tiles():
        return reduce_scatter_for_replica_sharding(named_tensors, threshold)


@overload
def load_from_buffers(buffers: NamedBuffers) -> NamedTensors:
    ...


@overload
def load_from_buffers(*buffers: NamedBuffers) -> Tuple[NamedTensors, ...]:
    ...


@debug_context_frame_offset(3)
@popxl.transforms.merge_exchange()
@popxl.in_sequence(False)
def load_from_buffers(*buffers: NamedBuffers):
    loaded = []
    entry_to_tensor = {}
    for named_buffer in buffers:
        load_dict = {}
        for key, entry in named_buffer.to_dict().items():
            if entry in entry_to_tensor.keys():
                load_dict[key] = entry_to_tensor[entry]
            else:
                t = ops.remote_load(entry[0], entry[1], key)
                load_dict[key] = t
                entry_to_tensor[entry] = t
        loaded.append(NamedTensors.from_dict(load_dict))
    if len(loaded) == 1:
        return loaded[0]
    return tuple(loaded)


@overload
def load_to_io_tiles(buffers: NamedBuffers) -> NamedTensors:
    ...


@overload
def load_to_io_tiles(*buffers: NamedBuffers) -> Tuple[NamedTensors, ...]:
    ...


@debug_context_frame_offset(2)
@io_tile_exchange()
def load_to_io_tiles(*buffers: NamedBuffers) -> Union[NamedTensors, Tuple[NamedTensors, ...]]:
    return load_from_buffers(*buffers)


@overload
def copy_from_io_tiles(named_tensors: NamedTensors) -> NamedTensors:
    ...


@overload
def copy_from_io_tiles(*named_tensors: NamedTensors) -> Tuple[NamedTensors, ...]:
    ...


@debug_context_frame_offset(1)
def copy_from_io_tiles(*named_tensors: NamedTensors):
    copied = []
    io_to_compute = {}
    for named_tensor in named_tensors:
        copy_dict = {}
        for key, tensor in named_tensor.to_dict().items():
            if tensor in io_to_compute.keys():
                copy_dict[key] = io_to_compute[tensor]
            else:
                copy_dict[key] = io_to_compute[tensor] = ops.io_tile_copy(tensor)
        copied.append(NamedTensors.from_dict(copy_dict))
    if len(copied) == 1:
        return copied[0]
    return tuple(copied)


@overload
def copy_to_io_tiles(named_tensors: NamedTensors) -> NamedTensors:
    ...


@overload
def copy_to_io_tiles(*named_tensors: NamedTensors) -> Tuple[NamedTensors, ...]:
    ...


@debug_context_frame_offset(2)
@io_tiles()
def copy_to_io_tiles(*named_tensors: NamedTensors):
    copied = []
    compute_to_io = {}
    for named_tensor in named_tensors:
        copy_dict = {}
        for key, tensor in named_tensor.to_dict().items():
            if tensor in compute_to_io.keys():
                copy_dict[key] = compute_to_io[tensor]
            else:
                copy_dict[key] = compute_to_io[tensor] = ops.io_tile_copy(tensor)
        copied.append(NamedTensors.from_dict(copy_dict))
    if len(copied) == 1:
        return copied[0]
    return tuple(copied)


@debug_context_frame_offset(1)
def store_to_buffers(named_tensors: NamedTensors, buffers: NamedBuffers) -> None:
    to_store = to_mapping(named_tensors, buffers)
    for tensor, (buffer, offset) in to_store.items():
        ops.remote_store(buffer, offset, tensor)


@debug_context_frame_offset(2)
@io_tile_exchange()
def store_from_io_tiles(named_tensors: NamedTensors, buffers: NamedBuffers) -> None:
    store_to_buffers(named_tensors, buffers)


@dataclass
class RemoteActivations:
    to_store: NamedTensors
    buffers: NamedBuffers
    _subgraph: NamedTensors

    def activation_map(self, loaded: NamedTensors):
        return to_mapping(self._subgraph, loaded)


def remote_activations(call_info: CallSiteInfo,
                       grad_info: GradGraphInfo,
                       buffers: RemoteBuffers,
                       existing: Optional[Mapping[popxl.Tensor, BufferEntry]] = None) -> RemoteActivations:
    return remote_activations_from_map(grad_info.inputs_dict(call_info), buffers, existing)


def activations_from_subgraph_io_map(subgraph_io_map: Mapping[popxl.Tensor, popxl.Tensor], grad_info: GradGraphInfo):
    return {
        popxl.Tensor._from_pb_tensor(grad_info.graph._pb_graph.getInputTensor(idx)): subgraph_io_map[act.fwd_tensor]
        for idx, act in enumerate(grad_info.expected_inputs) if act.connection_type == ExpectedConnectionType.Fwd
    }


def remote_activations_from_subgraph_io_map(
        subgraph_io_map: Mapping[popxl.Tensor, popxl.Tensor],
        grad_info: GradGraphInfo,
        buffers: RemoteBuffers,
        existing: Optional[Mapping[popxl.Tensor, BufferEntry]] = None) -> RemoteActivations:
    return remote_activations_from_map(activations_from_subgraph_io_map(subgraph_io_map, grad_info), buffers, existing)


def remote_activations_from_map(acts_map: Mapping[popxl.Tensor, popxl.Tensor],
                                buffers: RemoteBuffers,
                                existing: Optional[Mapping[popxl.Tensor, BufferEntry]] = None):
    existing = existing or {}
    to_store = {}
    _buffers = {}
    subgraph = {}

    for sg_t, t in acts_map.items():
        key = sanitise(sg_t.name)
        if t in existing.keys():
            _buffers[key] = existing[t]
        else:
            buffer = buffers.get_buffer(t.shape, t.dtype)
            offset = buffer.entries
            buffer.entries += 1
            to_store[key] = t
            _buffers[key] = (buffer, offset)
        subgraph[key] = sg_t

    return RemoteActivations(NamedTensors.from_dict(to_store), NamedBuffers.from_dict(_buffers),
                             NamedTensors.from_dict(subgraph))


def _make_constant(ts: Iterable[Union[int, popxl.Tensor]]):
    return tuple(map(lambda t: t if isinstance(t, popxl.Tensor) else popxl.constant(t, popxl.uint32), ts))


R = TypeVar('R')


def _static_graph_cache(fn: Callable[..., R]) -> Callable[..., R]:
    cache = GraphCache()

    def wrapper(*args, **kwargs) -> R:
        return fn(cache, *args, **kwargs)

    return wrapper


@_static_graph_cache
def _load_from_buffers_function(cache: GraphCache, buffers: NamedBuffers) -> NamedTensors:
    names: List[str]

    def flat_fn(ids: List[RemoteBuffer], offsets: List[popxl.Tensor]) -> List[popxl.Tensor]:
        nonlocal names
        output = []
        for buffer, offset, name in zip(ids, offsets, names):
            output.append(ops.remote_load(buffer, offset, name))
        return output

    buffer_names, entries = buffers.unpack()

    # Uniquify the buffer entries to avoid loading the same value twice
    unique_entries = []
    names = []
    seen = []
    index_names = defaultdict(list)
    for idx, entry in enumerate(entries):
        entry_hash = hash(entry)
        try:
            entry_idx = seen.index(entry_hash)
        except ValueError:
            entry_idx = len(names)
            names.append(buffer_names[idx])
            unique_entries.append(entry)
            seen.append(entry_hash)
        index_names[entry_idx].append(buffer_names[idx])

    ids, offsets = zip(*unique_entries)
    offset_ts = _make_constant(offsets)

    graph = cache.create_graph(flat_fn, ids, offset_ts)

    outputs = ops.call(graph, *offset_ts)
    result = {}
    for idx, t in enumerate(outputs):
        for name in index_names[idx]:
            result[name] = t
    return NamedTensors.from_dict(result)


def load_from_buffers_function(buffers: NamedBuffers) -> NamedTensors:
    return _load_from_buffers_function(buffers)


@_static_graph_cache
def _store_to_buffers_function(cache: GraphCache, tensors: NamedTensors, buffers: NamedBuffers) -> NamedTensors:
    def flat_fn(tensors: List[popxl.Tensor], ids: List[RemoteBuffer], offsets: List[popxl.Tensor]):
        for buffer, offset, tensor in zip(ids, offsets, tensors):
            ops.remote_store(buffer, offset, tensor)

    tensor_buffer_map = tensors.to_mapping(buffers)
    tensors, entries = zip(*tensor_buffer_map.items())
    ids, offsets = zip(*entries)

    # Check each buffer is only written to once.
    assert len(entries) == len(set(entries))

    offset_ts = _make_constant(offsets)

    graph = cache.create_graph(flat_fn, tensors, ids, offset_ts)

    ops.call(graph, tensors, offset_ts)


def store_to_buffers_function(tensors: NamedTensors, buffers: NamedBuffers):
    return _store_to_buffers_function(tensors, buffers)
