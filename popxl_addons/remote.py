# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from contextlib import contextmanager
from multiprocessing.sharedctypes import Value
from typing import List, Tuple, Union, Optional

import numpy as np

import popxl
from popxl import ReplicaGrouping, ops

from popxl_addons.dot_tree import DotTree
from popxl_addons import GraphWithNamedArgs, NamedVariableFactories, NamedTensors
from popxl_addons.named_replica_grouping import NamedReplicaGrouping, get_instance_replica_grouping, is_cross_instance
from popxl_addons.rts import replica_sharded_spec
from popxl_addons.utils import null_context

__all__ = ["load_remote_graph", "store_remote_graph", "named_buffers", "named_variable_buffers", "create_remote_buffer"]


class NamedRemoteBuffers(DotTree[popxl.RemoteBuffer]):
    pass


def named_buffers(tensors: NamedTensors, entries: int = 1, sharded_threshold: int = 1024) -> NamedRemoteBuffers:
    """Create a buffer for each Tensor in `tensors`. The buffers will have `entries` set.
    Any Tensor with `nelms >= sharded_threshold` will have a replica sharded RemoteBuffer created instead.

    Args:
        tensors (NamedTensors): Tensors to create buffers for.
        entries (int, optional): Number of entries of the buffer. Defaults to 1.
        sharded_threshold (int, optional): Tensors with nelms >= this will have replica sharded buffers. Defaults to 1024.

    Returns:
        NamedRemoteBuffers: A buffer for each Tensor with names matching the NamedTensors' names.
    """
    buffers = {}
    for name, t in tensors.to_dict().items():
        spec = replica_sharded_spec(t, sharded_threshold)
        buffer = popxl.remote_buffer(spec.shape, spec.dtype, entries)
        buffer.meta_shape = spec.meta_shape
        buffers[name] = buffer
    return NamedRemoteBuffers.from_dict(buffers)


def named_variable_buffers(factories: NamedVariableFactories,
                           entries: int = 1,
                           sharded_threshold: int = 1024,
                           shard_groups: Optional[NamedReplicaGrouping] = None):
    """Create a buffer for each VariableFactory in `factories`. The buffers will have `entries` set.
    Any factory with `nelms >= sharded_threshold` will have a replica sharded buffer over the instances instead.

    Args:
        factories (NamedVariableFactories): VariableFactories to create buffers for.
        entries (int, optional): Number of entries of the buffer. Defaults to 1.
        sharded_threshold (int, optional): factories with nelms >= this will have replica sharded buffers. Defaults to 1024.
        shard_groups (NamedReplicaGrouping, optional): rts groups for the factories.
    Returns:
        NamedRemoteBuffers: A buffer for each factory with names matching the NamedVariableFactories' names.
    """
    buffers = {}
    ir = popxl.gcg().ir

    shard_groups = shard_groups or factories.replica_groupings.map(get_instance_replica_grouping)
    shard_groups = shard_groups.to_dict()
    for name, f in factories.to_dict().items():
        tensor_spec = popxl.TensorSpec(shape=f.shape, dtype=f.dtype, meta_shape=f.meta_shape)
        replica_grouping = f.replica_grouping or ir.replica_grouping()
        buffers[name] = create_remote_buffer(tensor_spec,
                                             entries=entries,
                                             sharded_threshold=sharded_threshold,
                                             replica_group=f.replica_grouping,
                                             shard_group=shard_groups[name])
    return NamedRemoteBuffers.from_dict(buffers)


def create_remote_buffer(spec: popxl.TensorSpec,
                         entries: int = 1,
                         sharded_threshold: int = 1024,
                         replica_group: Optional[ReplicaGrouping] = None,
                         shard_group: Optional[ReplicaGrouping] = None):
    """Create a buffer given a TensorSpec and a replica grouping.
    If the spec has `nelms >= sharded_threshold` a replica sharded buffer over the instances will be produced instead.

    Args:
        spec (popxl.TensorSpec): tensor spec to create buffers for.
        entries (int, optional): Number of entries of the buffer. Defaults to 1.
        sharded_threshold (int, optional): if the spec has nelms >= this a replica sharded buffer will be created. Defaults to 1024.
        replica_group (ReplicaGrouping, optional): variable replica group
        shard_group (ReplicaGrouping, optional): rts replica group
    Returns:
        NamedRemoteBuffers: A buffer for each factory with names matching the NamedVariableFactories' names.
    """
    ir = popxl.gcg().ir
    nelms = np.prod(spec.shape)
    replica_group = replica_group or ir.replica_grouping()

    if is_cross_instance(shard_group):
        raise ValueError("The shard group must be restricted to a single instance. Use get_instance_replica_grouping.")

    # TODO add default shard

    if spec.meta_shape:
        buffer = popxl.remote_buffer(spec.shape, spec.dtype, entries)
        buffer.meta_shape = spec.meta_shape
    elif shard_group.group_size > 1 and nelms >= sharded_threshold and nelms % shard_group.group_size == 0:
        # Include replica_grouping dim
        shape = (replica_group.num_groups, *spec.shape) if replica_group.num_groups > 1 else spec.shape
        buffer = popxl.replica_sharded_buffer(shape, spec.dtype, replica_group, shard_group, entries)
    else:
        buffer = popxl.remote_buffer(spec.shape, spec.dtype, entries)
    return buffer


def load_remote_graph(buffers: NamedRemoteBuffers, entries: int = 1,
                      use_io_tiles: bool = False) -> Tuple[GraphWithNamedArgs, List[str]]:
    """Create a GraphWithNamedArgs that loads `buffers`.
    The graph will take one input that is the offset into each buffer to load.
    `entries` argument can be provided to resize each buffer as needed.

    Usage:
    ```
        g, names = load_remote_graph(buffers)
        ts = NamedTensors.pack(names, g.call(0))
    ```

    Args:
        buffers (NamedRemoteBuffers): Buffers to load
        entries (int, optional): Entries each buffer must have. Defaults to 1.

    Returns:
        Tuple[GraphWithNamedArgs, List[str]]: Created Graph, names of outputs from calling the graph.
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("load_remote")

    tile_context = popxl.io_tiles() if use_io_tiles else null_context()

    names = []
    with graph, popxl.transforms.merge_exchange(), popxl.in_sequence(False), tile_context:
        index = popxl.graph_input([], popxl.int32, "load_index")

        for name, buffer in zip(*buffers.unpack()):
            buffer.entries = max(buffer.entries, entries)
            names.append(name)

            loaded = ops.remote_load(buffer, index, name)

            popxl.graph_output(loaded)

    return GraphWithNamedArgs(graph), names


def load_remote(buffers: NamedRemoteBuffers, entry: Union[int, popxl.Tensor] = 0,
                use_io_tiles: bool = False) -> NamedTensors:
    tile_context = popxl.io_tiles() if use_io_tiles else null_context()

    loaded_ts = {}
    with popxl.transforms.merge_exchange(), popxl.in_sequence(False), tile_context:

        for name, buffer in buffers.to_dict().items():
            loaded = ops.remote_load(buffer, entry, name)

            loaded_ts[name] = loaded

    return NamedTensors.from_dict(loaded_ts)


def store_remote_graph(buffers: NamedRemoteBuffers, entries: int = 1, use_io_tiles: bool = False) -> GraphWithNamedArgs:
    """Create a GraphWithNamedArgs that stores tensors into `buffers`.
    The graph's first argument will be the offset into each buffer to load.
    The graph will have a NamedArg for each buffer in `buffer`.
    `entries` argument can be provided to resize each buffer as needed.

    Usage:
    ```
        g = store_remote_graph(buffers)
        g.bind(tensors).call(0)
    ```

    Args:
        buffers (NamedRemoteBuffers): Buffers to store into.
        entries (int, optional): Entries each buffer must have. Defaults to 1.

    Returns:
        GraphWithNamedArgs
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("store_remote")

    tile_context = popxl.io_tiles() if use_io_tiles else null_context()

    buffer_map = buffers.to_dict()
    args = {}
    with graph, popxl.transforms.merge_exchange(), popxl.in_sequence(False), tile_context:
        index = popxl.graph_input([], popxl.int32, "store_index")

        for name, buffer in zip(*buffers.unpack()):
            buffer.entries = max(buffer.entries, entries)

            store_t = popxl.graph_input(buffer.tensor_shape, buffer.tensor_dtype, name, meta_shape=buffer.meta_shape)
            ops.remote_store(buffer_map[name], index, store_t)

            args[name] = store_t

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args))


def store_remote(buffers: NamedRemoteBuffers,
                 tensors: NamedTensors,
                 entry: Union[int, popxl.Tensor] = 0,
                 use_io_tiles: bool = False):
    tile_context = popxl.io_tiles() if use_io_tiles else null_context()

    with popxl.transforms.merge_exchange(), popxl.in_sequence(False), tile_context:
        for buffer, tensor in buffers.to_mapping(tensors).items():
            ops.remote_store(buffer, entry, tensor)
