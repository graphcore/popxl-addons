# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from contextlib import contextmanager
from multiprocessing.sharedctypes import Value
from typing import List, Tuple, Union, Optional, Dict, Mapping
from webbrowser import get

import numpy as np

import popxl
from popxl import ReplicaGrouping, ops

from popxl_addons.dot_tree import DotTree
from popxl_addons import GraphWithNamedArgs, NamedVariableFactories, NamedTensors
from popxl_addons.rts import replica_sharded_spec
from popxl_addons.utils import null_context

__all__ = ["load_remote_graph", "store_remote_graph", "named_buffers", "named_variable_buffers", "create_remote_buffer"]


class NamedRemoteBuffers(DotTree[popxl.RemoteBuffer]):
    pass


def named_buffers(tensors: NamedTensors, entries: int = 1) -> NamedRemoteBuffers:
    """Create a buffer for each Tensor in `tensors`. The buffers will have `entries` set.

    Args:
        tensors (NamedTensors): Tensors to create buffers for.
        entries (int, optional): Number of entries of the buffer. Defaults to 1.

    Returns:
        NamedRemoteBuffers: A buffer for each Tensor with names matching the NamedTensors' names.
    """
    buffers = {}
    for name, t in tensors.to_dict().items():
        spec = replica_sharded_spec(t)
        buffer = popxl.remote_buffer(spec.shape, spec.dtype, entries)
        buffer.meta_shape = spec.meta_shape
        buffers[name] = buffer
    return NamedRemoteBuffers.from_dict(buffers)


def named_variable_buffers(factories: NamedVariableFactories,
                           entries: int = 1,
                           shard_over_dict: Union[Mapping[str, int], int, bool] = True) -> NamedRemoteBuffers:
    """Create a buffer for each VariableFactory in `factories`. The buffers will have `entries` set.

    Args:
        factories (NamedVariableFactories): VariableFactories to create buffers for.
        entries (int, optional): Number of entries of the buffer. Defaults to 1.
        shard_over_dict ( Union[Dict[str, int], int, bool], optional):  The number of IPUs to shard the tensor over in the replica group of the variable.
                                                                        For example, if the replica group is of size 4 and shard_over is 2, tensors will be
                                                                        sharded only on 2 ipus. A shard_over value of 1 implies no sharding.
                                                                        By default, sharded buffers will shard in the full replica group.
                                                                        If you don't want any sharding you can either set shard_over to False or 1.
    Returns:
        NamedRemoteBuffers: A buffer for each factory with names matching the NamedVariableFactories' names.
    """
    buffers = {}
    for name, f in factories.to_dict().items():
        tensor_spec = popxl.TensorSpec(shape=f.shape, dtype=f.dtype, meta_shape=f.meta_shape)
        if isinstance(shard_over_dict, Mapping):
            shard_over = shard_over_dict.get(name)
        elif isinstance(shard_over_dict, bool):
            shard_over = 1 if shard_over_dict == False else f.replica_grouping.group_size
        elif isinstance(shard_over_dict, int):
            if shard_over_dict < 1 or shard_over_dict > f.replica_grouping.group_size:
                raise ValueError(f"Invalid shard_over_dict int value: {shard_over_dict}")
            shard_over = shard_over_dict
        else:
            raise ValueError("shard_over_dict must be a dictionary, an integer or a boolean")

        buffers[name] = create_remote_buffer(tensor_spec,
                                             entries=entries,
                                             replica_group=f.replica_grouping,
                                             shard_over=shard_over)
    return NamedRemoteBuffers.from_dict(buffers)


def create_remote_buffer(spec: popxl.TensorSpec,
                         entries: int = 1,
                         replica_group: Optional[ReplicaGrouping] = None,
                         shard_over: Optional[int] = None) -> popxl.RemoteBuffer:
    """Create a buffer given a TensorSpec and a replica grouping.

    Args:
        spec (popxl.TensorSpec): tensor spec to create buffers for.
        entries (int, optional): Number of entries of the buffer. Defaults to 1.
        replica_group (ReplicaGrouping, optional): replica group for the tensor. Represent the devices where the tensor is equal, and the largest possible
                                                   set of devices for replicated tensor sharding. 
        shard_over (int, optional): number of replicas used to shard the tensor inside the provided replica group.
                                    See also `popxl.replica_sharded_buffer` documentation.
    Returns:
        NamedRemoteBuffers: A buffer for each factory with names matching the NamedVariableFactories' names.
    """
    ir = popxl.gcg().ir
    nelms = np.prod(spec.shape)
    replica_group = replica_group or ir.replica_grouping()
    shard_over = shard_over or replica_group.group_size
    # buffer for sharded tensor
    if spec.meta_shape:
        buffer = popxl.remote_buffer(spec.shape, spec.dtype, entries)
        buffer.meta_shape = spec.meta_shape
    # shard tensor and create buffer
    elif shard_over > 1:
        if nelms % shard_over != 0:
            raise ValueError(f"Can't shard tensor with {int(nelms)} elements over {shard_over} replicas")
        # Include replica_grouping dim
        shape = (replica_group.num_groups, *spec.shape) if replica_group.num_groups > 1 else spec.shape
        buffer = popxl.replica_sharded_buffer(shape, spec.dtype, replica_group, shard_over, entries)
    # standard buffer
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
