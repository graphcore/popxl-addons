# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from contextlib import contextmanager
from typing import List, Tuple

import numpy as np

import popxl
from popxl import ops
from popxl.ops.collectives.collectives import CollectiveOps

from popxl_addons.dot_tree import DotTree
from popxl_addons import GraphWithNamedArgs, NamedInputFactories, NamedTensors

__all__ = [
    "all_gather_replica_sharded_graph", "reduce_replica_sharded_graph", "load_remote_graph", "store_remote_graph",
    "named_buffers", "named_input_buffers"
]


# Backported to python3.6
@contextmanager
def null_context():
    yield


def replica_sharded_spec(t: popxl.Tensor, threshold: int = 1024) -> popxl.TensorSpec:
    rf = popxl.gcg().ir.replication_factor
    if rf > 1 and t.nelms >= threshold and not t.meta_shape and t.nelms % rf == 0:
        shape = (int(np.prod(t.shape)) // rf, )
        return popxl.TensorSpec(shape, t.dtype, t.shape)
    return t.spec


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
    rf = popxl.gcg().ir.replication_factor
    buffers = {}
    for name, t in tensors.to_dict().items():
        spec = replica_sharded_spec(t, sharded_threshold)
        buffer = popxl.remote_buffer(spec.shape, spec.dtype, entries)
        buffer.meta_shape = spec.meta_shape
        buffers[name] = buffer
    return NamedRemoteBuffers.from_dict(buffers)


def named_input_buffers(inputs: NamedInputFactories, entries: int = 1, sharded_threshold: int = 1024):
    """Create a buffer for each InputFactory in `inputs`. The buffers will have `entries` set.
    Any factory with `nelms >= sharded_threshold` will have a replica sharded RemoteBuffer created instead.

    Args:
        inputs (NamedInputFactories): InputFactories to create buffers for.
        entries (int, optional): Number of entries of the buffer. Defaults to 1.
        sharded_threshold (int, optional): factories with nelms >= this will have replica sharded buffers. Defaults to 1024.

    Returns:
        NamedRemoteBuffers: A buffer for each factory with names matching the NamedInputFactories' names.
    """
    buffers = {}
    for name, f in inputs.to_dict().items():
        nelms = np.prod(f.shape)
        rf = popxl.gcg().ir.replication_factor
        if f.replica_sharded:
            buffer = popxl.remote_buffer(f.shape, f.dtype, entries)
            buffer.meta_shape = f.meta_shape
        elif popxl.gcg().ir.replication_factor > 1 and nelms >= sharded_threshold and nelms % rf == 0:
            shape = (nelms // rf, )
            buffer = popxl.remote_buffer(shape, f.dtype, entries)
            buffer.meta_shape = f.shape
        else:
            buffer = popxl.remote_buffer(f.shape, f.dtype, entries)
        buffers[name] = buffer
    return NamedRemoteBuffers.from_dict(buffers)


def load_remote_graph(buffers: NamedRemoteBuffers, entries: int = 1) -> Tuple[GraphWithNamedArgs, List[str]]:
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

    names = []
    with graph, popxl.transforms.merge_exchange(), popxl.in_sequence(False):
        index = popxl.graph_input([], popxl.int32, "load_index")

        for name, buffer in zip(*buffers.unpack()):
            buffer.entries = max(buffer.entries, entries)
            names.append(name)

            loaded = ops.remote_load(buffer, index, name)

            popxl.graph_output(loaded)

    return GraphWithNamedArgs(graph), names


def store_remote_graph(buffers: NamedRemoteBuffers, entries: int = 1) -> GraphWithNamedArgs:
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

    buffer_map = buffers.to_dict()
    args = {}
    with graph, popxl.transforms.merge_exchange(), popxl.in_sequence(False):
        index = popxl.graph_input([], popxl.int32, "store_index")

        for name, buffer in zip(*buffers.unpack()):
            buffer.entries = max(buffer.entries, entries)

            store_t = popxl.graph_input(buffer.tensor_shape, buffer.tensor_dtype, name, meta_shape=buffer.meta_shape)
            ops.remote_store(buffer_map[name], index, store_t)

            args[name] = store_t

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args))


def all_gather_replica_sharded_graph(tensors: NamedTensors,
                                     use_io_tiles: bool = False) -> Tuple[GraphWithNamedArgs, List[str]]:
    """Create a GraphWithNamedArgs that replicated_all_gathers each Tensor in `tensors`.
    The Graph with have a NamedArg for each the tensors in `tensors`.

    Usage:
    ```
        g, names = all_gather_replica_sharded_graph(tensors)
        ...
        gathered_ts = NamedTensors.pack(names, g.bind(ts).call())
    ```

    Args:
        tensors (NamedTensors): Input Tensors to replicated_all_gather.
        use_io_tiles (bool): If true, the result of replicated_all_gather is then copied from IO Tiles.

    Returns:
        Tuple[GraphWithNamedArgs, List[str]]: Created Graph, names of outputs from calling the graph.
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("all_gather")

    names = []
    args = {}

    with graph, popxl.in_sequence(False):
        for name, tensor in zip(*tensors.unpack()):
            sg_t = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name, meta_shape=tensor.meta_shape)

            args[name] = sg_t
            names.append(name)

            tile_context = popxl.io_tiles() if use_io_tiles else null_context()

            if sg_t.meta_shape:
                with tile_context:
                    sg_t = ops.collectives.replicated_all_gather(sg_t).reshape_(sg_t.meta_shape)

            if use_io_tiles:
                sg_t = ops.io_tile_copy(sg_t)

            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names


def reduce_replica_sharded_graph(tensors: NamedTensors,
                                 op: CollectiveOps = 'add',
                                 threshold: int = 1024,
                                 use_io_tiles: bool = False) -> Tuple[GraphWithNamedArgs, List[str]]:
    """Create a GraphWithNamedArgs that mean reduces each Tensor in `tensors`.
    The Graph with have a NamedArg for each the tensors in `tensors`.
    Tensors with `nelms >= threshold` will be reduce scattered for replica sharding, otherwise all_reduced.

    Usage:
    ```
        g, names = reduce_replica_sharded_graph(tensors)
        ...
        reduced_ts = NamedTensors.pack(names, g.bind(ts).call())
    ```

    Args:
        tensors (NamedTensors): Input Tensors to replica reduced.
        op (CollectiveOps): Operation to use for reduction.
        threshold (int, optional): Tensors with nelms >= this will be reduce scattered for replica sharding. Defaults to 1024.
        use_io_tiles (bool, optional): If True, tensors will be copied to IO tiles before reducing. Defaults to False.

    Returns:
        Tuple[GraphWithNamedArgs, List[str]]: Created Graph, names of outputs from calling the graph.
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("replica_reduce")

    names = []
    args = {}

    tile_context = popxl.io_tiles() if use_io_tiles else null_context()

    with graph, popxl.in_sequence(False), tile_context:
        for name, tensor in zip(*tensors.unpack()):
            sg_t = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name, meta_shape=tensor.meta_shape)

            args[name] = sg_t
            names.append(name)

            if use_io_tiles:
                sg_t = ops.io_tile_copy(sg_t)

            if sg_t.nelms >= threshold and sg_t.nelms % graph.ir.replication_factor == 0:
                sg_t = ops.collectives.replicated_reduce_scatter(sg_t,
                                                                 op=op,
                                                                 configure_output_for_replicated_tensor_sharding=True)
            else:
                sg_t = ops.collectives.replicated_all_reduce(sg_t, op=op)

            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names
