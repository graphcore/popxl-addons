# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from contextlib import contextmanager
from typing import List, Tuple
import numpy as np

import popxl
from popxl import ops
from popxl.ops.collectives.collectives import CollectiveOps

from popxl_addons import GraphWithNamedArgs, NamedTensors

__all__ = ["all_gather_replica_sharded_graph", "reduce_replica_sharded_graph"]


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


def all_gather_replica_sharded(tensors: NamedTensors, use_io_tiles: bool = False):
    gathered = {}
    with popxl.in_sequence(False):
        for name, tensor in tensors.to_dict().items():

            tile_context = popxl.io_tiles() if use_io_tiles else null_context()

            if tensor.meta_shape:
                with tile_context:
                    tensor = ops.collectives.replicated_all_gather(tensor).reshape_(tensor.meta_shape)

            if use_io_tiles:
                tensor = ops.io_tile_copy(tensor)

            gathered[name] = tensor

    return NamedTensors.from_dict(gathered)


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


def reduce_replica_sharded(tensors: NamedTensors,
                           op: CollectiveOps = 'add',
                           threshold: int = 1024,
                           use_io_tiles: bool = False) -> NamedTensors:
    reduced = {}
    tile_context = popxl.io_tiles() if use_io_tiles else null_context()

    with popxl.in_sequence(False), tile_context:
        for name, tensor in tensors.to_dict().items():

            if use_io_tiles:
                tensor = ops.io_tile_copy(tensor)

            if tensor.nelms >= threshold and tensor.nelms % popxl.gcg().ir.replication_factor == 0:
                tensor = ops.collectives.replicated_reduce_scatter(tensor,
                                                                   op=op,
                                                                   configure_output_for_replicated_tensor_sharding=True)
            else:
                tensor = ops.collectives.replicated_all_reduce(tensor, op=op)

            reduced[name] = tensor

    return NamedTensors.from_dict(reduced)
