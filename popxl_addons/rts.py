# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import List, Optional, Tuple
import numpy as np

import popxl
from popxl import ReplicaGrouping, ops
from popxl.ops.collectives.collectives import CollectiveOps

from popxl_addons import GraphWithNamedArgs, NamedTensors
from popxl_addons.named_replica_grouping import NamedReplicaGrouping, get_instance_replica_grouping, is_cross_instance
from popxl_addons.utils import null_context
from popxl_addons.ops.replicated_strided_collectives import (
    replicated_all_gather_strided, replicated_all_reduce_strided, replicated_reduce_scatter_strided)

__all__ = [
    "all_gather_replica_sharded_graph", "replica_sharded_slice_graph", "reduce_replica_sharded_graph",
    "reduce_replica_sharded_tensor", "gather_replica_sharded_tensor"
]


def replica_sharded_spec(t: popxl.Tensor,
                         threshold: int = 1024,
                         replica_grouping: Optional[popxl.ReplicaGrouping] = None) -> popxl.TensorSpec:
    ir = popxl.gcg().ir
    group = replica_grouping or ir.replica_grouping()
    shard_size = ir.instance_replication_factor // group.num_groups
    if shard_size > 1 and t.nelms >= threshold and not t.meta_shape and t.nelms % shard_size == 0:
        shape = (int(np.prod(t.shape)) // shard_size, )
        return popxl.TensorSpec(shape, t.dtype, t.shape)
    return t.spec


def gather_replica_sharded_tensor(t: popxl.Tensor,
                                  use_io_tiles: bool = False,
                                  replica_group: Optional[ReplicaGrouping] = None):
    """
    All gather a tensor using the provided replica_grouping. Optionally copy the result to IO Tiles.
    Args:
        t (popxl.Tensor): Input Tensor to replicated_all_gather.
        use_io_tiles (bool): If true, the result of replicated_all_gather is then copied from IO Tiles.
        replica_grouping (ReplicaGrouping): replica group used in all gather collective. If none, gather between all replicas in a single instance.
    Returns:
        Tensor: gathered tensor

    """
    ir = popxl.gcg().ir
    replica_group = replica_group or ir.replica_grouping()

    replica_group = get_instance_replica_grouping(replica_group)

    tile_context = popxl.io_tiles() if use_io_tiles else null_context()
    if t.meta_shape:
        with tile_context:
            t = replicated_all_gather_strided(t, group=replica_group)
    if use_io_tiles:
        t = ops.io_tile_copy(t)

    return t


def all_gather_replica_sharded_graph(
        tensors: NamedTensors,
        use_io_tiles: bool = False,
        replica_groups: Optional[NamedReplicaGrouping] = None,
) -> Tuple[GraphWithNamedArgs, List[str]]:
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
        replica_groups (NamedReplicaGrouping, optional): replica groups for tensors to be used in the gather collective. Default all replicas in a single instance.

    Returns:
        Tuple[GraphWithNamedArgs, List[str]]: Created Graph, names of outputs from calling the graph.
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("all_gather")

    names = []
    args = {}
    full_instance_group = ir.replica_grouping(stride=1, group_size=ir.instance_replication_factor)
    replica_groups = replica_groups or NamedReplicaGrouping.build_groups(
        tensors.named_tensors.keys(), value=full_instance_group)  # defaults to all replicas in a single instance
    replica_groups = replica_groups.to_dict()

    with graph, popxl.in_sequence(False):
        for name, tensor in zip(*tensors.unpack()):
            sg_t = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name, meta_shape=tensor.meta_shape)
            args[name] = sg_t
            names.append(name)
            replica_group = replica_groups[name]
            sg_t = gather_replica_sharded_tensor(sg_t, use_io_tiles=use_io_tiles, replica_group=replica_group)
            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names


def reduce_replica_sharded_tensor(t: popxl.Tensor,
                                  op: CollectiveOps = 'add',
                                  threshold: int = 1024,
                                  replica_group: Optional[ReplicaGrouping] = None,
                                  shard_group: Optional[ReplicaGrouping] = None):
    """
    Reduce a tensor using op.
    Tensors with `nelms >= threshold` will be reduce scattered for replica sharding, otherwise all_reduced.
    Args:
        t (Tensor): Tensor to reduce.
        op (CollectiveOps): Operation to use for reduction.
        threshold (int, optional): Tensors with nelms >= this will be reduce scattered for replica sharding. Defaults to 1024.
        replica_group (ReplicaGrouping): Replica group to reduce the tensor. Default to all replicas. 
                                         If it spans across multiple instances, a two stage reduction is performed.
        shard_group (ReplicaGrouping): Replica group for rts. This is always between a single instance.
    Returns:
        Tensor: Fully reduced tensor
    """
    if shard_group and is_cross_instance(shard_group):
        raise ValueError("Shard group should be inside a single instance. Please use get_instance_replica_grouping")

    ir = popxl.gcg().ir
    replica_group = replica_group or ir.replica_grouping()  # defaults to all replicas
    shard_group = shard_group or get_instance_replica_grouping(
        replica_group)  # defaults to all replicas in a single instance

    if replica_group.group_size == 1:
        # No reduction required at all
        return t

    # RTS
    if t.nelms >= threshold and t.nelms % ir.replication_factor == 0:
        # A multi stage collective is required to keep the RTS behaviour within an instance
        second_stage = ir.replica_grouping(
            stride=ir.instance_replication_factor) if is_cross_instance(replica_group) else None
        reduce_group = get_instance_replica_grouping(replica_group)
        # RTS reduction
        if reduce_group == shard_group:
            t = replicated_reduce_scatter_strided(t,
                                                  op=op,
                                                  group=reduce_group,
                                                  configure_output_for_replicated_tensor_sharding=True)
        else:
            t = replicated_all_reduce_strided(t, group=reduce_group,
                                              op=op)  # all reduce across single instance reduce group only
            t = ops.collectives.replica_sharded_slice(t, group=shard_group)  # slice in rts group

        # if the replica group spans across multiple instances, complete the reduction
        if second_stage:
            t = replicated_all_reduce_strided(t, op=op, group=second_stage)
    else:
        t = replicated_all_reduce_strided(t, group=replica_group, op=op)

    return t


def reduce_replica_sharded_graph(
        tensors: NamedTensors,
        op: CollectiveOps = 'add',
        threshold: int = 1024,
        use_io_tiles: bool = False,
        shard_groups: Optional[NamedReplicaGrouping] = None,
        replica_group: Optional[ReplicaGrouping] = None) -> Tuple[GraphWithNamedArgs, List[str]]:
    """Create a GraphWithNamedArgs that reduces each Tensor in `tensors` using op.
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
        shard_groups (NamedReplicaGrouping, optional): shard groups (rts) for each tensor in tensors. The result of the single instance reduction will be scattered
                                                       in this group.
        replica_group (ReplicaGrouping, optional): full group to perform the reduction on (data parallel group). If the single instance restriction of this group
                                                   is equal to the shard group of a tensor, a reduce_scatter collective will be used.
                                                   Otherwise, the tensor will be reduced over the replica group (single instance) and then sliced in the shard group (rts).
    Returns:
        Tuple[GraphWithNamedArgs, List[str]]: Created Graph, names of outputs from calling the graph.
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("replica_reduce")

    names = []
    args = {}

    tile_context = popxl.io_tiles() if use_io_tiles else null_context()
    full_instance_group = ir.replica_grouping(stride=1, group_size=ir.instance_replication_factor)
    shard_groups = shard_groups or NamedReplicaGrouping.build_groups(tensors.named_tensors.keys(),
                                                                     value=full_instance_group)
    shard_groups = shard_groups.to_dict()

    with graph, popxl.in_sequence(False), tile_context:
        for name, tensor in zip(*tensors.unpack()):
            sg_t = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name, meta_shape=tensor.meta_shape)

            args[name] = sg_t
            names.append(name)

            if use_io_tiles:
                sg_t = ops.io_tile_copy(sg_t)

            sg_t = reduce_replica_sharded_tensor(sg_t,
                                                 op,
                                                 threshold,
                                                 replica_group=replica_group,
                                                 shard_group=shard_groups[name])

            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names


def replica_sharded_slice_graph(
        tensors: NamedTensors,
        threshold: int = 1024,
        use_io_tiles: bool = False,
        shard_groups: Optional[NamedReplicaGrouping] = None) -> Tuple[GraphWithNamedArgs, List[str]]:
    """Create a GraphWithNamedArgs that replica_sharded_slices `t` in `tensors`.
    The Graph with have a NamedArg for each the tensors in `tensors`.
    Tensors with `nelms < threshold` will be untouched.

    Usage:
    ```
        g, names = replica_sharded_slice_graph(tensors)
        ...
        sliced_ts = NamedTensors.pack(names, g.bind(ts).call())
    ```

    Args:
        tensors (NamedTensors): Input Tensors to replica sharded sliced.
        threshold (int, optional): Tensors with nelms >= this will be reduce scattered for replica sharding. Defaults to 1024.
        use_io_tiles (bool, optional): If True, tensors will be copied to IO tiles before reducing. Defaults to False.
        shard_groups (NamedReplicaGrouping, optional): replica groupings for tensors.

    Returns:
        Tuple[GraphWithNamedArgs, List[str]]: Created Graph, names of outputs from calling the graph.
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("replica_reduce")

    names = []
    args = {}

    tile_context = popxl.io_tiles() if use_io_tiles else null_context()
    full_instance_group = ir.replica_grouping(stride=1, group_size=ir.instance_replication_factor)
    shard_groups = shard_groups or NamedReplicaGrouping.build_groups(tensors.named_tensors.keys(),
                                                                     value=full_instance_group)
    shard_groups = shard_groups.to_dict()

    with graph, popxl.in_sequence(False), tile_context:
        for name, tensor in zip(*tensors.unpack()):
            sg_t = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name, meta_shape=tensor.meta_shape)

            args[name] = sg_t
            names.append(name)

            if use_io_tiles:
                sg_t = ops.io_tile_copy(sg_t)

            if sg_t.nelms >= threshold and sg_t.nelms % graph.ir.replication_factor == 0:
                sg_t = ops.collectives.replica_sharded_slice(sg_t, group=shard_groups[name])

            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names
