# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import List, Optional, Tuple
import numpy as np

import popxl
from popxl import ReplicaGrouping, ops
from popxl.ops.collectives.collectives import CollectiveOps

from popxl_addons import GraphWithNamedArgs, NamedTensors
from popxl_addons.named_replica_grouping import NamedReplicaGrouping, get_ild_replica_grouping, get_ild_size_from_popdist, is_cross_ild
from popxl_addons.utils import null_context
from popxl_addons.ops.replicated_strided_collectives import (
    replicated_all_gather_strided, replicated_all_reduce_strided, replicated_reduce_scatter_strided)

__all__ = [
    "all_gather_replica_sharded_graph", "reduce_replica_sharded_graph", "reduce_replica_sharded_tensor",
    "gather_replica_sharded_tensor", "replica_sharded_spec"
]


def replica_sharded_spec(t: popxl.Tensor, shard_over: popxl.ReplicaGrouping) -> popxl.TensorSpec:
    # If t is already sharded then return the current spec
    if t.meta_shape:
        return t.spec
    # If shard_over is too small or t is not divisible by the group_size then return the current spec
    if shard_over.group_size == 1 or t.nelms % shard_over.group_size != 0:
        return t.spec
    # Otherwise return a spec with the current shard over
    shape = (int(np.prod(t.shape)) // shard_over.group_size, )
    return popxl.TensorSpec(shape, t.dtype, t.shape)


def gather_replica_sharded_tensor(t: popxl.Tensor,
                                  use_io_tiles: bool = False,
                                  replica_group: Optional[ReplicaGrouping] = None):
    """
    All gather a tensor using the provided replica_grouping. Optionally copy the result to IO Tiles.
    Args:
        t (popxl.Tensor): Input Tensor to replicated_all_gather.
        use_io_tiles (bool): If true, the result of replicated_all_gather is then copied from IO Tiles.
        replica_grouping (ReplicaGrouping): replica group used in all gather collective. If none, gather between all replicas.
    Returns:
        Tensor: gathered tensor

    """
    ir = popxl.gcg().ir
    replica_group = replica_group or ir.replica_grouping()

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
    replica_groups = replica_groups or NamedReplicaGrouping.build_groups(tensors.named_tensors.keys(),
                                                                         value=get_ild_replica_grouping())
    replica_groups = replica_groups.to_dict()

    with graph, popxl.in_sequence(False):
        for name, tensor in zip(*tensors.unpack()):
            sg_t = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name, meta_shape=tensor.meta_shape)
            args[name] = sg_t
            names.append(name)
            replica_group = replica_groups[name].const_rg
            sg_t = gather_replica_sharded_tensor(sg_t, use_io_tiles=use_io_tiles, replica_group=replica_group)
            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names


def reduce_replica_sharded_tensor(t: popxl.Tensor,
                                  op: CollectiveOps = 'add',
                                  replica_group: Optional[ReplicaGrouping] = None,
                                  shard_group: Optional[ReplicaGrouping] = None):
    """
    Reduce a tensor using op.
    Args:
        t (Tensor): Tensor to reduce.
        op (CollectiveOps): Operation to use for reduction.
        replica_group (ReplicaGrouping): Replica group to reduce the tensor. Default to all replicas. 
                                         If it spans across multiple instances, a two stage reduction is performed.
        shard_group (ReplicaGrouping): Replica group for rts. If not specified it will match replica_group
    Returns:
        Tensor: Fully reduced tensor
    """

    ir = popxl.gcg().ir
    replica_group = replica_group or ir.replica_grouping()  # defaults to all replicas
    shard_group = shard_group or replica_group  # defaults to replica_group

    if replica_group.group_size == 1:
        # No reduction required at all
        return t

    # RTS
    if shard_group.group_size > 1 and t.nelms % shard_group.group_size == 0:
        # A multi stage collective is required to keep the RTS behaviour within an ild
        second_stage = ir.replica_grouping(stride=get_ild_size_from_popdist()) if is_cross_ild(replica_group) else None
        reduce_group = get_ild_replica_grouping(replica_group)
        # RTS reduction
        if reduce_group == shard_group:
            t = replicated_reduce_scatter_strided(t,
                                                  op=op,
                                                  group=reduce_group,
                                                  configure_output_for_replicated_tensor_sharding=True)
        else:
            t = replicated_all_reduce_strided(t, group=reduce_group,
                                              op=op)  # all reduce across single ild reduce group only
            t = ops.collectives.replica_sharded_slice(t, group=shard_group)  # slice in rts group

        # if the replica group spans across multiple ilds, complete the reduction
        if second_stage:
            t = replicated_all_reduce_strided(t, op=op, group=second_stage)
    else:
        t = replicated_all_reduce_strided(t, group=replica_group, op=op)

    return t


def reduce_replica_sharded_graph(
        tensors: NamedTensors,
        op: CollectiveOps = 'add',
        use_io_tiles: bool = False,
        shard_groups: Optional[NamedReplicaGrouping] = None,
        replica_group: Optional[ReplicaGrouping] = None) -> Tuple[GraphWithNamedArgs, List[str]]:
    """Create a GraphWithNamedArgs that reduces each Tensor in `tensors` using op.
    The Graph with have a NamedArg for each the tensors in `tensors`.

    Usage:
    ```
        g, names = reduce_replica_sharded_graph(tensors)
        ...
        reduced_ts = NamedTensors.pack(names, g.bind(ts).call())
    ```

    Args:
        tensors (NamedTensors): Input Tensors to replica reduced.
        op (CollectiveOps): Operation to use for  reduction.
        use_io_tiles (bool, optional): If True, tensors will be copied to IO tiles before reducing. Defaults
            to False.
        shard_groups (NamedReplicaGrouping, optional): shard groups (rts) for each tensor in tensors. The
            result of the single instance reduction will be scattered in this group.
        replica_group (ReplicaGrouping, optional): full group to perform the reduction on (data parallel group). If the
            single instance restriction of this group is equal to the shard group of a tensor, a reduce_scatter
            collective will be used. Otherwise,  the tensor will be reduced over the replica group (single instance) and
            then sliced in the shard group (rts).
    Returns:
        Tuple[GraphWithNamedArgs, List[str]]: Created Graph, names of outputs from calling the graph.
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("replica_reduce")

    names = []
    args = {}

    tile_context = popxl.io_tiles() if use_io_tiles else null_context()
    shard_groups = shard_groups or NamedReplicaGrouping.build_groups(tensors.named_tensors.keys(),
                                                                     value=get_ild_replica_grouping())
    shard_groups = shard_groups.to_dict()

    with graph, popxl.in_sequence(False), tile_context:
        for name, tensor in zip(*tensors.unpack()):
            sg_t = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name, meta_shape=tensor.meta_shape)

            args[name] = sg_t
            names.append(name)

            if use_io_tiles:
                sg_t = ops.io_tile_copy(sg_t)

            sg_t = reduce_replica_sharded_tensor(sg_t, op, replica_group=replica_group, shard_group=shard_groups[name])

            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names
