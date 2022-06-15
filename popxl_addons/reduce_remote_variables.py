# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Optional, Tuple, List

import popxl
from popxl import ReplicaGrouping, ops
from popxl.ops.collectives.collectives import CollectiveOps

from popxl_addons import NamedTensors, GraphWithNamedArgs, null_context
from popxl_addons.ops.replicated_strided_collectives import replicated_reduce_scatter_strided, \
    replicated_all_reduce_strided


def reduce_replica_sharded_graph(
        tensors: NamedTensors,
        op: CollectiveOps = 'add',
        threshold: int = 1024,
        use_io_tiles: bool = False,
        use_data_parallel: bool = True,
        replica_grouping: Optional[ReplicaGrouping] = None) -> Tuple[GraphWithNamedArgs, List[str]]:
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
        replica_grouping (ReplicaGrouping): Replica group to reduce the gradients. Otherwise will be all replicas

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
                # RTS + DP
                sg_t = replicated_reduce_scatter_strided(sg_t,
                                                         rg=replica_grouping,
                                                         op=op,
                                                         configure_output_for_replicated_tensor_sharding=True)

            elif use_data_parallel:
                # DP
                sg_t = replicated_all_reduce_strided(sg_t, rg=replica_grouping, op=op)

            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names
