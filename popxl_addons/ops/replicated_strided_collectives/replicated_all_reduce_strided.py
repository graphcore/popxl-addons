# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook
# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import replicated_all_reduce_strided_binding

from typing import Tuple, List
import popxl
from popxl import Tensor, ReplicaGrouping, ops
from popxl.context import op_debug_context, get_current_context
from popxl.ops.collectives.collectives import CollectiveOps, to_collective_op
from popxl.ops.utils import check_in_graph
from popxl_addons import NamedTensors, GraphWithNamedArgs
from popxl_addons.utils import null_context

__all__ = [
    "replicated_all_reduce_strided", "replicated_all_reduce_strided_identical_inputs",
    "replicated_all_reduce_strided_identical_grad_inputs", "replicated_all_reduce_strided_graph"
]


@op_debug_context
def replicated_all_reduce_strided(t: Tensor, rg: ReplicaGrouping, op: CollectiveOps = "add") -> Tensor:
    """
    Replicated all reduce.

    Args:
        t (Tensor): Tensor to be reduced
        rg (ReplicaGrouping): Stride and group size used in the partition of the replicas.
        op (str, optional): Operation to reduce with. 'add' is currently only supported.

    Returns:

    """
    return _replicated_all_reduce_strided(
        t,
        rg.stride,
        rg.group_size,
        op,
        identical_inputs=False,
        identical_grad_inputs=False,
    )


@op_debug_context
def replicated_all_reduce_strided_identical_inputs(t: Tensor, rg: ReplicaGrouping, op: CollectiveOps = "add") -> Tensor:
    """
    Replicated all reduce.

    You must run the `OpToIdentityPattern` pattern (which is part of `PreAliasPatterns`) after applying autodiff
    for this op to work correctly.

    Args:
        t (Tensor): Tensor to be reduced
        rg (ReplicaGrouping): Stride and group size used in the partition of the replicas.
        op (str, optional): Operation to reduce with. 'add' is currently only supported.

    Returns:

    """
    return _replicated_all_reduce_strided(
        t,
        rg.stride,
        rg.group_size,
        op,
        identical_inputs=True,
        identical_grad_inputs=False,
    )


@op_debug_context
def replicated_all_reduce_strided_identical_grad_inputs(t: Tensor, rg: ReplicaGrouping,
                                                        op: CollectiveOps = "add") -> Tensor:
    """
    Replicated all reduce.

    You must run the `OpToIdentityPattern` pattern (which is part of `PreAliasPatterns`) after applying autodiff
    for this op to work correctly.

    Args:
        t (Tensor): Tensor to be reduced
        rg (ReplicaGrouping): Stride and group size used in the partition of the replicas.
        op (str, optional): Operation to reduce with. 'add' is currently only supported.

    Returns:

    """
    return _replicated_all_reduce_strided(
        t,
        rg.stride,
        rg.group_size,
        op,
        identical_inputs=False,
        identical_grad_inputs=True,
    )


def _replicated_all_reduce_strided(
        t: Tensor,
        stride: int,
        group_size: int,
        op: CollectiveOps,
        identical_inputs: bool,
        identical_grad_inputs: bool,
) -> Tensor:

    is_mean = op == "mean"
    if is_mean:
        op = "add"

    op_ = to_collective_op(op)  # Only add is currently supported

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("ReplicatedAllReduceStrided")
    op = replicated_all_reduce_strided_binding.ReplicatedAllReduceStridedOp.createOpInGraph(
        pb_g,
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("replicated_all_reduce_strided_out"),
        },
        op_,
        stride,
        group_size,
        identical_inputs,
        identical_grad_inputs,
        settings,
    )
    ctx._op_created(op)

    out = Tensor._from_pb_tensor(op.outTensor(0))
    if is_mean:
        out = out / group_size

    return out


def replicated_all_reduce_strided_graph(tensors: NamedTensors,
                                        group: ReplicaGrouping,
                                        op: CollectiveOps,
                                        use_io_tiles: bool = False) -> Tuple[GraphWithNamedArgs, List[str]]:
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

            sg_t = replicated_all_reduce_strided(sg_t, group, op)

            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names
