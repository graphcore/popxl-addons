# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook

# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import replicated_all_reduce_strided_binding

from typing import Optional

from popxl import Tensor, ReplicaGrouping
from popxl.context import op_debug_context, get_current_context
from popxl.ops.collectives.collectives import CollectiveOps, to_collective_op
from popxl.ops.utils import check_in_graph

__all__ = [
    "replicated_all_reduce_strided",
    "replicated_all_reduce_strided_identical_inputs",
    "replicated_all_reduce_strided_identical_grad_inputs",
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
        groupSize: int,
        op: CollectiveOps,
        identical_inputs: bool,
        identical_grad_inputs: bool,
) -> Tensor:

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
        groupSize,
        identical_inputs,
        identical_grad_inputs,
        settings,
    )
    ctx._op_created(op)

    return Tensor._from_pb_tensor(op.outTensor(0))
