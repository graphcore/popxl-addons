# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook
# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import replicated_all_reduce_TP_binding

from typing import Optional

from popxl import Tensor, ReplicaGrouping
from popxl.context import op_debug_context, get_current_context
from popxl.ops.collectives.collectives import CollectiveOps, to_collective_op
from popxl.ops.utils import check_in_graph

__all__ = [
    'replicated_all_reduce_identical_inputs', 'replicated_all_reduce_identical_grad_inputs', 'replicated_all_reduce'
]


@op_debug_context
def replicated_all_reduce_identical_inputs(t: Tensor,
                                           op: CollectiveOps = 'add',
                                           group: Optional[ReplicaGrouping] = None) -> Tensor:
    """
    Replicated all reduce but where the input tensors are identical.

    This means the op is an identity but the corresponding grad op is a `replicated_all_reduce`.

    You must run the `OpToIdentityPattern` pattern (which is part of `PreAliasPatterns`) on all
    graph containing the op for it to work correctly. If you are using autodiff, you must run
    the pattern after it.

    Args:
        t (Tensor): Tensor to be reduced
        op (str, optional): Operation to reduce with. 'add' is currently only supported.
        group (Optional[ReplicaGrouping]): Replicas to reduce across. Defaults to All replicas.

    Returns:

    """
    return _replicated_all_reduce_TP(t, op, group, identical_inputs=True)


@op_debug_context
def replicated_all_reduce_identical_grad_inputs(t: Tensor,
                                                op: CollectiveOps = 'add',
                                                group: Optional[ReplicaGrouping] = None) -> Tensor:
    """
    Replicated all reduce but where the grad tensors of the corresponding grad op are identical.

    This means that this op is an `replicated_all_reduce` and the corresponding grad op an identity.

    You must run the `OpToIdentityPattern` pattern (which is part of `PreAliasPatterns`) on all
    graph containing the op for it to work correctly. If you are using autodiff, you must run
    the pattern after it.

    Args:
        t (Tensor): Tensor to be reduced
        op (str, optional): Operation to reduce with. 'add' is currently only supported.
        group (Optional[ReplicaGrouping]): Replicas to reduce across. Defaults to All replicas.

    Returns:

    """
    return _replicated_all_reduce_TP(t, op, group, identical_grad_inputs=True)


@op_debug_context
def replicated_all_reduce(t: Tensor, op: CollectiveOps = 'add', group: Optional[ReplicaGrouping] = None) -> Tensor:
    """
    Replicated all reduce that is differentiable.

    Args:
        t (Tensor): Tensor to be reduced
        op (str, optional): Operation to reduce with. 'add' is currently only supported.
        group (Optional[ReplicaGrouping]): Replicas to reduce across. Defaults to All replicas.

    Returns:

    """
    return _replicated_all_reduce_TP(t, op, group)


def _replicated_all_reduce_TP(
        t: Tensor,
        op: CollectiveOps = 'add',
        group: Optional[ReplicaGrouping] = None,
        identical_inputs: bool = False,
        identical_grad_inputs: bool = False,
) -> Tensor:

    op_ = to_collective_op(op)  # Only add is currently supported

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    group = g.ir.replica_grouping() if group is None else group

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('ReplicatedAllReduceTP')
    op = replicated_all_reduce_TP_binding.ReplicatedAllReduceTPOp.createOpInGraph(
        pb_g,
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("replicated_all_reduce_TP_out"),
        },
        op_,
        group._pb_replica_grouping,
        identical_inputs,
        identical_grad_inputs,
        settings,
    )
    ctx._op_created(op)

    return Tensor._from_pb_tensor(op.outTensor(0))
