# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook
# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import replicated_reduce_scatter_strided_binding

from popxl import Tensor, ReplicaGrouping
from popxl.context import op_debug_context, get_current_context
from popxl.ops.collectives.collectives import CollectiveOps, to_collective_op
from popxl.ops.utils import check_in_graph

__all__ = [
    "replicated_reduce_scatter_strided",
]


@op_debug_context
def replicated_reduce_scatter_strided(
        t: Tensor,
        rg: ReplicaGrouping,
        op: CollectiveOps = "add",
        configure_output_for_replicated_tensor_sharding: bool = False,
) -> Tensor:
    """
    Replicated reduce scatter.

    Args:
        t (Tensor): Tensor to be gathered.
        rg (ReplicaGrouping): Stride and group size used in the partition of the replicas.
        op (str, optional): Operation to reduce with. 'add' is currently only supported.
        configure_output_for_replicated_tensor_sharding (bool): Configures the Op for replicated tensor sharding if True.
    Returns:

    """
    op_ = to_collective_op(op)  # Only add is currently supported

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("ReplicatedReduceScatterStrided")
    op = replicated_reduce_scatter_strided_binding.ReplicatedReduceScatterStridedOp.createOpInGraph(
        pb_g,
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("replicated_reduce_scatter_strided_out"),
        },
        op_,
        rg.stride,
        rg.group_size,
        configure_output_for_replicated_tensor_sharding,
        settings,
    )
    ctx._op_created(op)

    return Tensor._from_pb_tensor(op.outTensor(0))