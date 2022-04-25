# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook

# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import replicated_all_gather_strided_binding

from typing import Optional

from popxl import Tensor, ReplicaGrouping
from popxl.context import op_debug_context, get_current_context
from popxl.ops.collectives.collectives import CollectiveOps, to_collective_op
from popxl.ops.utils import check_in_graph

__all__ = [
    "replicated_all_gather_strided",
]


@op_debug_context
def replicated_all_gather_strided(t: Tensor, rg: ReplicaGrouping) -> Tensor:
    """
    Replicated all gather.

    Args:
        t (Tensor): Tensor to be gathered.
        rg (ReplicaGrouping): Stride and group size used in the partition of the replicas.

    Returns:

    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("ReplicatedAllGatherStrided")
    op = replicated_all_gather_strided_binding.ReplicatedAllGatherStridedOp.createOpInGraph(
        pb_g,
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("replicated_all_gather_strided_out"),
        },
        rg.stride,
        rg.group_size,
        settings,
    )
    ctx._op_created(op)

    return Tensor._from_pb_tensor(op.outTensor(0))
