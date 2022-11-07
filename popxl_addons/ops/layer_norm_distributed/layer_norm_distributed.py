# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook
# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import layer_norm_distributed_binding
import numpy as np

from typing import Optional, Tuple, Any

import popxl
from popxl import ReplicaGrouping
from popxl.context import op_debug_context, get_current_context
from popxl.ops.utils import check_in_graph, check_tensor_ipu_and_tile_set

__all__ = ['layer_norm_distributed']


@op_debug_context
def layer_norm_distributed(t: popxl.Tensor,
                           w: popxl.Tensor,
                           b: popxl.Tensor,
                           epsilon: float = 1e-5,
                           group: Optional[ReplicaGrouping] = None) -> popxl.Tensor:
    """
    Apply layer normalisation to a tensor that is sharded across it's hidden dimension within the group `replica_grouping`.

    Args:
        t (Tensor[N, H]): Tensor to be normalised.
        w (Tensor[H]): Tensor representing scale (gamma) in the learnable affine transformation.
        b (Tensor[H]): Tensor representing shift (beta) in the learnable affine transformation.
        epsilon (float): value added to denominator for numerical stability.
        group (Optional[ReplicaGrouping]): Replicas to reduce across. Defaults to all replicas.

    Returns:
        y (Tensor[N, H]): Normalised tensor of the same shape as the input.

    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    group = g.ir.replica_grouping() if group is None else group

    assert t.rank == 2
    assert w.rank == 1
    assert b.rank == 1

    check_in_graph(g, t=t, w=w, b=b)
    check_tensor_ipu_and_tile_set(t=t, w=w, b=b)

    settings = ctx._get_op_settings('LayerNormDistributedOp')
    op = layer_norm_distributed_binding.LayerNormDistributedOp.createOpInGraph(
        pb_g,
        {
            0: t.id,
            1: w.id,
            2: b.id
        },
        {
            0: g._create_tensor_id("outputs"),
            1: g._create_tensor_id("mean"),
            2: g._create_tensor_id("std"),
        },
        epsilon,
        group._pb_replica_grouping,
        settings,
    )
    ctx._op_created(op)

    return popxl.Tensor._from_pb_tensor(op.outTensor(0))
