# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from modulefinder import ReplacePackage
from typing import Iterable, Optional, List

import popxl
from typing_extensions import Literal
from popxl import ops, ReplicaGrouping
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from popxl.errors import UndefinedValue
from popxl.ops.utils import check_in_graph, check_tensor_ipu_and_tile_set

# Auto compile cpp files
import cppimport.import_hook
# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import crossentropysharded_binding

__all__ = ['cross_entropy_sharded_loss']

REDUCTION_TYPE = Literal['mean', 'sum', 'none']


@op_debug_context
def cross_entropy_sharded_loss(logits: Tensor,
                               indices: Tensor,
                               ignore_index: Optional[Tensor] = None,
                               replica_grouping: Optional[ReplicaGrouping] = None,
                               reduction: REDUCTION_TYPE = 'mean',
                               available_memory_proportion: float = 0.4) -> Tensor:
    """
    Tensor Model Parallelism (TP) sharded cross-entropy loss,

    Logits are sharded across the devices along the class axis i.e. different data on each device. Logits should be
    of shape {n_samples, sharded_n_classes}. If you have multiple dimensions for n_samples (e.g. batch and sequence length
    for LMs) then you need to first flatten these two axes. Indices are broadcasted across devices i.e. the same data on each
    device. Indices should be of shape {n_samples} and have values for the true indices or labels. Indices should
    already be adjusted to account for the sharded class range.

    The forward op outputs the unreduced loss - this will be identical on all shards.

    Args:
        logits: shape {n_samples, sharded_n_classes}. Sharded between devices
        indices: shape {n_samples}. Same indices on all devices but pre-adjusted
        ignore_index (Optional[Tensor]): Specify label values that should not contribute to `l` or `dE/dx`.
            Defaults to None. The index should be adjusted to align with the indices.
        reduction (REDUCTION_TYPE): Specify how to reduce the loss. Defaults to `mean`. Options `mean`, `sum` and `none`
        replica_grouping (ReplicaGrouping): Tensor parallel replica group: ReplicaGrouping(stride=1, group_size=n_shards). Only stride 1 supported.
        available_memory_proportion: applied to the grouped multi-slice op used in the forwards and grad op

    Returns:
        loss: shape {n_samples}. Identical output on all devices
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, logits=logits, indices=indices)
    check_tensor_ipu_and_tile_set(logits=logits, indices=indices)

    settings = ctx._get_op_settings('cross_entropy_sharded')
    group = replica_grouping if replica_grouping is not None else g.ir.replica_grouping()

    op = crossentropysharded_binding.CrossEntropyShardedOp.createOpInGraph(
        pb_g, {
            0: logits.id,
            1: indices.id,
        }, {
            0: g._create_tensor_id(f"cross_entropy_sharded_out"),
            1: g._create_tensor_id(f"cross_entropy_sharded_soft_max_out"),
        },
        group=group._pb_replica_grouping,
        availableMemoryProportion=available_memory_proportion,
        settings=settings)

    loss = Tensor._from_pb_tensor(op.outTensor(0))

    if ignore_index is not None:
        mask = ops.cast(ops.logical_not(ops.equal(indices, ignore_index)), loss.dtype)
        loss = loss * mask

    if reduction == 'mean' and ignore_index is not None:
        loss = ops.sum(loss) / ops.sum(mask)
    elif reduction == 'mean':
        loss = ops.mean(loss)
    elif reduction == 'sum':
        loss = ops.sum(loss)
    elif reduction != 'none':
        raise ValueError(f"`reduction` should be one of `mean`, `sum` and `none`. Passed: {reduction}")

    return loss
