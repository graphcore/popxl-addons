# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Iterable, Optional, List
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from popxl.errors import UndefinedValue
from popxl.ops.utils import check_in_graph

# Auto compile cpp files
import cppimport.import_hook

# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import crossentropysharded_wr_binding

__all__ = ["cross_entropy_sharded_wr_loss"]


def handle_ipus(ts: List[Tensor], ipus: Optional[Iterable[int]]) -> List[int]:
    """Check or obtain the ipus for each input tensor"""
    if ipus is None:
        try:
            ipus = [t.ipu for t in ts]
        except UndefinedValue as e:
            raise ValueError(
                "Could not automatically infer the IPU of all input Tensors. "
                "Please specify the IPUs via the `ipus` parameter."
            ) from e
    else:
        ipus = list(ipus)

    if len(ts) != len(ipus):
        raise ValueError(
            f"Number of specified tensor does not equal number of specified IPUs. " f"{len(ts)} != {len(ipus)}"
        )

    return ipus


@op_debug_context
def cross_entropy_sharded_wr_loss(
    logits: List[Tensor], indices: List[Tensor], ipus: Optional[Iterable[int]] = None
) -> Tensor:
    """
    Tensor Model Parallelism (TP) within replica sharded cross-entropy loss.

    Logits are sharded across the devices along the classes axis i.e. different data on each device. Logits should be
    of shape {n_samples, n_classes}. If you have multiple dimensions for n_samples (e.g. batch and sequence length
    for LMs) then you need to first flatten these two axes. Indices are broadcasted across devices i.e. the same data on each
    device. Indices should be of shape {n_samples} and have values for the true indices or labels. Indices will be
    adjusted to account for the sharding within the op and so there is no need to do so beforehand.

    The forward op outputs the unreduced loss onto the same device of the first logits tensor (IPU0).

    The number of shards needs to equal the number of IPU otherwise all-reduce might segfault due to a bug
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(
        g, **{f"logits_{i}": t for i, t in enumerate(logits)}, **{f"indices_{i}": t for i, t in enumerate(indices)}
    )

    n_shards = len(logits)

    if len(logits) != len(indices):
        raise ValueError("You must provide the same number of logits and indices.")

    ipus = handle_ipus(logits, ipus)
    ipus_indices = handle_ipus(indices, ipus)

    if not all(ipu_l == ipu_i for ipu_l, ipu_i in zip(ipus, ipus_indices)):
        raise ValueError("The logits and indices IPUs do not match.")

    settings = ctx._get_op_settings("cross_entropy_sharded")

    op = crossentropysharded_wr_binding.CrossEntropyShardedOp.createOpInGraph(
        pb_g,
        {i: t.id for i, t in enumerate(logits + indices)},
        {
            **{0: g._create_tensor_id(f"cross_entropy_sharded_out")},
            **{i + 1: g._create_tensor_id(f"cross_entropy_sharded_soft_max_out_{i}") for i in range(n_shards)},
        },
        ipus,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
