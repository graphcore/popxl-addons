# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook

# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import linear_layout_impl

import popxl
from popxl.context import op_debug_context, get_current_context
from popxl.ops.utils import check_in_graph, check_tensor_ipu_and_tile_set
from popxl.tensor import Tensor

__all__ = [
    "linear_layout",
]


@op_debug_context
def linear_layout(t: popxl.Tensor, bs: int = 1) -> popxl.Tensor:
    """
    Change the tile mapping to linear mapping. The number of elements mapped to each tile will be an integer multiple of the batch size `bs`.

    Args:
        t (Tensor): Tensor to be remapped.
        bs (int): batch size. This is the grainSize to map linearly on tile.

    Returns:

    """
    ctx = get_current_context()
    graph = ctx.graph
    pb_graph = graph._pb_graph

    check_in_graph(graph, t=t)

    settings = ctx._get_op_settings("linear_layout")
    params = linear_layout_impl.LinearLayoutParams(bs)
    op = linear_layout_impl.LinearLayoutOp.create_op_in_graph(
        graph=pb_graph,
        inputs={0: t.id},
        outputs={0: graph._create_tensor_id("linear_layout")},
        params=params,
        settings=settings,
    )
    ctx._op_created(op)
    return Tensor._from_pb_tensor(op.outTensor(0))
