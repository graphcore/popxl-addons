# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook
# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import grad_reduce_square_add_binding

from typing import Union

import popxl
from popxl.context import op_debug_context, get_current_context
from popxl.ops.utils import check_in_graph, check_tensor_ipu_and_tile_set

__all__ = ['grad_reduce_square_add']


@op_debug_context
def grad_reduce_square_add(t: popxl.Tensor, loss_scaling: Union[int, float, popxl.Tensor]) -> popxl.Tensor:
    """Reduce Square Add a tensor accounting for any scaling of the values.
    This is helpful for calculating the norm of a loss scaled gradient.
    Usage:
    ```
    grad_norm = ops.sqrt(grad_reduce_square_add(grad, loss_scaling=64))
    ```

    Args:
        t (popxl.Tensor): Tensor to reduce.
        loss_scaling (Union[int, float, popxl.Tensor]): Scaling of the values in `t`.

    Returns:
        popxl.Tensor: Reduced Tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    scale = 1 / (loss_scaling * loss_scaling)

    if not isinstance(scale, popxl.Tensor):
        scale = popxl.constant(scale, popxl.float32)

    check_in_graph(g, t=t, loss_scaling=scale)
    check_tensor_ipu_and_tile_set(t=t, loss_scaling=scale)

    settings = ctx._get_op_settings('GradReduceSquareAdd')
    op = grad_reduce_square_add_binding.GradReduceSquareAddOp.createOpInGraph(
        pb_g,
        {
            0: t.id,
            1: scale.id
        },
        {
            0: g._create_tensor_id("grad_reduced"),
        },
        settings,
    )
    ctx._op_created(op)

    return popxl.Tensor._from_pb_tensor(op.outTensor(0))
