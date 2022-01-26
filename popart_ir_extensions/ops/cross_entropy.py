# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Union
import popart.ir as pir
import popart.ir.ops as ops

__all__ = ["cross_entropy_with_grad"]


def cross_entropy_with_grad(x: pir.Tensor,
                            targets: pir.Tensor,
                            loss_scaling: Union[float, pir.Tensor] = 1,
                            ignore_index: Optional[int] = None) -> Tuple[pir.Tensor, pir.Tensor]:
    """Calculate the cross entropy `l` from `x` and `targets`. Returns `l` and `dl/dx`.
        `loss_scaling` can be used to scale the returned `dl/dx`. `l` will not be scaled by `loss_scaling`.
        The returned loss will be mean reduced across items in `targets`.
        Any item in `target` equal to `ignore_index` will not contribute to `l` or `dl/dx`.


    Args:
        x (pir.Tensor): Unnormalized Tensor with shape `(*dims, C)`.
        targets (pir.Tensor): Tensor with shape `(*dims)` with values in [0,C) or `ignore_index`.
        loss_scaling (Union[float, pir.Tensor], optional): Amount to scale `dl/dx`. This is useful to avoid underflow when using float16. Defaults to 1.
        ignore_index (Optional[int], optional): Values in `targets` to ignore. For example, padding. Defaults to None.

    Returns:
        Tuple[pir.Tensor, pir.Tensor]: (l, dl/dx)
    """
    probs = ops.softmax(x, axis=-1)

    if not isinstance(loss_scaling, pir.Tensor):
        loss_scaling = pir.constant(loss_scaling, pir.float32)

    loss, dx = ops.nll_loss_with_softmax_grad(probs, targets, loss_scaling, ignore_index=ignore_index)
    return loss, dx