# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Union
import popxl
from popxl import ops
from popxl.ops.negative_log_likelihood import REDUCTION_TYPE

__all__ = ["cross_entropy", "cross_entropy_with_grad"]


def cross_entropy(x: popxl.Tensor,
                  labels: popxl.Tensor,
                  ignore_index: Optional[int] = None,
                  reduction: REDUCTION_TYPE = 'mean') -> popxl.Tensor:
    """
    Calculate the cross entropy `l` from `x` and `labels`. Returns the `l` loss.

    The returned loss will be reduced by `reduction` (default mean) across items in `labels`.
    Any item in `target` equal to `ignore_index` will not contribute to `l` or `dl/dx`.


    Args:
        x (popxl.Tensor): Unnormalized Tensor with shape `(*dims, C)`.
        labels (popxl.Tensor): Tensor with shape `(*dims)` with values in [0,C) or `ignore_index`.
        ignore_index (Optional[int], optional): Values in `labels` to ignore. For example, padding. Defaults to None.
        reduction (str): Type of reduction to apply to output loss. Options 'mean', 'sum' and 'none'

    Returns:
        popxl.Tensor: loss `l`
    """
    probs = ops.softmax(x, axis=-1)
    loss = ops.nll_loss(probs, labels, ignore_index=ignore_index, reduction=reduction)
    return loss


def cross_entropy_with_grad(x: popxl.Tensor,
                            labels: popxl.Tensor,
                            loss_scaling: Union[float, popxl.Tensor] = 1,
                            ignore_index: Optional[int] = None,
                            reduction: REDUCTION_TYPE = 'mean') -> Tuple[popxl.Tensor, popxl.Tensor]:
    """Calculate the cross entropy `l` from `x` and `labels`. Returns `l` and `dl/dx`.
        `loss_scaling` can be used to scale the returned `dl/dx`. `l` will not be scaled by `loss_scaling`.
        The returned loss will be mean reduced across items in `labels`.
        Any item in `target` equal to `ignore_index` will not contribute to `l` or `dl/dx`.


    Args:
        x (popxl.Tensor): Unnormalized Tensor with shape `(*dims, C)`.
        labels (popxl.Tensor): Tensor with shape `(*dims)` with values in [0,C) or `ignore_index`.
        loss_scaling (Union[float, popxl.Tensor], optional): Amount to scale `dl/dx`. This is useful to avoid underflow
            when using float16. Defaults to 1.
        ignore_index (Optional[int], optional): Values in `labels` to ignore. For example, padding. Defaults to None.
        reduction (str): Type of reduction to apply to output loss. Options 'mean', 'sum' and 'none'

    Returns:
        Tuple[popxl.Tensor, popxl.Tensor]: (l, dl/dx)
    """
    probs = ops.softmax(x, axis=-1)

    if not isinstance(loss_scaling, popxl.Tensor):
        loss_scaling = popxl.constant(loss_scaling, popxl.float32)

    loss, dx = ops.nll_loss_with_softmax_grad(probs,
                                              labels,
                                              loss_grad=loss_scaling,
                                              ignore_index=ignore_index,
                                              reduction=reduction)
    return loss, dx
