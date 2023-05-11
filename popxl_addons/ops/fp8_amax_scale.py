# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Union, Literal
import popxl
from popxl import ops
import numpy as np
from popxl import float8_143, float8_152
from popxl_addons.ops.fp8_utils import ieee_fp32_no_denorms, float8_143_def, float8_152_def

__all__ = ["fp8_amax_scale"]


def fp8_amax_scale(
    src: popxl.Tensor,
    dtype: Union[Literal[float8_152], Literal[float8_143]] = float8_143,
) -> popxl.Tensor:
    """Find the best scale using the absolute max of input tensor for the given FP8 format.
    Assume that there is no input data outside the FP32 histogram range.

    Args:
        src:
            A PopXL tensor to convert to `dtype`. Its dtype must be float16 or float32.
        dtype:
            The PopXL dtype representing the target data type. This must be one
            of `popxl.float8_143` or `popxl.float8_152`.

    Returns:
        popxl.Tensor: The scaling bias.
    """
    if src.dtype == popxl.float32 or src.dtype == popxl.float16:
        src_ftype = ieee_fp32_no_denorms
    else:
        raise RuntimeError(f"dtype {src.dtype} not currently supported.")

    if dtype == float8_143:
        dest_ftype = float8_143_def
    if dtype == float8_152:
        dest_ftype = float8_152_def

    amax = ops.max(ops.abs(src))
    log2_max = ops.log2(dest_ftype.max_value / amax)
    log2_max = ops.cast(ops.floor(log2_max), popxl.int32)
    log2_max = ops.clip(log2_max, -31, 31)

    return log2_max
