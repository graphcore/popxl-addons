# Auto compile cpp files
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import cppimport.import_hook  # pylint: disable=unused-import

from . import fp8_mse_scale_impl
from typing import Union, Literal
import numpy as np
import popxl
from popxl.context import get_current_context, op_debug_context
from popxl.ops.utils import check_in_graph
from popxl.tensor import Tensor

from collections import namedtuple, defaultdict
from popxl import float8_143, float8_152, float16, float32, float64
from popxl_addons.ops.fp8_utils import ieee_fp32_no_denorms, ieee_fp16_denorms, float8_143_def, float8_152_def
import popart._internal.ir as _ir
import math

__all__ = ["fp8_mse_scale"]


max_scaling_bias = 31
min_scaling_bias = -31


class CandidateConfig:
    def __init__(self, format, src_ftype, dest_ftype, src_type):
        self.format = format
        self.num_exp_bits = dest_ftype.num_exp_bits
        self.num_mantissa_bits = dest_ftype.num_mantissa_bits
        self.fix_exponent_bias = dest_ftype.fix_exponent_bias
        error_lsb = (
            ieee_fp32_no_denorms.num_mantissa_bits if src_type == np.float32 else ieee_fp16_denorms.num_mantissa_bits
        )
        error_lsb -= self.num_mantissa_bits
        self.quantisation_error = 2 ** (-2.0 * error_lsb)
        self.format_span = dest_ftype.max_exp - dest_ftype.min_exp + 1

        # the min index of the exp of source data that can be represented in dest type
        self.min_src_exp = dest_ftype.min_exp + min_scaling_bias - src_ftype.min_exp

        # the max index of the exp of source data that can be represented in dest type
        self.max_src_exp = min(
            dest_ftype.min_exp + max_scaling_bias - src_ftype.min_exp,
            src_ftype.max_exp - src_ftype.min_exp,
        )
        self.num_exp_src = self.max_src_exp - self.min_src_exp + 1


@op_debug_context
def fp8_mse_scale(
    src: popxl.Tensor,
    dtype: Union[Literal[float8_152], Literal[float8_143]] = float8_143,
) -> popxl.Tensor:
    """
    Args:
        src:
            A PopXL tensor to convert to `dtype`. Its dtype must be float16 or float32.
        dtype:
            The PopXL dtype representing the target data type. This must be one
            of `popxl.float8_143` or `popxl.float8_152`.

    Returns:
        popxl.Tensor: The scaling bias.
    """
    ctx = get_current_context()
    graph = ctx.graph
    pb_graph = graph._pb_graph

    settings = ctx._get_op_settings("fp8_scale")
    check_in_graph(graph, **{src.id: src})
    src = src.flatten()
    if src.dtype == popxl.float32 or src.dtype == popxl.float16:
        src_ftype = ieee_fp32_no_denorms
    else:
        raise RuntimeError(f"dtype {src.dtype} not currently supported.")

    if dtype == float8_143:
        dest_ftype = float8_143_def
    if dtype == float8_152:
        dest_ftype = float8_152_def

    candidate = CandidateConfig(dtype, src_ftype, dest_ftype, src.dtype)

    # decide the parameters for each cast type
    params = fp8_mse_scale_impl.Fp8ScalingBiasParams(
        min_fp32_exp_index=candidate.min_src_exp,
        quantisation_error=candidate.quantisation_error,
        format_span=candidate.format_span,
    )

    # Building the op using default operator id
    op = fp8_mse_scale_impl.Fp8FindBestScaleOp.create_op_in_graph(
        graph=pb_graph,
        inputs={0: src.id},
        outputs={0: graph._create_tensor_id("scaling_bias")},
        params=params,
        settings=settings,
    )
    # Applying context all registered hooks to the new op.
    # NOTE: crucial to support PopXL graph transforms.
    ctx._op_created(op)
    return Tensor._from_pb_tensor(op.outTensor(0))
