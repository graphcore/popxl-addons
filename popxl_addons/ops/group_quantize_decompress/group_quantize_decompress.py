# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook

from . import group_quantize_decompress_binding

import popxl
from popxl.context import op_debug_context, get_current_context
from popxl.ops.utils import check_in_graph, check_tensor_ipu_and_tile_set

from typing import Tuple
import numpy as np

__all__ = [
    "group_quantize_decompress",
    "group_quantize_decompress_numpy",
    "group_quantize_compress_numpy",
]


@op_debug_context
def group_quantize_decompress(
    t: popxl.Tensor, group_scale: popxl.Tensor, group_bias: popxl.Tensor, dim=-1
) -> popxl.Tensor:
    """
    Decompression of 4-bit group-quantized floating point tensors as described in
    "FlexGen: High-throughput Generative Inference of Large Language Models with a
    Single GPU" https://arxiv.org/abs/2303.06865

    Assumes that input tensors have been compressed by grouping along a particular axis,
    scaling according to the min and max value within the group, then rounding to the
    nearest 4-bit value [0, 16) (or 0-f if you prefer). These 4-bit values have then been
    packed into a standard datatype (uint16 in this case).

    To decompress, the packed tensor must be unpacked and rescaled according to the min
    and max value. For efficiency, the min and max values are passed here as scale and
    bias terms.

    Args:
        t (popxl.Tensor): 4-bit compressed, packed Tensor
                          (shape=(num_rows, num_groups, num_group_ids), dtype=uint16)
        group_scale (popxl.Tensor): scaling factor of unpacked uint16 inputs
                          (shape=(num_rows, num_groups, 1), dtype=float16)
        group_bias (popxl.Tensor): bias term for unpacked uint16 inputs
                          (shape=(num_rows, num_groups, 1), dtype=float16)
        dim (int): dimension to decompress tensor into.

    Returns:
        popxl.Tensor: Decompressed result (float16)
                      (shape=(num_rows, num_groups * num_group_ids * 4), dtype=float16)
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    if t.rank != 3:
        raise NotImplementedError("4-bit decompression currently supported only for packed tensors with rank 3")

    if group_scale.rank < t.rank - 1 or group_scale.rank > t.rank:
        raise ValueError("group_scale.rank must be equal to or one less than t.rank")
    elif group_scale.rank == t.rank - 1:
        group_scale = group_scale.reshape((*group_scale.shape, 1))

    if group_bias.rank < t.rank - 1 or group_bias.rank > t.rank:
        raise ValueError("group_bias.rank must be equal to or one less than t.rank")
    elif group_bias.rank == t.rank - 1:
        group_bias = group_bias.reshape((*group_bias.shape, 1))

    if t.shape[:-1] != group_scale.shape[: t.rank - 1]:
        raise ValueError("The first two dimensions of t and group_scale must be the same")
    if t.shape[:-1] != group_bias.shape[: t.rank - 1]:
        raise ValueError("The first two dimensions of t and group_bias must be the same")

    check_in_graph(g, t=t, group_scale=group_scale, group_bias=group_bias)
    check_tensor_ipu_and_tile_set(t=t, group_scale=group_scale, group_bias=group_bias)

    settings = ctx._get_op_settings("GroupQuantizeDecompress")
    op = group_quantize_decompress_binding.GroupQuantizeDecompressOp.createOpInGraph(
        pb_g,
        {0: t.id, 1: group_scale.id, 2: group_bias.id},
        {0: g._create_tensor_id("t_decompressed")},
        settings,
    )
    ctx._op_created(op)
    t_out = popxl.Tensor._from_pb_tensor(op.outTensor(0))
    if dim != -1:
        perm = list(range(t_out.rank))
        i = perm.pop(-1)
        perm.insert(dim, i)
        t_out = t_out.transpose(perm)
    return t_out


def group_quantize_decompress_numpy(
    t: np.ndarray, group_scale: np.ndarray, group_bias: np.ndarray, dim: int
) -> np.ndarray:
    """
    Numpy implementation of 4-bit group-quantized floating point tensors decompression
    as described in "FlexGen: High-throughput Generative Inference of Large Language Models
    with a Single GPU" https://arxiv.org/abs/2303.06865

    Assumes that input tensors have been compressed by grouping along a particular axis,
    scaling according to the min and max value within the group, then rounding to the
    nearest 4-bit value [0, 16) (or 0-f if you prefer). These 4-bit values have then been
    packed into a standard datatype (uint16 in this case).

    To decompress, the packed tensor must be unpacked and rescaled according to the min
    and max value. For efficiency, the min and max values are passed here as scale and
    bias terms.

    Args:
        t (np.ndarray): 4-bit compressed, packed Tensor
                          (shape=(*leading_dims, *trailing_dims, num_groups, num_group_ids), dtype=uint16)
        group_scale (np.ndarray): scaling factor of unpacked uint16 inputs
                          (shape=(*leading_dims, *trailing_dims, num_groups, 1), dtype=float16)
        group_bias (np.ndarray): bias term for unpacked uint16 inputs
                          (shape=(*leading_dims, *trailing_dims, num_groups, 1), dtype=float16)
        dim (int): Compressed dimension is stored as the trailing dimension. If dim !=-1,
                          tensor is transposed to place decompressed dimension at specified location

    Returns:
        np.ndarray: Decompressed result (float16)
                      (shape=(*leading_dims, num_groups * num_group_ids * 4, *trailing_dims), dtype=float16)

    """
    leading_dims = t.shape[:-2]
    n_groups, n_group_ids = t.shape[-2:]
    t_unpacked = (
        np.stack([(t >> s) & 15 for s in [12, 8, 4, 0]])  # mask and shift
        .transpose(*list(range(1, t.ndim + 1)), 0)
        .reshape(*leading_dims, n_groups, n_group_ids * 4)  # reshape
    ).astype(np.float16)
    t_scaled = t_unpacked * group_scale + group_bias  # rescale
    t_scaled = t_scaled.reshape(*leading_dims, n_groups * n_group_ids * 4)  # reshape
    if dim != -1:
        perm = list(range(t_scaled.ndim))
        i = perm.pop(-1)
        perm.insert(dim, i)
        t_scaled = t_scaled.transpose(perm)
    return t_scaled


def group_quantize_compress_numpy(
    t: np.ndarray, group_size: int, dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy implementation of 4-bit group-quantized floating point tensors compression
    as described in "FlexGen: High-throughput Generative Inference of Large Language Models
    with a Single GPU" https://arxiv.org/abs/2303.06865

    Input tensors are compressed by grouping along a particular axis, scaling according to the
    min and max value within the group, then rounding to the nearest 4-bit value [0, 16) (or
    0-f if you prefer). These 4-bit values are been packed into a standard datatype
    (uint16 in this case).

    Args:
        t (np.ndarray): Tensor to compress (float16)
                      (shape=(*leading_dims, num_groups * group_size * 4, *trailing_dims), dtype=float16)
        group_size (int): number of elements to compute floating point stats
        dim (int): dimension of input tensor selected for group quantization

    Returns:
        (np.ndarray): 4-bit compressed, packed Tensor
                      (shape=(*leading_dims, *trailing_dims, num_groups, group_size), dtype=uint16)
        (np.ndarray): scaling factor of unpacked uint16 inputs
                      (shape=(*leading_dims, *trailing_dims, num_groups, 1), dtype=float16)
        (np.ndarray): bias term for unpacked uint16 inputs
                      (shape=(*leading_dims, *trailing_dims, num_groups, 1), dtype=float16)

    """
    if dim != -1:
        perm = list(range(t.ndim))
        i = perm.pop(dim)
        perm.append(i)
        t = t.transpose(perm)

    leading_dims = t.shape[:-1]
    n_cols = t.shape[-1]
    n_groups = n_cols // group_size

    if not np.isfinite(t).all():
        raise ValueError("Cannot compress t which contains non-finite values")

    t_grouped = t.reshape(*leading_dims, n_groups, group_size)
    t_max = t_grouped.max(-1, keepdims=True).astype(np.float16)
    t_min = t_grouped.min(-1, keepdims=True).astype(np.float16)

    t_scale = (t_max - t_min) / (2**4 - 1)
    t_bias = t_min

    t_quantized = np.round((t_grouped - t_bias) / t_scale).astype(np.uint16)
    int16_scales = 2 ** (np.arange(4) * 4)[::-1].astype(np.uint16)

    t_packed = t_quantized.reshape(*leading_dims, n_groups, group_size // 4, 4) @ int16_scales
    return (t_packed.astype(np.uint16), t_scale, t_bias)
