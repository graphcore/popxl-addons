# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl
import popxl_addons as addons

from popxl import ops
from popxl_addons.ops.group_quantize_decompress import (
    group_quantize_decompress,
    group_quantize_decompress_numpy,
    group_quantize_compress_numpy,
)


def generate_tensor():
    num_rows = 4096
    num_groups = 64
    group_size = 64

    np.random.seed(11235813)

    t_group_ids = np.random.randn(num_rows, num_groups, group_size).argsort(-1) % 16
    t_group_ids = t_group_ids.astype(np.uint16)
    int16_scales = 2 ** (np.arange(4) * 4)[::-1].astype(np.uint16)
    t_packed = t_group_ids.reshape(num_rows, num_groups, group_size // 4, 4) @ int16_scales
    t_packed = t_packed.astype(np.uint16)

    group_lims = np.random.randn(num_rows, num_groups, 2).astype(np.float16)

    group_maxs = (
        group_lims.max(-1, keepdims=True) + 0.1
    )  # make sure there is some non-negligible difference between min and max
    group_mins = group_lims.min(-1, keepdims=True)

    group_scale = (group_maxs - group_mins) / (2**4 - 1)
    group_scale = group_scale.astype(np.float16)
    group_bias = group_mins.astype(np.float16)

    t_grouped = t_group_ids.astype(np.float16) * group_scale + group_bias

    t = t_grouped.reshape(num_rows, num_groups * group_size).astype(np.float16)
    return (t_packed, group_scale, group_bias), t


def test_numpy():
    (t_packed, group_scale, group_bias), t = generate_tensor()
    t_rec = group_quantize_decompress_numpy(t_packed, group_scale, group_bias)
    compressed = group_quantize_compress_numpy(t, group_size=64)
    (t_packed_rec, group_scale_rec, group_bias_rec) = compressed

    np.testing.assert_allclose(t, t_rec, atol=1e-6, rtol=0)
    np.testing.assert_allclose(t_packed, t_packed_rec, atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        group_scale, group_scale_rec, atol=1e-3, rtol=0
    )  # tolerance higher due to smaller values -> underflow
    np.testing.assert_allclose(
        group_bias, group_bias_rec, atol=1e-3, rtol=0
    )  # tolerance higher due to smaller values -> underflow


def test_compress_is_finite():
    _, t = generate_tensor()
    t[0][0] = np.nan
    try:
        group_quantize_compress_numpy(t, group_size=64)
        assert False, "Did not catch non-finite value in t"
    except ValueError:
        pass


def test_group_quantize_decompress():
    (t_packed, group_scale, group_bias), t = generate_tensor()

    ir = popxl.Ir()

    with ir.main_graph:
        t_packed_xl = popxl.variable(t_packed, dtype=popxl.uint16)
        group_scale_xl = popxl.constant(group_scale, dtype=popxl.float16)
        group_bias_xl = popxl.constant(group_bias, dtype=popxl.float16)

        decompress_graph = ir.create_graph(group_quantize_decompress, t_packed_xl, group_scale_xl, group_bias_xl)
        (t_decompressed,) = ops.call(decompress_graph, t_packed_xl, group_scale_xl, group_bias_xl)

        t_decompressed_d2h = addons.host_store(t_decompressed)

    with popxl.Session(ir, "ipu_hw") as sess:
        out = sess.run()

    t_out = out[t_decompressed_d2h]

    np.testing.assert_allclose(t, t_out, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test_compress_is_finite()
