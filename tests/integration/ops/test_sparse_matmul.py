# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
from scipy.sparse import csr_matrix
from popxl_addons.ops.static_sparse_ops import dense_const_sparse_matmul, random_sparse


def test_dense_sparse_matmul_fp16():
    lhs = np.random.normal(0, 1, (2, 3)).astype(np.float16)
    rhs = np.random.normal(0, 1, (3, 4)).astype(np.float16)
    random_sparse(rhs)
    csr = csr_matrix(rhs)
    output_cols = rhs.shape[-1]

    # Creating a model with popxl
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        lhs_input = popxl.h2d_stream(lhs.shape, popxl.float16, name="lhs_stream")
        x = ops.host_load(lhs_input, "dense")

        o = dense_const_sparse_matmul(x, csr, output_cols)

        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
        ops.host_store(o_d2h, o)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run({lhs_input: lhs})

    np.testing.assert_almost_equal(lhs @ rhs, outputs[o_d2h], 2)


def test_dense_sparse_matmul_fp32():
    lhs = np.random.normal(0, 1, (1, 2, 3)).astype(np.float32)
    rhs = np.random.normal(0, 1, (3, 4)).astype(np.float32)
    random_sparse(rhs)
    csr = csr_matrix(rhs)
    output_cols = rhs.shape[-1]

    # Creating a model with popxl
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        lhs_input = popxl.h2d_stream(lhs.shape, popxl.float32, name="lhs_stream")
        x = ops.host_load(lhs_input, "dense")

        o = dense_const_sparse_matmul(x, csr, output_cols)

        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
        ops.host_store(o_d2h, o)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run({lhs_input: lhs})

    np.testing.assert_almost_equal(lhs @ rhs, outputs[o_d2h], 2)


if __name__ == "__main__":
    test_dense_sparse_matmul_fp32()
