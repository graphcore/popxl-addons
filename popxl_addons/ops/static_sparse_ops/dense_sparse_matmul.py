# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook  # pylint: disable=unused-import

# The custom op and its pybinding will be automatically compiled by cppimport
# into a module of this name.
from . import dense_sparse_matmul_impl

import numpy as np
from scipy.stats import bernoulli
from scipy.sparse import csr_matrix
from collections import namedtuple

import popxl
from popxl.context import get_current_context, op_debug_context
from popxl.ops.utils import check_in_graph
from popxl.tensor import Tensor

__all__ = ["dense_const_sparse_matmul", "random_sparse"]


def random_sparse(dense: np.ndarray, density: float = 0.5):
    """Randomly sparsify a dense matrix for a given density.

    Args:
        dense (np.ndarray): a dense matrix.
        density (float): percentage of nz values in the matrix to keep.

    Returns:
        ndarrays: The output sparsified dense matrix.

    """
    mask = bernoulli.rvs(1 - density, size=dense.shape).astype(np.bool_)
    # Ensure at least one value is kept
    if np.all(mask):
        mask.reshape(-1)[0] = False
    dense[mask] = 0
    return dense


@op_debug_context
def dense_const_sparse_matmul(lhs: popxl.Tensor, rhs: csr_matrix, output_cols: int) -> popxl.Tensor:
    """Sparse matmul in the form of dense x sparse = dense. This op is intended for inference with static point / element sparsity as the RHS sparsity patterns do not change.
    This is based on the static sparse matmul in `popsparse` that uses gather scatter to speed up the sparse matmul.
    In this implementation, the sparse matrix is represented by a constant CSR matrix that includes non-zero values `data`, column indices `indices` and row indices `indptr`.
    RHS has the same data type on device as the LHS dense matrix. Only float16 and float32 are supported.

    Note that this operator is for 2D matrix. It can accept [G][m][k] x [G][k][n] only when G is 1.

    Args:
        lhs (popxl.Tensor): The dense LHS.
        rhs (csr_matrix): The constant CSR representation of the sparse RHS.
        output_cols (int): The number of columns in the sparse RHS.

    Returns:
        popxl.Tensor: The output tensor.
    """
    ctx = get_current_context()
    graph = ctx.graph
    pb_graph = graph._pb_graph

    settings = ctx._get_op_settings("sparse_matmul")
    check_in_graph(graph, **{lhs.id: lhs})

    if lhs.dtype != popxl.float16 and lhs.dtype != popxl.float32:
        raise RuntimeError(f"dtype {lhs.dtype} not currently supported.")

    data = np.asarray(rhs.data, dtype=np.float32, order="C")
    indices = np.asarray(rhs.indices, dtype=np.uint32, order="C")
    indptr = np.asarray(rhs.indptr, dtype=np.uint32, order="C")

    params = dense_sparse_matmul_impl.DenseSparseMatMulParams(data, indices, indptr, output_cols)
    # Building the op using default operator id
    op = dense_sparse_matmul_impl.DenseSparseMatMulOp.create_op_in_graph(
        graph=pb_graph,
        inputs={0: lhs.id},
        outputs={0: graph._create_tensor_id("sparse_matmul_out")},
        params=params,
        settings=settings,
    )
    # Applying context all registered hooks to the new op.
    # NOTE: crucial to support PopXL graph transforms.
    ctx._op_created(op)
    return Tensor._from_pb_tensor(op.outTensor(0))
