# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

import popxl
from popxl_addons.performance_utils import evaluate_FLOPs


def test_matmul_flops():
    ir = popxl.Ir()
    ir.replication_factor = 1
    with ir.main_graph, popxl.in_sequence(True):
        # matrix vector
        x = popxl.constant(np.full((4), 1), dtype=popxl.float32)
        a = popxl.constant(np.full((4, 8), 1), dtype=popxl.float32)
        y = x @ a

        # dot product vector vector
        x = popxl.constant(np.full((4), 1), dtype=popxl.float32)
        a = popxl.constant(np.full((4), 1), dtype=popxl.float32)
        s = x.T @ a

        # matrix matrix
        a = popxl.constant(np.full((10, 9), 1), dtype=popxl.float32)
        b = popxl.constant(np.full((9, 5), 1), dtype=popxl.float32)
        c = a @ b

        # tensor tensor generic
        a = popxl.constant(np.full((3, 5, 10, 9), 1), dtype=popxl.float32)
        b = popxl.constant(np.full((3, 5, 9, 3), 1), dtype=popxl.float32)
        c = popxl.ops.matmul(a, b)

    breakd = evaluate_FLOPs(ir.main_graph)
    matmuls = breakd["MatMul"]
    assert matmuls[100] == 2 * 4 * 8
    assert matmuls[102] == 2 * 4
    assert matmuls[103] == 2 * 9 * 10 * 5
    assert matmuls[104] == 2 * 9 * 3 * 10 * 3 * 5
