# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from timeit import repeat
import numpy as np
import popxl
from popxl import TensorSpec, ops
import popxl_addons as addons
from popxl_addons.transforms.repeat_graph import repeat_graph


def test_avail_mem_prop_by_ipu():
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        with popxl.ipu(0):
            a = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            b = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            c = a @ b
        c1 = c.copy_to_ipu(1)
        with popxl.ipu(1):
            d = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            c1 @ d

    addons.set_available_memory_proportion_by_ipu(ir, [0.1, 0.2])

    # TODO: Add asserts when available_memory_proportion getter is added to `popxl`


def test_graph_available_memory_proportion_by_ipu():
    ir = popxl.Ir()
    main = ir.main_graph

    def mul(a: popxl.Tensor, b: popxl.Tensor):
        return a @ b

    spec = TensorSpec((2, 2), dtype=popxl.float32)
    with main:
        with popxl.ipu(0):
            a = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            b = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            g = ir.create_graph(mul, a, b)
            c = b
            for i in range(3):
                (c,) = ops.call(g, a, c)
        c1 = c.copy_to_ipu(1)
        with popxl.ipu(1):
            d = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            c1 @ d

    addons.set_graph_available_memory_proportion_by_ipu(main, [0.1, 0.2])
    session = popxl.Session(ir, "ipu_hw")
    with session:
        session.run()

    # TODO: Add asserts when available_memory_proportion getter is added to `popxl`
