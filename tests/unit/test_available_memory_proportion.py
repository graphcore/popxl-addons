# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl_addons as addons


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
