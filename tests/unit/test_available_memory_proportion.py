# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart.ir as pir
import popart_ir_extensions as pir_ext


def test_avail_mem_prop_by_ipu():
    ir = pir.Ir()
    main = ir.main_graph
    with main:
        with pir.ipu(0):
            a = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            b = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            c = a @ b
        c1 = c.copy_to_ipu(1)
        with pir.ipu(1):
            d = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            c1 @ d

    pir_ext.set_available_memory_proportion_by_ipu(ir, [0.1, 0.2])

    # TODO: Add asserts when available_memory_proportion getter is added to `popart.ir`
