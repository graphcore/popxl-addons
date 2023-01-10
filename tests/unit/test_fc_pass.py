# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
from popxl_addons.transforms.use_fc_pass import enable_fc_pass, disable_fc_pass


def test_enable_fc_pass():
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        a = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
        b = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
        c = a @ b

    enable_fc_pass(ir)

    for g in ir._pb_ir.getAllGraphs():
        for op in g.getOps():
            if hasattr(op, "useFullyConnectedPass"):
                assert op.useFullyConnectedPass() == True


def test_disable_fc_pass():
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        a = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
        b = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
        c = a @ b

    disable_fc_pass(ir)

    for g in ir._pb_ir.getAllGraphs():
        for op in g.getOps():
            if hasattr(op, "useFullyConnectedPass"):
                assert op.useFullyConnectedPass() == False
