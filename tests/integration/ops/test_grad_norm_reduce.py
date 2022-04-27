# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np

import popart._internal.ir as _ir

from popxl_addons.ops.grad_reduce_square_add import grad_reduce_square_add


def test_grad_norm_reduce_op():

    inputs = np.arange(2 * 3, dtype='float32').reshape((2, 3))

    ir = popxl.Ir()
    main = ir.main_graph
    with main:

        x_h2d = popxl.h2d_stream((2, 3), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name='x')

        y = grad_reduce_square_add(x, loss_scaling=64)

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    ir._pb_ir.setPatterns(_ir.patterns.Patterns(_ir.patterns.PatternsLevel.Default))
    for g in ir._pb_ir.getAllGraphs():
        ir._pb_ir.applyPreAliasPatterns(g)

    session = popxl.Session(ir, device_desc='ipu_hw')
    y_host = session.run({x_h2d: inputs})[y_d2h]
    session.device.detach()

    np.testing.assert_equal(np.sum(np.square(inputs / 64)), y_host)
