# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np

import popart._internal.ir as _ir

from popxl_addons.ops.replicated_strided_collectives import *


def test_all_gather_strided_op():
    n_ipus = 16
    stride = 4
    group_size = 4

    inputs = np.arange(n_ipus * 2 * 3, dtype="float32").reshape((n_ipus, 2, 3))
    target = np.zeros(n_ipus * group_size * 2 * 3, dtype="float32").reshape((n_ipus, group_size, 2, 3))
    for i in range(stride):
        psum = inputs[i::stride]
        for j in range(group_size):
            target[i + j * stride] = psum

    ir = popxl.Ir()
    ir.replication_factor = n_ipus
    main = ir.main_graph
    with main:

        x_h2d = popxl.h2d_stream((2, 3), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name="x")
        rg = ir.replica_grouping(stride, group_size)
        y = replicated_all_gather_strided(x, rg)
        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    ir._pb_ir.setPatterns(_ir.patterns.Patterns(_ir.patterns.PatternsLevel.Default))
    for g in ir._pb_ir.getAllGraphs():
        ir._pb_ir.applyPreAliasPatterns(g)

    session = popxl.Session(ir, device_desc="ipu_hw")
    y_host = session.run({x_h2d: inputs})
    y_host = y_host[y_d2h]
    session.device.detach()

    assert len(y_host) == n_ipus
    target = np.reshape(target, (n_ipus, group_size * 2 * 3))
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target[i], y_host_i)
