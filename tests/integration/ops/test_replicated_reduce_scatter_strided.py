# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import math
import popart._internal.ir as _ir

from popxl_addons.ops.replicated_strided_collectives import *


def test_reduce_scatter_strided_op():
    n_ipus = 16
    stride = 4
    group_size = 4

    inputs = np.arange(n_ipus * 2 * 3, dtype="float32").reshape((n_ipus, 2 * 3))
    data_size = 2 * 3
    replica_size = math.ceil(data_size / group_size)
    target = np.zeros(n_ipus * replica_size, dtype="float32").reshape((n_ipus, replica_size))
    for i in range(stride):
        psum = inputs[i::stride].sum(axis=0)
        tmp = psum.resize((group_size, replica_size), refcheck=False)
        for j in range(group_size):
            target[i + j * stride] = psum[j]

    ir = popxl.Ir()
    ir.replication_factor = n_ipus
    main = ir.main_graph
    with main:

        x_h2d = popxl.h2d_stream((2 * 3, ), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name="x")

        rg = ir.replica_grouping(stride, group_size)
        y = replicated_reduce_scatter_strided(x, rg)

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
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target[i], y_host_i)
