# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import math
import popart._internal.ir as _ir

from popxl_addons.ops.replicated_strided_collectives import *


def test_reduce_scatter_strided_op():
    n_ipus = 8
    stride = 2
    group_size = 2

    data_size = 2 * 3
    inner_size = stride * group_size
    inputs = np.arange(n_ipus * data_size, dtype="float32").reshape((n_ipus, data_size))
    replica_size = math.ceil(data_size / group_size)
    target = np.zeros(n_ipus * replica_size, dtype="float32").reshape((n_ipus, replica_size))
    for o in range(n_ipus // inner_size):
        inner_inputs = inputs[o * inner_size:(o + 1) * inner_size]
        for i in range(stride):
            psum = inner_inputs[i::stride].sum(axis=0)
            psum.resize((group_size, replica_size), refcheck=False)
            for j in range(group_size):
                target[i + j * stride + o * inner_size] = psum[j]

    ir = popxl.Ir()
    ir.replication_factor = n_ipus
    main = ir.main_graph
    with main:
        x_h2d = popxl.h2d_stream((data_size, ), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name="x")

        y = replicated_reduce_scatter_strided(x, group=ir.replica_grouping(stride, group_size))

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    with popxl.Session(ir, device_desc="ipu_hw") as session:
        y_host = session.run({x_h2d: inputs})
    y_host = y_host[y_d2h]

    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target[i], y_host_i)
