# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl
from popxl import ops
import popxl_addons as addons


def test_seed_variable():
    ir = popxl.Ir(4)
    tp_group = ir.replica_grouping(group_size=2)
    with ir.main_graph:
        seed_v, seed = addons.seed_variable(1984, tp_group)
        d2h = popxl.d2h_stream(seed.shape, seed.dtype)
        ops.host_store(d2h, seed)

    with popxl.Session(ir, device_desc="ipu_hw") as sess:
        output = sess.run()[d2h]

    output = output.reshape(2, 2, -1)
    for a, b in output:
        np.testing.assert_equal(a, b)

    np.testing.assert_equal(sess.get_tensor_data(seed_v), output[:, 0, :])
