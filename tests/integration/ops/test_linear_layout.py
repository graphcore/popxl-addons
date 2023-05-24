# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np

import popxl
import popxl.ops as ops

from popxl_addons.ops.linear_layout import linear_layout


def test_linear_layout():
    sequence = 10000
    inOut = np.random.normal(0, 1, (1, sequence)).astype(np.float16)
    value = 10000

    # Creating a model with popxl
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        inOut_input = popxl.h2d_stream(inOut.shape, popxl.float16, name="input_stream")
        x = ops.host_load(inOut_input, "input")
        x = x + 1
        o = linear_layout(x, 1)

        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
        ops.host_store(o_d2h, o)

    with popxl.Session(ir, "ipu_model") as session:
        outputs = session.run({inOut_input: inOut})


if __name__ == "__main__":
    test_linear_layout()
