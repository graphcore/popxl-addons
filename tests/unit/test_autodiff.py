# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
import popart._internal.ir as _ir
import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons.testing_utils import ops_of_type


class Scale(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        scale = self.add_variable_input("scale", partial(np.ones, x.shape), x.dtype)
        return x * scale


def test_autodiff_patterns_executed():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x_h2d = popxl.h2d_stream((2, 2), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        args, graph = Scale().create_graph(x)

        dgraph = addons.autodiff(graph)

    grad_ops = dgraph.graph._pb_graph.getOps()

    mul_inplace_ops = ops_of_type(grad_ops, _ir.op.MulRhsInplaceOp) + ops_of_type(grad_ops, _ir.op.MulLhsInplaceOp)
    assert mul_inplace_ops == 2
