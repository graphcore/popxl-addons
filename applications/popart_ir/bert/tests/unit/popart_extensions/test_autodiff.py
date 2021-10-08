# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np

import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
import popart_extensions as pir_ext

from tests.unit.utils import ops_of_type


class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = pir_ext.variable_def(np.ones(x.shape, x.dtype.as_numpy()), "scale")
        return x * self.scale


def test_autodiff_patterns_executed():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        scale_graph = Scale().to_concrete(x)

        grad_scale_graph = pir_ext.autodiff(scale_graph)

    grad_ops = grad_scale_graph.graph._pb_graph.getOps()

    mul_inplace_ops = ops_of_type(grad_ops, _ir.op.MulRhsInplaceOp) + ops_of_type(grad_ops, _ir.op.MulLhsInplaceOp)
    assert mul_inplace_ops == 2
