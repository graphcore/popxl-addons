# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

import popart_ir_extensions as pir_ext
from popart_ir_extensions.testing_utils import ops_of_type


class Scale(pir_ext.Module):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        scale = self.add_input_tensor("scale", partial(np.ones, x.shape), x.dtype)
        return x * scale


def test_autodiff_patterns_executed():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        args, graph = Scale().create_graph(x)

        dgraph = pir_ext.autodiff(graph)

    grad_ops = dgraph.graph._pb_graph.getOps()

    mul_inplace_ops = ops_of_type(grad_ops, _ir.op.MulRhsInplaceOp) + ops_of_type(grad_ops, _ir.op.MulLhsInplaceOp)
    assert mul_inplace_ops == 2
