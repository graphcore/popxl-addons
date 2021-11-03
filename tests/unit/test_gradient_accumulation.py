# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

import popart_ir_extensions as pir_ext
from popart_ir_extensions.testing_utils import ops_of_type


class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        scale = self.add_input_tensor("scale", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return x * scale


def test_accumulator_ops_added():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        scale_graph = Scale().to_concrete(x)

        d_scale_graph = pir_ext.autodiff_with_accumulation(
            scale_graph,
            [scale_graph.scale])  # type: ignore

        # Construct variables for the graph.
        scale = scale_graph.to_callable(True)
        # Construct accumulators
        d_scale = d_scale_graph.to_callable(True)

        # Call forward
        fwd = scale.call_with_info(x)

        # Connect activations from forward call to gradient graph.
        pir_ext.connect_activations(fwd, d_scale_graph.grad_info, d_scale)

        # Call backward. With seed tensor and connected activations
        d_scale.call(pir.constant(np.ones((2, 2)), pir.float32))

    # One weight and one accumulator
    assert len(main.get_variables()) == 2

    d_ops = d_scale_graph._pb_graph.getOps()
    assert ops_of_type(d_ops, _ir.op.AccumulateOp) == 1
