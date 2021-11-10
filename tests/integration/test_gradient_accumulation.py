# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np

import popart.ir as pir
import popart.ir.ops as ops

import popart_ir_extensions as pir_ext


class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        scale = self.add_input_tensor("scale", lambda: np.random.normal(0, 1, x.shape).astype(x.dtype.as_numpy()))
        return x @ scale


def model(with_accumulation):
    np.random.seed(42)
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x = pir.variable(np.random.normal(0, 1, (2, 2)), pir.float32)

        scale_graph = Scale().to_concrete(x)
        if with_accumulation:
            d_scale_graph = pir_ext.autodiff_with_accumulation(scale_graph, [scale_graph.scale],
                                                               scale_graph.get_input_tensors())
        else:
            d_scale_graph = pir_ext.autodiff(scale_graph, grads_required=scale_graph.get_input_tensors())

        # Construct variables for the graph.
        scale1 = scale_graph.to_callable(True)
        scale2 = scale_graph.to_callable(True)
        # Construct accumulators
        d_scale1 = d_scale_graph.to_callable(True)
        d_scale2 = d_scale_graph.to_callable(True)

        # Call forward
        fwd1 = scale1.call_with_info(x)
        fwd2 = scale2.call_with_info(fwd1.get_op_output_tensor(0))

        # Connect activations from forward call to gradient graph.
        pir_ext.connect_activations(fwd1, d_scale1)
        pir_ext.connect_activations(fwd2, d_scale2)

        # Call backward. With seed tensor and connected activations
        bwd2 = d_scale2.call_with_info(pir.constant(np.ones((2, 2)), pir.float32))
        bwd1 = d_scale1.call_with_info(bwd2.get_op_output_tensor(0))

        if with_accumulation:
            outs = []
        else:
            outs = [pir_ext.host_store(t) for t in (bwd1.get_op_output_tensor(1), bwd2.get_op_output_tensor(1))]

    runner = pir_ext.Runner(ir, outs)
    outputs = runner.run()

    if with_accumulation:
        accums = [d_scale1.Accum__scale, d_scale2.Accum__scale]
        results = runner.read_weights(accums)
        return tuple(results[t] for t in accums)
    return outputs


def test_gradient_accumulation_correctness():
    normal = model(with_accumulation=False)
    accumulation = model(with_accumulation=True)
    np.testing.assert_almost_equal(normal, accumulation)
