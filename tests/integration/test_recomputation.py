# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Tuple
import numpy as np

import popart.ir as pir
import popart_ir_extensions as pir_ext


class DoubleLinear(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        # Add is an example of a Op that does not require the input tensor to calculate the gradient.
        # This means that when DoubleLinear is recomputed x should be a 'new' expected input as  with recomputation
        # y would have been used instead.
        y = x + 2

        w1 = self.add_input_tensor("w1", lambda: np.random.normal(0, 0.1, (2, 2)).astype(x.dtype.as_numpy()))
        w2 = self.add_input_tensor("w2", lambda: np.random.normal(0, 0.1, (2, 2)).astype(x.dtype.as_numpy()))
        return (y @ w1) @ w2


def get_model_outputs(recompute: bool) -> Tuple[pir.Tensor, ...]:
    np.random.seed(1984)
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x_data, x_h2d, x = pir_ext.host_load(np.random.normal(0, 0.1, (2, 2)).astype(np.float32), pir.float32, "x")
        scale_graph = DoubleLinear().to_concrete(x)

        d_scale_graph = pir_ext.autodiff(scale_graph)

        if recompute:
            d_scale_graph = pir_ext.recompute_graph(scale_graph, d_scale_graph)

        scale = scale_graph.to_callable(True)

        fwd_info = scale.call_with_info(x)
        act, *_ = fwd_info.get_output_tensors()

        d_scale = d_scale_graph.to_callable(True)

        pir_ext.connect_activations(fwd_info, d_scale)

        gradient = pir.constant(np.ones(act.shape), act.dtype, "gradient")
        outputs_t: Tuple[pir.Tensor, ...] = d_scale.call(gradient)  # type: ignore

        outputs = tuple(map(pir_ext.host_store, outputs_t))

    return pir_ext.Runner(ir=ir, outputs=outputs).run({x_h2d: x_data})  # type: ignore


def test_recompute_correctness():
    outputs = get_model_outputs(False)
    recomp_outputs = get_model_outputs(True)
    for out, recomp in zip(outputs, recomp_outputs):
        np.testing.assert_almost_equal(out, recomp)
