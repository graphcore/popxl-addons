# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Tuple
import numpy as np

import popxl
import popxl_addons as addons


class DoubleLinear(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        # Add is an example of a Op that does not require the input tensor to calculate the gradient.
        # This means that when DoubleLinear is recomputed x should be a 'new' expected input as  with recomputation
        # y would have been used instead.
        y = x + 2

        w1 = self.add_variable_input("w1", partial(np.random.normal, 0, 0.1, (2, 2)), x.dtype)
        w2 = self.add_variable_input("w2", partial(np.random.normal, 0, 0.1, (2, 2)), x.dtype)
        return (y @ w1) @ w2


def get_model_outputs(recompute: bool) -> Tuple[popxl.Tensor, ...]:
    np.random.seed(1984)
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x_data, x_h2d, x = addons.host_load(np.random.normal(0, 0.1, (2, 2)).astype(np.float32), popxl.float32, "x")
        args, graph = DoubleLinear().create_graph(x)
        dgraph = addons.autodiff(graph)

        if recompute:
            dgraph = addons.recompute_graph(dgraph)

        scale = graph.bind(args.init())

        x, *_ = scale.call(x)
        call_info = scale.call_with_info(x)
        x, *_ = call_info.outputs

        gradient = popxl.constant(np.ones(x.shape), x.dtype, "gradient")
        outputs_t: Tuple[popxl.Tensor, ...] = dgraph.call(gradient, args=dgraph.grad_graph_info.inputs_dict(call_info))

        out_streams = tuple(map(addons.host_store, outputs_t))

    ir.num_device_transfers = 1
    session = popxl.Session(ir, "ipu_hw")
    outputs = session.run({x_h2d: x_data})
    session.device.detach()
    return tuple(outputs[o_d2h] for o_d2h in out_streams)  # type: ignore


def test_recompute_correctness():
    outputs = get_model_outputs(False)
    recomp_outputs = get_model_outputs(True)
    for out, recomp in zip(outputs, recomp_outputs):
        np.testing.assert_almost_equal(out, recomp)


# TODO: Test with accumulation
