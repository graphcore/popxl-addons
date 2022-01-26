# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np

import popart.ir as pir

import popart_ir_extensions as pir_ext
from popart_ir_extensions.input_factory import NamedInputFactories
from popart_ir_extensions.module import Module
from popart_ir_extensions.transforms.autodiff import autodiff, autodiff_with_accumulation


class Scale(Module):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        scale = self.add_input_tensor("scale", partial(np.full, x.shape, 2), x.dtype)
        return x * scale


def model(with_accumulation):
    np.random.seed(42)
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x = pir.variable(np.random.normal(0, 1, (2, 2)), pir.float32)

        args, graph = Scale().create_graph(x)
        if with_accumulation:
            dargs, dgraph = autodiff_with_accumulation(graph, [graph.args.scale], graph.graph.get_input_tensors())
        else:
            dgraph = autodiff(graph, grads_required=graph.graph.get_input_tensors())
            # Create empty. No named inputs.
            dargs = NamedInputFactories()

        # Construct variables for the graph.
        scale1 = args.init()
        scale2 = args.init()
        accum1 = dargs.init()
        accum2 = dargs.init()

        # Bind args
        fwd1 = graph.bind(scale1)
        fwd2 = graph.bind(scale2)

        # Call forward
        call_info_1 = fwd1.call_with_info(x)
        call_info_2 = fwd2.call_with_info(*call_info_1.get_output_tensors())

        # Call backward
        seed = pir.constant(np.ones((2, 2)), pir.float32)
        dscale2_out = dgraph.bind(accum1).call(
            seed, args=dgraph.grad_graph_info.get_inputs_from_forward_call_info(call_info_2))
        dscale1_out = dgraph.bind(accum2).call(
            dscale2_out[0], args=dgraph.grad_graph_info.get_inputs_from_forward_call_info(call_info_1))

        if with_accumulation:
            outs = []
        else:
            outs = [pir_ext.host_store(t) for t in (dscale1_out[1], dscale2_out[1])]

    runner = pir_ext.Runner(ir, outs)
    outputs = runner.run()

    if with_accumulation:
        accums = [accum1.scale, accum2.scale]
        results = runner.read_weights(accums)
        return tuple(results[t] for t in accums)

    runner.detach()

    return outputs


def test_gradient_accumulation_correctness():
    normal = model(with_accumulation=False)
    accumulation = model(with_accumulation=True)
    np.testing.assert_almost_equal(normal, accumulation)
