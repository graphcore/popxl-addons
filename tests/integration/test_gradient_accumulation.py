# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np

import popxl

import popxl_addons as addons
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons.module import Module
from popxl_addons.transforms.autodiff import autodiff, autodiff_with_accumulation


class Scale(Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        scale = self.add_variable_input("scale", partial(np.full, x.shape, 2), x.dtype)
        return x * scale


def model(with_accumulation):
    np.random.seed(42)
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 1, (2, 2)), popxl.float32)

        args, graph = Scale().create_graph(x)
        if with_accumulation:
            dargs, dgraph = autodiff_with_accumulation(graph, [graph.args.scale], graph.graph.inputs)
        else:
            dgraph = autodiff(graph, grads_required=graph.graph.inputs)
            # Create empty. No named inputs.
            dargs = NamedVariableFactories()

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
        call_info_2 = fwd2.call_with_info(*call_info_1.outputs)

        # Call backward
        seed = popxl.constant(np.ones((2, 2)), popxl.float32)
        dscale2_out = dgraph.bind(accum1).call(seed, args=dgraph.grad_graph_info.inputs_dict(call_info_2))
        dscale1_out = dgraph.bind(accum2).call(dscale2_out[0], args=dgraph.grad_graph_info.inputs_dict(call_info_1))

        if with_accumulation:
            out_streams = []
        else:
            out_streams = [addons.host_store(t) for t in (dscale1_out[1], dscale2_out[1])]

    ir.num_host_transfers = 1
    session = popxl.Session(ir, "ipu_hw")
    outputs = session.run()

    if with_accumulation:
        accums = [accum1.scale, accum2.scale]
        results = session.get_tensors_data(accums)
        return tuple(results[t] for t in accums)

    session.device.detach()

    return tuple(outputs[o_d2h] for o_d2h in out_streams)


def test_gradient_accumulation_correctness():
    normal = model(with_accumulation=False)
    accumulation = model(with_accumulation=True)
    np.testing.assert_almost_equal(normal, accumulation)
