# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
import popart._internal.ir as _ir
import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons.testing_utils import ops_of_type
from popxl.tensor import Variable


class Scale(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        scale = self.add_input_tensor("scale", partial(np.ones, x.shape), x.dtype)
        return x * scale


def test_accumulator_ops_added():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x_h2d = popxl.h2d_stream((2, 2), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        args, graph = Scale().create_graph(x)

        dargs, dgraph = addons.autodiff_with_accumulation(graph, graph.args.tensors)  # type: ignore

        # Construct variables for the graph.
        scale = args.init()
        # Construct accumulators
        accum = dargs.init()

        # Call forward
        fwd = graph.bind(scale)
        call_info = fwd.call_with_info(x)

        # Call backward. With seed tensor and connected activations
        grad = popxl.constant(np.ones((2, 2)), popxl.float32)
        dgraph.bind(accum).call(grad, args=dgraph.grad_graph_info.inputs_dict(call_info))

    # One weight, one accumulator and one counter
    variables = [t for t in main.tensors if isinstance(t, Variable)]
    assert len(variables) == 3

    d_ops = dgraph.graph._pb_graph.getOps()
    # Accumulator, Counter
    assert ops_of_type(d_ops, _ir.op.AccumulateOp) == 2
