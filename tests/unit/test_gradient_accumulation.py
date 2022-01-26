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


def test_accumulator_ops_added():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        args, graph = Scale().create_graph(x)

        dargs, dgraph = pir_ext.autodiff_with_accumulation(graph, graph.args.tensors)  # type: ignore

        # Construct variables for the graph.
        scale = args.init()
        # Construct accumulators
        accum = dargs.init()

        # Call forward
        fwd = graph.bind(scale)
        call_info = fwd.call_with_info(x)

        # Call backward. With seed tensor and connected activations
        grad = pir.constant(np.ones((2, 2)), pir.float32)
        dgraph.bind(accum).call(grad, args=dgraph.grad_graph_info.get_inputs_from_forward_call_info(call_info))

    # One weight, one accumulator and one counter
    assert len(main.get_variables()) == 3

    d_ops = dgraph.graph._pb_graph.getOps()
    # Accumulator, Counter
    assert ops_of_type(d_ops, _ir.op.AccumulateOp) == 2
