# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops

import popart_ir_extensions as pir_ext
from popart_ir_extensions.module import Module
from popart_ir_extensions.transforms.batch_serialisation import batch_serialise, batch_serialise_forward_and_grad
from popart_ir_extensions.transforms.autodiff import autodiff_with_accumulation


class Scale(Module):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        scale = self.add_input_tensor("scale", partial(np.full, x.shape, 2), x.dtype)
        return x * scale


def test_batch_serialisation_fwd_only():
    ir = pir.Ir()
    main = ir.main_graph()

    cb = 2
    bf = 4

    with main:
        # Create graphs
        input_spec = pir.constant(np.ones((cb, 2)), pir.float32)
        args, graph = Scale().create_graph(input_spec)

        # Transform graphs
        graph = batch_serialise(graph, bf, graph.graph.get_input_tensors()[:1])

        # Create variables and bind
        scale = graph.bind(args.init())

        # Create Program
        x_h2d = pir.h2d_stream((bf, cb, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scaled, = scale.call(x)

        out = pir_ext.host_store(scaled)

    inputs = np.arange(x.nelms).reshape(x.shape).astype(x.dtype.as_numpy())

    outputs = pir_ext.Runner(ir, out).run({x_h2d: inputs})

    print(inputs * 2)
    print(outputs)
    np.testing.assert_equal(inputs * 2, outputs)


class Linear(Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features

    def build(self, x: pir.Tensor) -> pir.Tensor:
        x = x + 1
        w = self.add_input_tensor("weight", partial(np.random.normal, 0, 1, (x.shape[-1], self.out_features)), x.dtype)
        return x @ w


def test_batch_serialisation_grad():
    np.random.seed(42)
    cb = 3
    bf = 4
    inputs = np.random.normal(0, 1, (bf, cb, 2)).astype(np.float32)
    grad = np.random.normal(0, 1, (bf, cb, 4)).astype(np.float32)

    def graph():
        np.random.seed(42)
        ir = pir.Ir()
        main = ir.main_graph()
        combined_inputs = inputs.reshape((-1, 2))
        with main:
            # Create graphs
            args, graph = Linear(4).create_graph(pir.constant(combined_inputs))
            dargs, dgraph = autodiff_with_accumulation(graph, [graph.args.weight], graph.graph.get_input_tensors())

            # Create variables
            linear = args.init()
            accums = dargs.init()

            # Bind variables
            fwd = graph.bind(linear)
            bwd = dgraph.bind(accums)

            # Create Program
            x_h2d = pir.h2d_stream(combined_inputs.shape, pir.float32, name="x_stream")
            x = ops.host_load(x_h2d, "x")

            call_info = fwd.call_with_info(x)

            # Note Mean over batch_serialisation_factor
            grad_seed = pir.constant(grad.reshape(-1, 4) / bf)
            bwd.call(grad_seed, args=dgraph.grad_graph_info.get_inputs_from_forward_call_info(call_info))

        runner = pir_ext.Runner(ir)
        runner.run({x_h2d: combined_inputs})
        accum = runner.read_weights([accums.weight])[accums.weight]
        runner.detach()
        return accum

    def batch_serial():
        np.random.seed(42)
        ir = pir.Ir()
        main = ir.main_graph()
        with main:

            # Create graphs
            args, graph = Linear(4).create_graph(pir.constant(inputs[0]))
            dargs, dgraph = autodiff_with_accumulation(graph, [graph.args.weight], graph.graph.get_input_tensors())

            # Transform graphs
            graph, dgraph = batch_serialise_forward_and_grad(graph, dgraph, bf, graph.graph.get_input_tensors()[:1])

            # Create variablesrecompute_graphre
            linear = args.init()
            accums = dargs.init()

            # Bind variables
            fwd = graph.bind(linear)
            bwd = dgraph.bind(accums)

            # Create Program
            x_h2d = pir.h2d_stream(inputs.shape, pir.float32, name="x_stream")
            x = ops.host_load(x_h2d, "x")

            call_info = fwd.call_with_info(x)

            # Note Mean over batch_serialisation_factor
            bwd.call(pir.constant(grad), args=dgraph.grad_graph_info.get_inputs_from_forward_call_info(call_info))

        runner = pir_ext.Runner(ir)
        runner.run({x_h2d: inputs})
        accum = runner.read_weights([accums.weight])[accums.weight]
        runner.detach()
        return accum

    print("\nNormal")
    normal = graph()
    print("Batch Serial")
    batch_ser = batch_serial()
    print(batch_ser)
    np.testing.assert_almost_equal(normal, batch_ser, 6)
