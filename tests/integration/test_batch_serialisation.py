# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons.module import Module
from popxl_addons.transforms.batch_serialisation import batch_serialise, batch_serialise_forward_and_grad
from popxl_addons.transforms.autodiff import autodiff_with_accumulation


class Scale(Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        scale = self.add_input_tensor("scale", partial(np.full, x.shape, 2), x.dtype)
        return x * scale


def test_batch_serialisation_fwd_only():
    ir = popxl.Ir()
    main = ir.main_graph

    cb = 2
    bf = 4

    with main:
        # Create graphs
        input_spec = popxl.constant(np.ones((cb, 2)), popxl.float32)
        args, graph = Scale().create_graph(input_spec)

        # Transform graphs
        graph = batch_serialise(graph, bf, graph.graph.inputs[:1])

        # Create variables and bind
        scale = graph.bind(args.init())

        # Create Program
        x_h2d = popxl.h2d_stream((bf, cb, 2), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scaled, = scale.call(x)

        out = addons.host_store(scaled)

    inputs = np.arange(x.nelms).reshape(x.shape).astype(x.dtype.as_numpy())

    ir.num_host_transfers = 1
    outputs = popxl.Session(ir, "ipu_hw").run({x_h2d: inputs})[out]

    print(inputs * 2)
    print(outputs)
    np.testing.assert_equal(inputs * 2, outputs)


class Linear(Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
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
        ir = popxl.Ir()
        main = ir.main_graph
        combined_inputs = inputs.reshape((-1, 2))
        with main:
            # Create graphs
            args, graph = Linear(4).create_graph(popxl.constant(combined_inputs))
            dargs, dgraph = autodiff_with_accumulation(graph, [graph.args.weight], graph.graph.inputs)

            # Create variables
            linear = args.init()
            accums = dargs.init()

            # Bind variables
            fwd = graph.bind(linear)
            bwd = dgraph.bind(accums)
            # Create Program
            x_h2d = popxl.h2d_stream(combined_inputs.shape, popxl.float32, name="x_stream")
            x = ops.host_load(x_h2d, "x")

            call_info = fwd.call_with_info(x)

            # Note Mean over batch_serialisation_factor
            grad_seed = popxl.constant(grad.reshape(-1, 4) / bf)
            bwd.call(grad_seed, args=dgraph.grad_graph_info.inputs_dict(call_info))

        ir.num_host_transfers = 1
        session = popxl.Session(ir)
        session.run({x_h2d: combined_inputs})
        accum = session.get_tensor_data(accums.weight)
        session.device.detach()
        return accum.copy()

    def batch_serial():
        np.random.seed(42)
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # Create graphs
            args, graph = Linear(4).create_graph(popxl.constant(inputs[0]))
            dargs, dgraph = autodiff_with_accumulation(graph, [graph.args.weight], graph.graph.inputs)

            # Transform graphs
            graph, dgraph = batch_serialise_forward_and_grad(graph, dgraph, bf, graph.graph.inputs[:1])

            # Create variablesrecompute_graphre
            linear = args.init()
            accums = dargs.init()

            # Bind variables
            fwd = graph.bind(linear)
            bwd = dgraph.bind(accums)
            # Create Program
            x_h2d = popxl.h2d_stream(inputs.shape, popxl.float32, name="x_stream")
            x = ops.host_load(x_h2d, "x")

            call_info = fwd.call_with_info(x)

            # no mean over bf
            bwd.call(popxl.constant(grad), args=dgraph.grad_graph_info.inputs_dict(call_info))

        ir.num_host_transfers = 1
        session = popxl.Session(ir, "ipu_hw")
        session.run({x_h2d: inputs})
        accum = session.get_tensor_data(accums.weight)
        session.device.detach()
        return accum.copy()

    print("\nNormal")
    normal = graph()
    print(normal)
    print("Batch Serial")
    batch_ser = batch_serial()
    print(batch_ser)
    np.testing.assert_almost_equal(normal, batch_ser, 6)
