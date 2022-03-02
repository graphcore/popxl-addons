# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np

import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons.module import Module
from popxl_addons.transforms.autodiff import autodiff_with_accumulation
from popxl_addons.transforms.pipelining import stash_and_restore_activations


def test_pipeline_2_stage():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        in_stream = popxl.h2d_stream((), popxl.uint32)
        out_stream = popxl.d2h_stream((), popxl.uint32)

        with addons.pipelined_execution(10):
            with popxl.pipeline_stage(0), popxl.ipu(0):
                x = ops.host_load(in_stream)
                x = x + 1
                x = x.copy_to_ipu(1)

            with popxl.pipeline_stage(1), popxl.ipu(1):
                x = x + 1
                ops.host_store(out_stream, x)

    result: np.ndarray = addons.Runner(ir, out_stream, device_iterations=10, device_num=2).run(
        {in_stream: np.arange(10).reshape(-1, 1).astype(np.uint32)})  # type: ignore
    np.testing.assert_equal(result.reshape(-1), np.arange(10) + 2)


def test_pipeline_4_stage():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        in_stream = popxl.h2d_stream((), popxl.uint32)
        out_stream = popxl.d2h_stream((), popxl.uint32)

        with addons.pipelined_execution(10):
            with popxl.pipeline_stage(0), popxl.ipu(0):
                x = ops.host_load(in_stream)
                x = x + 1
                x = x.copy_to_ipu(1)

            with popxl.pipeline_stage(1), popxl.ipu(1):
                x = x + 1
                x = x.copy_to_ipu(2)

            with popxl.pipeline_stage(2), popxl.ipu(2):
                x = x + 1
                x = x.copy_to_ipu(3)

            with popxl.pipeline_stage(3), popxl.ipu(3):
                x = x + 1
                ops.host_store(out_stream, x)

    result: np.ndarray = addons.Runner(ir, out_stream, device_iterations=10, device_num=4).run(
        {in_stream: np.arange(10).reshape(-1, 1).astype(np.uint32)})  # type: ignore
    np.testing.assert_equal(result.reshape(-1), np.arange(10) + 4)


class Linear(Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        x = x + 1
        w = self.add_input_tensor("weight", partial(np.random.normal, 0, 0.02, (x.shape[-1], self.out_features)),
                                  x.dtype)
        return x @ w


def test_pipeline_training():
    steps = 6

    def graph():
        np.random.seed(42)
        ir = popxl.Ir()
        main = ir.main_graph

        data = np.random.normal(0, 1, (steps, 1, 4)).astype(np.float32)

        with main:
            with popxl.ipu(0):
                _, x_stream, x = addons.host_load(data[0], popxl.float32, "x")

                args, linear_graph = Linear(4).create_graph(x)
                dargs, dlinear_graph = autodiff_with_accumulation(linear_graph, [linear_graph.args.weight])

                linear = args.init()
                dlinear = dargs.init()

                fwd = linear_graph.bind(linear)
                call_info = fwd.call_with_info(x)
                x, *_ = call_info.outputs
                x = x.copy_to_ipu(1)

            with popxl.ipu(1):
                x = x + 1
                x = x.copy_to_ipu(0)

            with popxl.ipu(0):
                dlinear_graph.bind(dlinear).call(x, args=dlinear_graph.grad_graph_info.inputs_dict(call_info))

        runner = addons.Runner(ir, [], device_iterations=1, device_num=2)
        for n in range(steps):
            runner.run({x_stream: data[n]})  # type: ignore

        weights = runner.read_weights([dlinear.weight])
        runner.detach()

        return weights[dlinear.weight]

    def pipelined_graph():
        np.random.seed(42)
        ir = popxl.Ir()
        main = ir.main_graph
        linear = Linear(4)

        data = np.random.normal(0, 1, (steps, 1, 4)).astype(np.float32)

        with main:
            with addons.pipelined_execution(steps):
                with popxl.pipeline_stage(0), popxl.ipu(0):
                    _, x_stream, x = addons.host_load(data[0], popxl.float32, "x")

                    # Compute Graphs
                    args, linear_graph = Linear(4).create_graph(x)
                    dargs, dlinear_graph = autodiff_with_accumulation(linear_graph, [linear_graph.args.weight])

                    # Variables
                    linear = args.init()
                    dlinear = dargs.init()

                    fwd = linear_graph.bind(linear)
                    call_info = fwd.call_with_info(x)
                    x, _ = call_info.outputs
                    x = x.copy_to_ipu(1)

                with popxl.pipeline_stage(1), popxl.ipu(1):
                    x = x + 1
                    x = x.copy_to_ipu(0)

                with popxl.pipeline_stage(2), popxl.ipu(0):
                    dlinear_graph.bind(dlinear).call(x,
                                                     args=stash_and_restore_activations(
                                                         call_info, dlinear_graph.grad_graph_info))

        runner = addons.Runner(ir, [], device_iterations=steps, device_num=2)
        runner.run({x_stream: data})  # type: ignore

        weights = runner.read_weights([dlinear.weight])
        runner.detach()

        return weights[dlinear.weight]

    normal = graph()

    pipelined = pipelined_graph()

    np.testing.assert_almost_equal(normal, pipelined)
