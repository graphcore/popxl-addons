# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np

import popart.ir as pir
import popart.ir.ops as ops

import popart_ir_extensions as pir_ext


def test_pipeline_2_stage():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        in_stream = pir.h2d_stream((), pir.uint32)
        out_stream = pir.d2h_stream((), pir.uint32)

        with pir_ext.pipelined_execution(10):
            with pir.pipeline_stage(0), pir.virtual_graph(0):
                x = ops.host_load(in_stream)
                x = x + 1
                x = x.copy_to_ipu(1)

            with pir.pipeline_stage(1), pir.virtual_graph(1):
                x = x + 1
                ops.host_store(out_stream, x)

    result: np.ndarray = pir_ext.Runner(ir, out_stream, device_iterations=10, device_type=2).run(
        {in_stream: np.arange(10).reshape(-1, 1).astype(np.uint32)})  # type: ignore
    np.testing.assert_equal(result.reshape(-1), np.arange(10) + 2)


def test_pipeline_4_stage():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        in_stream = pir.h2d_stream((), pir.uint32)
        out_stream = pir.d2h_stream((), pir.uint32)

        with pir_ext.pipelined_execution(10):
            with pir.pipeline_stage(0), pir.virtual_graph(0):
                x = ops.host_load(in_stream)
                x = x + 1
                x = x.copy_to_ipu(1)

            with pir.pipeline_stage(1), pir.virtual_graph(1):
                x = x + 1
                x = x.copy_to_ipu(2)

            with pir.pipeline_stage(2), pir.virtual_graph(2):
                x = x + 1
                x = x.copy_to_ipu(3)

            with pir.pipeline_stage(3), pir.virtual_graph(3):
                x = x + 1
                ops.host_store(out_stream, x)

    result: np.ndarray = pir_ext.Runner(ir, out_stream, device_iterations=10, device_type=4).run(
        {in_stream: np.arange(10).reshape(-1, 1).astype(np.uint32)})  # type: ignore
    np.testing.assert_equal(result.reshape(-1), np.arange(10) + 4)


class Linear(pir_ext.GenericGraph):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features

    def build(self, x: pir.Tensor) -> pir.Tensor:
        x = x + 1
        w = self.add_input_tensor(
            "weight", lambda: np.random.normal(0, 0.02, (x.shape[-1], self.out_features)).astype(x.dtype.as_numpy()))
        return x @ w


def test_pipeline_training():
    steps = 6

    def graph():
        np.random.seed(42)
        ir = pir.Ir()
        main = ir.main_graph()
        linear = Linear(4)

        data = np.random.normal(0, 1, (steps, 1, 4)).astype(np.float32)

        with main:
            with pir.virtual_graph(0):
                _, x_stream, x = pir_ext.host_load(data[0], pir.float32, "x")

                linear_graph = linear.to_concrete(x)
                dlinear_graph = pir_ext.autodiff_with_accumulation(linear_graph, [linear.weight])

                fwd = linear_graph.to_callable(True).call_with_info(x)
                x = fwd.get_op_output_tensor(0).copy_to_ipu(1)

            with pir.virtual_graph(1):
                x = x + 1
                x = x.copy_to_ipu(0)

            with pir.virtual_graph(0):
                dlinear = dlinear_graph.to_callable(True)
                pir_ext.connect_activations(fwd, dlinear)
                dlinear.call(x)

        runner = pir_ext.Runner(ir, [], device_iterations=1, device_type=2)
        for n in range(steps):
            runner.run({x_stream: data[n]})  # type: ignore

        accum_tensor = dlinear.get_grad_accumulator_for_fwd_input(linear_graph.weight)
        weights = runner.read_weights([accum_tensor])
        runner.detach()

        return weights[accum_tensor]

    def pipelined_graph():
        np.random.seed(42)
        ir = pir.Ir()
        main = ir.main_graph()
        linear = Linear(4)

        data = np.random.normal(0, 1, (steps, 1, 4)).astype(np.float32)

        with main:
            stashes = pir_ext.PipelineStashHelper()
            with pir_ext.pipelined_execution(steps):
                with pir.pipeline_stage(0), pir.virtual_graph(0):
                    _, x_stream, x = pir_ext.host_load(data[0], pir.float32, "x")

                    linear_graph = linear.to_concrete(x)
                    dlinear_graph = pir_ext.autodiff_with_accumulation_and_recomputation(linear_graph, [linear.weight])

                    fwd = linear_graph.to_callable(True).call_with_info(x)
                    x = fwd.get_op_output_tensor(0).copy_to_ipu(1)

                with pir.pipeline_stage(1), pir.virtual_graph(1):
                    x = x + 1
                    x = x.copy_to_ipu(0)

                with pir.pipeline_stage(2), pir.virtual_graph(0):
                    dlinear = dlinear_graph.to_callable(True)
                    stashes.stash_and_restore_activations(fwd, dlinear)
                    dlinear.call(x)

        runner = pir_ext.Runner(ir, [], device_iterations=steps, device_type=2)
        runner.run({x_stream: data})  # type: ignore

        accum_tensor = dlinear.get_grad_accumulator_for_fwd_input(linear_graph.weight)
        weights = runner.read_weights([accum_tensor])
        runner.detach()

        return weights[accum_tensor]

    normal = graph()

    pipelined = pipelined_graph()

    np.testing.assert_almost_equal(normal, pipelined)
