# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons.transforms.batch_serialisation import (batch_serial_buffer, batch_serialise,
                                                         batch_serialise_fwd_and_grad)


class Scale(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        scale = self.add_input_tensor("scale", partial(np.full, x.shape, 2), x.dtype)
        return x * scale


def test_batch_serialisation_fwd_single():
    ir = popxl.Ir()
    main = ir.main_graph

    cb = 2
    bf = 4

    with main, popxl.in_sequence(True):
        in_h2d = popxl.h2d_stream((cb, 2), popxl.float32, name="x_stream")
        # Create graphs
        args, graph = Scale().create_graph(in_h2d.spec)

        # Transform graphs
        bs_result = batch_serialise(
            graph,
            bf,
            load_handles={graph.graph.inputs[0]: in_h2d},
            store_streams={},
            store_buffers={t: batch_serial_buffer(t)
                           for t in graph.graph.outputs},
        )

        # Create variables and bind
        scale = bs_result.graph.bind(args.init())

        # Create Program
        scale.call(0)  # Call with offset=0

        # Load values out of the remote buffers
        out_buffer, _ = bs_result.stored_buffers[graph.graph.outputs[0]]
        out = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype)
        for i in range(bf):
            y = ops.remote_load(out_buffer, i)
            ops.host_store(out, y)

    inputs = np.arange(bf * np.prod(in_h2d.shape)).reshape((bf, *in_h2d.shape)).astype(in_h2d.dtype.as_numpy())

    ir.num_host_transfers = bf
    outputs = popxl.Session(ir, "ipu_hw").run({in_h2d: inputs})[out]
    np.testing.assert_equal(inputs * 2, outputs)


def test_batch_serialisation_entries():
    ir = popxl.Ir()
    main = ir.main_graph

    cb = 2
    bf = 4

    with main, popxl.in_sequence(True):
        in_h2d = popxl.h2d_stream((cb, 2), popxl.float32, name="x_stream")
        # Create graphs
        args, graph = Scale().create_graph(in_h2d.spec)

        # Transform graphs
        bs_result = batch_serialise(graph,
                                    bf,
                                    load_handles={graph.graph.inputs[0]: in_h2d},
                                    store_streams={},
                                    store_buffers={t: batch_serial_buffer(t)
                                                   for t in graph.graph.outputs},
                                    entries=2)

        # Create variables and bind
        scale = bs_result.graph.bind(args.init())

        # Create Program
        scale.call(1)  # Call with offset=1

        # Load values out of the remote buffers
        out_buffer, _ = bs_result.stored_buffers[graph.graph.outputs[0]]
        out = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype)
        for i in range(bf):
            y = ops.remote_load(out_buffer, i)
            ops.host_store(out, y)

    inputs = np.arange(bf * np.prod(in_h2d.shape)).reshape((bf, *in_h2d.shape)).astype(in_h2d.dtype.as_numpy())

    ir.num_host_transfers = bf
    outputs = popxl.Session(ir, "ipu_hw").run({in_h2d: inputs})[out]
    np.testing.assert_equal(inputs * 2, outputs)


def test_batch_serialisation_sequence():
    ir = popxl.Ir()
    main = ir.main_graph

    cb = 2
    bf = 4

    with main, popxl.in_sequence(True):
        in_h2d = popxl.h2d_stream((cb, 2), popxl.float32, name="in_stream")
        out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
        # Create graphs
        args, graph = Scale().create_graph(in_h2d.spec)

        # Transform graphs
        buffer, _ = batch_serial_buffer(graph.graph.outputs[0])
        bs_load = batch_serialise(graph,
                                  bf,
                                  load_handles={graph.graph.inputs[0]: in_h2d},
                                  store_streams={},
                                  store_buffers={graph.graph.outputs[0]: (buffer, 0)})

        bs_remote = batch_serialise(graph,
                                    bf,
                                    load_handles={graph.graph.inputs[0]: (buffer, 0)},
                                    store_streams={},
                                    store_buffers={graph.graph.outputs[0]: (buffer, 1)},
                                    entries=2)

        bs_store = batch_serialise(graph,
                                   bf,
                                   load_handles={graph.graph.inputs[0]: (buffer, 2)},
                                   store_streams={graph.graph.outputs[0]: out_d2h},
                                   store_buffers={})

        # Create variables and bind
        var = args.init()
        scale_load = bs_load.graph.bind(var)
        scale_remote = bs_remote.graph.bind(var)
        scale_store = bs_store.graph.bind(var)

        # Create Program
        scale_load.call(0)

        scale_remote.call(0)
        scale_remote.call(1)

        scale_store.call(0)

    inputs = np.arange(bf * np.prod(in_h2d.shape)).reshape((bf, *in_h2d.shape)).astype(in_h2d.dtype.as_numpy())

    ir.num_host_transfers = bf
    outputs = popxl.Session(ir, "ipu_hw").run({in_h2d: inputs})[out_d2h]

    np.testing.assert_equal(inputs * (2**4), outputs)


class Linear(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        w = self.add_input_tensor("w", partial(np.random.normal, 0, 0.1, (2, 2)), x.dtype)
        return x @ w


def test_batch_serialisation_grad():
    cb = 2
    bf = 4

    inputs = np.random.normal(0, 1, (bf, cb, 2)).astype(np.float32)
    w = np.random.normal(0, 0.1, (2, 2)).astype(np.float32)
    grad_inputs = np.random.normal(0, 1, (bf, cb, 2)).astype(np.float32)

    def normal():
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            # --- Create inputs
            in_h2d = popxl.h2d_stream((bf * cb, 2), popxl.float32, name="in_stream")
            out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
            # --- Create graphs
            args, graph = Linear().create_graph(in_h2d.spec)
            dargs, dgraph = addons.autodiff_with_accumulation(graph,
                                                              graph.args.tensors,
                                                              grads_required=graph.graph.inputs[:1])

            # --- Create variables and bind
            weights = args.init()
            fwd = graph.bind(weights)
            grad = dgraph.bind(dargs.init())

            # --- Create Program
            with popxl.in_sequence():
                x = ops.host_load(in_h2d)
                # Call forward
                fwd_info = fwd.call_with_info(x)
                # Call gradient
                dx, = grad.call(popxl.constant(grad_inputs.reshape(-1, 2)),
                                args=dgraph.grad_graph_info.inputs_dict(fwd_info))

                ops.host_store(out_d2h, dx)

        sess = popxl.Session(ir, "ipu_hw")
        sess.write_variable_data(weights.w, w)
        out = sess.run({in_h2d: inputs.reshape(-1, 2)})[out_d2h]
        sess.device.detach()
        return out

    def batch_serial():
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            # --- Create inputs
            in_h2d = popxl.h2d_stream((cb, 2), popxl.float32, name="in_stream")
            out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
            # --- Create graphs
            args, graph = Linear().create_graph(in_h2d.spec)
            dargs, dgraph = addons.autodiff_with_accumulation(graph,
                                                              graph.args.tensors,
                                                              grads_required=graph.graph.inputs[:1])

            # --- Transform graphs
            input_grad = dgraph.graph.inputs[0]
            grad_buffer, _ = batch_serial_buffer(input_grad)
            bs_fwd, bs_grad, named_expected_inputs = batch_serialise_fwd_and_grad(
                graph,
                dgraph,
                bf,
                load_handles={
                    graph.graph.inputs[0]: in_h2d,
                    input_grad: (grad_buffer, 0)
                },
                store_streams={dgraph.graph.outputs[0]: out_d2h},
                store_buffers={})

            # --- Create variables and bind
            weights = args.init()
            fwd = bs_fwd.graph.bind(weights)
            grad = bs_grad.graph.bind(dargs.init())

            # --- Create Program
            with popxl.in_sequence():
                # Store upstream grad in input buffer
                for i in range(bf):
                    ops.remote_store(grad_buffer, i, popxl.constant(grad_inputs[i]))

                # Call forward
                fwd.call(popxl.constant(0))
                # Call gradient
                grad.call(popxl.constant(0), args=named_expected_inputs.to_mapping(weights))

        ir.num_host_transfers = bf
        sess = popxl.Session(ir, "ipu_hw")
        sess.write_variable_data(weights.w, w)
        out = sess.run({in_h2d: inputs})[out_d2h]
        sess.device.detach()
        return out

    norm = normal()
    bs = batch_serial()
    np.testing.assert_almost_equal(norm.reshape(-1), bs.reshape(-1))


def test_batch_serialisation_grad_remote_buffer():
    cb = 2
    bf = 4

    inputs = np.random.normal(0, 1, (bf, cb, 2)).astype(np.float32)
    w = np.random.normal(0, 0.1, (2, 2)).astype(np.float32)
    grad_inputs = np.random.normal(0, 1, (bf, cb, 2)).astype(np.float32)

    def normal():
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            # --- Create inputs
            in_h2d = popxl.h2d_stream((bf * cb, 2), popxl.float32, name="in_stream")
            out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
            # --- Create graphs
            args, graph = Linear().create_graph(in_h2d.spec)
            dargs, dgraph = addons.autodiff_with_accumulation(graph,
                                                              graph.args.tensors,
                                                              grads_required=graph.graph.inputs[:1])

            # --- Create variables and bind
            weights = args.init()
            fwd = graph.bind(weights)
            grad = dgraph.bind(dargs.init())

            # --- Create Program
            with popxl.in_sequence():
                x = ops.host_load(in_h2d)
                # Call forward
                fwd_info = fwd.call_with_info(x)
                # Call gradient
                dx, = grad.call(popxl.constant(grad_inputs.reshape(-1, 2)),
                                args=dgraph.grad_graph_info.inputs_dict(fwd_info))

                ops.host_store(out_d2h, dx)

        sess = popxl.Session(ir, "ipu_hw")
        sess.write_variable_data(weights.w, w)
        out = sess.run({in_h2d: inputs.reshape(-1, 2)})[out_d2h]
        sess.device.detach()
        return out

    def batch_serial():
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            # --- Create inputs
            in_h2d = popxl.h2d_stream((cb, 2), popxl.float32, name="in_stream")
            out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
            # --- Create graphs
            args, graph = Linear().create_graph(in_h2d.spec)
            dargs, dgraph = addons.autodiff_with_accumulation(graph,
                                                              graph.args.tensors,
                                                              grads_required=graph.graph.inputs[:1])

            input_rb = popxl.remote_buffer(in_h2d.shape, in_h2d.dtype, bf)

            # --- Transform graphs
            input_grad = dgraph.graph.inputs[0]
            grad_buffer, _ = batch_serial_buffer(input_grad)
            bs_fwd, bs_grad, named_expected_inputs = batch_serialise_fwd_and_grad(
                graph,
                dgraph,
                bf,
                load_handles={
                    graph.graph.inputs[0]: (input_rb, 0),
                    input_grad: (grad_buffer, 0)
                },
                store_streams={dgraph.graph.outputs[0]: out_d2h},
                store_buffers={})

            # --- Create variables and bind
            weights = args.init()
            fwd = bs_fwd.graph.bind(weights)
            grad = bs_grad.graph.bind(dargs.init())

            # --- Create Program
            with popxl.in_sequence():
                # Store input and upstream grad in buffers
                for i in range(bf):
                    ops.remote_store(input_rb, i, popxl.constant(inputs[i]))
                    ops.remote_store(grad_buffer, i, popxl.constant(grad_inputs[i]))

                # Call forward
                fwd.call(0)
                # Call gradient
                grad.call(0, args=named_expected_inputs.to_mapping(weights))

        ir.num_host_transfers = bf
        sess = popxl.Session(ir, "ipu_hw")
        sess.write_variable_data(weights.w, w)
        out = sess.run()[out_d2h]
        sess.device.detach()
        return out

    norm = normal()
    bs = batch_serial()
    np.testing.assert_almost_equal(norm.reshape(-1), bs.reshape(-1))
