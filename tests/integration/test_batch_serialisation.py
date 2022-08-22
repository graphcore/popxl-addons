# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import fractions
import pytest
from functools import partial
import numpy as np
import popxl
from popxl import io_tiles, ops
from typing import List, Optional, Callable

import popxl_addons as addons
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.batch_serialisation import (batch_serial_buffer, batch_serialise,
                                                         batch_serialise_fwd_and_grad, RemoteHandle)

from popxl_addons.layers import Linear, LayerNorm
from popxl_addons.ops.replicated_all_reduce_TP import (replicated_all_reduce_identical_inputs,
                                                       replicated_all_reduce_identical_grad_inputs)
from popxl_addons.patterns import apply_pre_alias_patterns
import popart
from popxl.transforms.autodiff import ExpectedConnectionType
from contextlib import contextmanager


class Scale(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        scale = self.add_variable_input("scale", partial(np.full, x.shape, 2), x.dtype)
        return x * scale


@pytest.mark.parametrize('io_mode', ['compute', 'io', 'io_overlapped'])
def test_batch_serialisation_fwd_single(io_mode):
    ir = popxl.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = 64
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
            store_buffers={t: (batch_serial_buffer(t, steps=bf), 0)
                           for t in graph.graph.outputs},
            io_mode=io_mode)

        # Create variables and bind
        scale = bs_result.graph.bind(args.init())

        # Create Program
        scale.call(0)  # Call with offset=0

        # Load values out of the remote buffers
        out_buffer = bs_result.stored_buffers[graph.graph.outputs[0]].buffer
        out = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype)
        for i in range(bf):
            y = ops.remote_load(out_buffer, i)
            ops.host_store(out, y)

    inputs = np.arange(bf * np.prod(in_h2d.shape)).reshape((bf, *in_h2d.shape)).astype(in_h2d.dtype.as_numpy())

    ir.num_host_transfers = bf
    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run({in_h2d: inputs})[out]
    np.testing.assert_equal(inputs * 2, outputs)


@pytest.mark.parametrize('io_mode', ['compute', 'io', 'io_overlapped'])
def test_batch_serialisation_entries(io_mode):
    ir = popxl.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = 64
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
            store_buffers={t: (batch_serial_buffer(t, steps=bf), 0)
                           for t in graph.graph.outputs},
            rows=2,
            io_mode=io_mode)

        # Create variables and bind
        scale = bs_result.graph.bind(args.init())

        # Create Program
        scale.call(1)  # Call with offset=1

        # Load values out of the remote buffers
        out_buffer = bs_result.stored_buffers[graph.graph.outputs[0]].buffer
        out = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype)
        for i in range(bf):
            y = ops.remote_load(out_buffer, bf + i)
            ops.host_store(out, y)

    inputs = np.arange(bf * np.prod(in_h2d.shape)).reshape((bf, *in_h2d.shape)).astype(in_h2d.dtype.as_numpy())

    ir.num_host_transfers = bf
    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run({in_h2d: inputs})[out]
    np.testing.assert_equal(inputs * 2, outputs)


@pytest.mark.parametrize('io_mode', ['compute', 'io', 'io_overlapped'])
def test_batch_serialisation_sequence(io_mode):
    ir = popxl.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = 64
    main = ir.main_graph

    cb = 2
    bf = 4

    with main, popxl.in_sequence(True):
        in_h2d = popxl.h2d_stream((cb, 2), popxl.float32, name="in_stream")
        out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
        # Create graphs
        args, graph = Scale().create_graph(in_h2d.spec)

        # Transform graphs
        buffer = batch_serial_buffer(graph.graph.outputs[0], steps=bf)
        bs_load = batch_serialise(graph,
                                  bf,
                                  load_handles={graph.graph.inputs[0]: in_h2d},
                                  store_streams={},
                                  store_buffers={graph.graph.outputs[0]: (buffer, 0)},
                                  io_mode=io_mode)

        bs_remote = batch_serialise(graph,
                                    bf,
                                    load_handles={graph.graph.inputs[0]: (buffer, 0)},
                                    store_streams={},
                                    store_buffers={graph.graph.outputs[0]: (buffer, 1)},
                                    rows=2,
                                    io_mode=io_mode)

        bs_store = batch_serialise(graph,
                                   bf,
                                   load_handles={graph.graph.inputs[0]: (buffer, 2)},
                                   store_streams={graph.graph.outputs[0]: out_d2h},
                                   store_buffers={},
                                   io_mode=io_mode)

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
    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run({in_h2d: inputs})[out_d2h]

    np.testing.assert_equal(inputs * (2**4), outputs)


@pytest.mark.parametrize('io_mode', ['compute', 'io', 'io_overlapped'])
def test_batch_serialisation_grad(io_mode):
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
            args, graph = Linear(2).create_graph(in_h2d.spec)
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
        sess.write_variable_data(weights.weight, w)
        with sess:
            out = sess.run({in_h2d: inputs.reshape(-1, 2)})[out_d2h]
        return out

    def batch_serial():
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.numIOTiles = 64
        main = ir.main_graph

        with main:
            # --- Create inputs
            in_h2d = popxl.h2d_stream((cb, 2), popxl.float32, name="in_stream")
            out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
            # --- Create graphs
            args, graph = Linear(2).create_graph(in_h2d.spec)
            dargs, dgraph = addons.autodiff_with_accumulation(graph,
                                                              graph.args.tensors,
                                                              grads_required=graph.graph.inputs[:1])

            # --- Transform graphs
            input_grad = dgraph.graph.inputs[0]
            grad_buffer = batch_serial_buffer(input_grad, steps=bf)
            bs_fwd, bs_grad = batch_serialise_fwd_and_grad(graph,
                                                           dgraph,
                                                           graph.args,
                                                           bf,
                                                           load_handles={
                                                               graph.graph.inputs[0]: in_h2d,
                                                               input_grad: (grad_buffer, 0)
                                                           },
                                                           store_streams={dgraph.graph.outputs[0]: out_d2h},
                                                           store_buffers={},
                                                           io_mode=io_mode)

            # --- Create variables and bind
            weights = args.init()
            fwd = bs_fwd.graph.bind(weights)

            grad_vars = dargs.init()
            grad_vars.update(weights)
            grad = bs_grad.graph.bind(grad_vars)

            # --- Create Program
            with popxl.in_sequence():
                # Store upstream grad in input buffer
                for i in range(bf):
                    ops.remote_store(grad_buffer, i, popxl.constant(grad_inputs[i]))

                # Call forward
                fwd.call(popxl.constant(0))
                # Call gradient
                grad.call(popxl.constant(0))

        ir.num_host_transfers = bf
        sess = popxl.Session(ir, "ipu_hw")
        sess.write_variable_data(weights.weight, w)
        with sess:
            out = sess.run({in_h2d: inputs})[out_d2h]
        return out

    norm = normal()
    bs = batch_serial()
    np.testing.assert_almost_equal(norm.reshape(-1), bs.reshape(-1))


@pytest.mark.parametrize('io_mode', ['compute', 'io', 'io_overlapped'])
def test_batch_serialisation_rb_only_grad(io_mode):
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
            args, graph = Linear(2).create_graph(in_h2d.spec)
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
        sess.write_variable_data(weights.weight, w)
        with sess:
            out = sess.run({in_h2d: inputs.reshape(-1, 2)})[out_d2h]
        return out

    def batch_serial():
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.numIOTiles = 64
        main = ir.main_graph

        with main:
            # --- Create inputs
            in_h2d = popxl.h2d_stream((cb, 2), popxl.float32, name="in_stream")
            out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
            # --- Create graphs
            args, graph = Linear(2).create_graph(in_h2d.spec)
            dargs, dgraph = addons.autodiff_with_accumulation(graph,
                                                              graph.args.tensors,
                                                              grads_required=graph.graph.inputs[:1])

            input_rb = popxl.remote_buffer(in_h2d.shape, in_h2d.dtype, bf)

            # --- Transform graphs
            input_grad = dgraph.graph.inputs[0]
            grad_buffer = batch_serial_buffer(input_grad, steps=bf)
            bs_fwd, bs_grad = batch_serialise_fwd_and_grad(graph,
                                                           dgraph,
                                                           graph.args,
                                                           bf,
                                                           load_handles={
                                                               graph.graph.inputs[0]: (input_rb, 0),
                                                               input_grad: (grad_buffer, 0)
                                                           },
                                                           store_streams={dgraph.graph.outputs[0]: out_d2h},
                                                           store_buffers={},
                                                           io_mode=io_mode)

            # --- Create variables and bind
            weights = args.init()
            fwd = bs_fwd.graph.bind(weights)

            grad_vars = dargs.init()
            grad_vars.update(weights)
            grad = bs_grad.graph.bind(grad_vars)
            # --- Create Program
            with popxl.in_sequence():
                # Store input and upstream grad in buffers
                for i in range(bf):
                    ops.remote_store(input_rb, i, popxl.constant(inputs[i]))
                    ops.remote_store(grad_buffer, i, popxl.constant(grad_inputs[i]))

                # Call forward
                fwd.call(0)
                # Call gradient
                grad.call(0)

        ir.num_host_transfers = bf
        sess = popxl.Session(ir, "ipu_hw")
        sess.write_variable_data(weights.weight, w)
        with sess:
            out = sess.run()[out_d2h]
        return out

    norm = normal()
    bs = batch_serial()
    np.testing.assert_almost_equal(norm.reshape(-1), bs.reshape(-1))


class FeedForwardTP(addons.Module):
    def __init__(self, tensor_parallel: int, data_parallel: int, fc1_size: int):
        super().__init__()
        tp = tensor_parallel
        dp = data_parallel
        self.n_shards = tp
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        assert fc1_size % self.n_shards == 0
        # ----- Layers -----
        # Sharded across devices - column wise
        self.fc1 = Linear(fc1_size // self.n_shards, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor) -> List[popxl.Tensor]:
        """Identical input (x, seed) and identical output across shards."""
        # ----- Identical computation -----
        #z = self.ln_2(x)
        z = x
        z = replicated_all_reduce_identical_inputs(z, group=self.replica_grouping.transpose())

        # ----- Sharded computation -----

        z = self.fc1(z)
        z = ops.gelu(z)
        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())

        # ----- Identical computation -----
        self.bias = self.add_variable_input('bias', lambda: np.zeros(z.shape[-1]), z.dtype)
        z = z + self.bias
        z = ops.gelu(z)
        return z


@pytest.mark.parametrize('io_mode', ['compute', 'io', 'io_overlapped'])
def test_batch_serialisation_sharded_activations(io_mode):
    cb = 3
    bf = 4

    tp = 4
    dp = 2
    fc1_shard_size = 16
    input_size = 8
    np.random.seed(42)
    rf = tp * dp

    input_shape = (bf, cb, 8)
    inputs = np.random.normal(0, 1, (rf, *input_shape)).astype(np.float32)
    grad_inputs = np.random.normal(0, 1, (bf, cb, 16)).astype(np.float32)
    w = np.random.normal(0, 1, (tp, input_size, fc1_shard_size)).astype(np.float32)
    b = np.ones((tp, fc1_shard_size)).astype(np.float32)
    shared_bias = np.full((fc1_shard_size, ), 2).astype(np.float32)

    def normal():
        ir = popxl.Ir()
        main = ir.main_graph
        ir.replication_factor = tp * dp

        with main:
            # --- Create inputs
            in_h2d = popxl.h2d_stream(input_shape, popxl.float32, name="in_stream")
            out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
            # --- Create graphs
            args, graph = FeedForwardTP(tp, dp, tp * fc1_shard_size).create_graph(in_h2d.spec)
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
                dx, = grad.call(popxl.constant(grad_inputs), args=dgraph.grad_graph_info.inputs_dict(fwd_info))
                ops.host_store(out_d2h, dx)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level='default')

        sess = popxl.Session(ir, "ipu_hw")

        with sess:
            sess.write_variables_data({weights.fc1.weight: w, weights.fc1.bias: b, weights.bias: shared_bias})
            out = sess.run({in_h2d: inputs})[out_d2h]
        return out

    def batch_serial():
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.numIOTiles = 64
        main = ir.main_graph
        ir.replication_factor = tp * dp

        tp_group = ir.replica_grouping(stride=1, group_size=tp)
        with main:
            # --- Create inputs
            in_h2d = popxl.h2d_stream((cb, input_size), popxl.float32, name="in_stream")
            out_d2h = popxl.d2h_stream(in_h2d.shape, in_h2d.dtype, name="out_stream")
            # --- Create graphs
            args, graph = FeedForwardTP(tp, dp, tp * fc1_shard_size).create_graph(in_h2d.spec)
            dargs, dgraph = addons.autodiff_with_accumulation(graph,
                                                              graph.args.tensors,
                                                              grads_required=graph.graph.inputs[:1])

            activations = [
                i_ec[1].fwd_tensor for i_ec in enumerate(dgraph.grad_graph_info._expected_inputs) if
                i_ec[1].connection_type == ExpectedConnectionType.Fwd and i_ec[1].fwd_tensor not in graph.args.tensors
            ]
            activations_shard_groups = {activations[0]: tp_group}
            # --- Transform graphs
            input_grad = dgraph.graph.inputs[0]

            # all dp replicas will compute on the same data in this case
            grad_buffer = batch_serial_buffer(input_grad, steps=bf, sharded_threshold=0, shard_group=tp_group)
            bs_fwd, bs_grad = batch_serialise_fwd_and_grad(graph,
                                                           dgraph,
                                                           graph.args,
                                                           bf,
                                                           load_handles={
                                                               graph.graph.inputs[0]: in_h2d,
                                                               input_grad: (grad_buffer, 0, tp_group)
                                                           },
                                                           store_streams={dgraph.graph.outputs[0]: out_d2h},
                                                           store_buffers=activations_shard_groups,
                                                           io_mode=io_mode,
                                                           sharded_threshold=0)
            # --- Create variables and bind
            weights = args.init()
            fwd = bs_fwd.graph.bind(weights)

            grad_vars = dargs.init()
            grad_vars.update(weights)
            grad = bs_grad.graph.bind(grad_vars)

            @contextmanager
            def null_context():
                yield

            tileset = popxl.io_tiles if io_mode == 'io_overlapped' or io_mode == 'io' else null_context

            # --- Create Program
            with popxl.in_sequence():
                # Store upstream grad in input buffer
                for i in range(bf):
                    v = popxl.variable(grad_inputs[i])
                    v = ops.collectives.replica_sharded_slice(v, group=tp_group)
                    if io_mode == 'io_overlapped' or io_mode == 'io':
                        v = ops.io_tile_copy(v)

                    with tileset():
                        ops.remote_store(grad_buffer, i, v)

                # Call forward
                fwd.call(popxl.constant(0))
                # Call gradient
                grad.call(popxl.constant(0))

        ir.num_host_transfers = bf
        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level='default')

        sess = popxl.Session(ir, "ipu_hw")
        with sess:
            sess.write_variables_data({weights.fc1.weight: w, weights.fc1.bias: b, weights.bias: shared_bias})
            out = sess.run({in_h2d: np.transpose(inputs, (1, 0, 2, 3))})[out_d2h]
        return np.transpose(out, (1, 0, 2, 3))

    norm = normal()
    bs = batch_serial()
    np.testing.assert_almost_equal(norm.reshape(-1), bs.reshape(-1))


if __name__ == '__main__':
    test_batch_serialisation_sharded_activations('compute')
