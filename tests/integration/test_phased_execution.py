# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np

import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons import Module, RemoteBuffers
from popxl_addons.dot_tree import to_mapping
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.ops import host_load
from popxl_addons.ops.streams import host_store
from popxl_addons.transforms.phased import (NamedBuffers, all_gather_replica_sharded_tensors_io, load_from_buffers,
                                            remote_activations, remote_replica_sharded_variables, remote_variables,
                                            load_to_io_tiles, copy_from_io_tiles, copy_to_io_tiles, store_from_io_tiles,
                                            store_to_buffers)


class Add(Module):
    def build(self, x: popxl.Tensor):
        w = self.add_input_tensor("weight", partial(np.random.normal, 0, 0.02, x.shape), x.dtype, by_ref=True)
        ops.scaled_add_(w, x)


def test_phased_inference():
    ir = popxl.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = 32
    main = ir.main_graph
    buffers = RemoteBuffers()

    data = np.random.normal(0, 1, (1, 4)).astype(np.float32)

    with main, popxl.in_sequence():
        data, x_stream, x = host_load(data, popxl.float32, "x")

        args, graph = Add().create_graph(x)

        variables = [args.init(), args.init()]
        r_variables = [remote_variables(v, buffers) for v in variables]

        add1 = load_to_io_tiles(r_variables[0])
        add1 = copy_from_io_tiles(add1)

        # Overlap
        add2 = load_to_io_tiles(r_variables[1])
        graph.bind(add1).call(x)
        # -------

        add1 = copy_to_io_tiles(add1)
        add2 = copy_from_io_tiles(add2)

        # Overlap
        store_from_io_tiles(add1, r_variables[0])
        graph.bind(add2).call(x)
        # -------

        add2 = copy_to_io_tiles(add2)
        store_from_io_tiles(add2, r_variables[1])

    addons.Runner(ir, device_num=1).run({x_stream: data})


class Linear(Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        w = self.add_input_tensor("weight", partial(np.random.normal, 0, 1, (x.shape[-1], self.out_features)), x.dtype)
        return x @ w


def test_phased_training():
    def graph():
        np.random.seed(42)
        ir = popxl.Ir()
        main = ir.main_graph

        data = np.random.normal(0, 1, (1, 4)).astype(np.float32)

        with main, popxl.in_sequence():
            data, x_stream, x = host_load(data, popxl.float32, "x")
            results = []

            args, graph = Linear(4).create_graph(x)
            dgraph = addons.autodiff(graph)

            fwd1 = graph.bind(args.init())
            call_info_1 = fwd1.call_with_info(x)
            x, = call_info_1.outputs

            fwd2 = graph.bind(args.init())
            call_info_2 = fwd2.call_with_info(x)
            x, = call_info_2.outputs

            fwd3 = graph.bind(args.init())
            call_info_3 = fwd3.call_with_info(x)
            x, = call_info_3.outputs

            dx: popxl.Tensor
            dw: popxl.Tensor

            dx, dw = dgraph.call(x, args=dgraph.grad_graph_info.inputs_dict(call_info_3))
            results.append(dw)

            dx, dw = dgraph.call(dx, args=dgraph.grad_graph_info.inputs_dict(call_info_2))
            results.append(dw)

            _, dw = dgraph.call(dx, args=dgraph.grad_graph_info.inputs_dict(call_info_1))
            results.append(dw)

            outputs = tuple(host_store(t) for t in results)

        return addons.Runner(ir, outputs, device_num=1).run({x_stream: data})

    def phased_graph():
        np.random.seed(42)
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.numIOTiles = 32
        main = ir.main_graph
        buffers = RemoteBuffers()

        data = np.random.normal(0, 1, (1, 4)).astype(np.float32)

        with main, popxl.in_sequence():
            data, x_stream, x = host_load(data, popxl.float32, "x")
            results = []

            args, graph = Linear(4).create_graph(x)
            dgraph = addons.autodiff(graph)

            variables = [args.init(), args.init(), args.init()]
            r_variables = [remote_variables(v, buffers) for v in variables]

            load1 = load_to_io_tiles(r_variables[0])
            load1 = copy_from_io_tiles(load1)
            fwd1 = graph.bind(load1)

            # Overlap
            load2 = load_to_io_tiles(r_variables[1])
            call_info_1 = fwd1.call_with_info(x)
            x, = call_info_1.outputs
            # -------

            acts1 = remote_activations(call_info_1,
                                       dgraph.grad_graph_info,
                                       buffers,
                                       existing=to_mapping(load1, r_variables[0]))

            store1 = copy_to_io_tiles(acts1.to_store)
            load2 = copy_from_io_tiles(load2)
            fwd2 = graph.bind(load2)

            # Overlap
            with popxl.transforms.io_tile_exchange():
                store_from_io_tiles(store1, acts1.buffers)
                load3 = load_to_io_tiles(r_variables[2])
            call_info_2 = fwd2.call_with_info(x)
            x, = call_info_2.outputs
            # -------

            acts2 = remote_activations(call_info_2,
                                       dgraph.grad_graph_info,
                                       buffers,
                                       existing=to_mapping(load2, r_variables[1]))

            store2 = copy_to_io_tiles(acts2.to_store)
            load3 = copy_from_io_tiles(load3)
            fwd3 = graph.bind(load3)

            # Overlap - TODO: Don't store, just copy to io tiles
            store_from_io_tiles(store2, acts2.buffers)
            call_info_3 = fwd3.call_with_info(x)
            x, = call_info_3.outputs
            # ------

            dx: popxl.Tensor
            dw: popxl.Tensor

            # Overlap
            load2 = load_to_io_tiles(acts2.buffers)
            dx, dw = dgraph.call(x, args=dgraph.grad_graph_info.inputs_dict(call_info_3))
            # ------
            results.append(dw)

            load2 = copy_from_io_tiles(load2)

            # Overlap
            load1 = load_to_io_tiles(acts1.buffers)
            dx, dw = dgraph.call(dx, args=acts2.activation_map(load2))  # type: ignore
            # -------
            results.append(dw)

            load1 = copy_from_io_tiles(load1)

            _, dw = dgraph.call(dx, args=acts1.activation_map(load1))  # type: ignore
            results.append(dw)

            outputs = tuple(host_store(t) for t in results)

        return addons.Runner(ir, outputs, device_num=1).run({x_stream: data})

    normal = graph()
    phased = phased_graph()
    np.testing.assert_almost_equal(normal, phased)


# Any optimizer with state
class SGDM(addons.Module):
    @popxl.in_sequence()
    def build(self, w: popxl.TensorByRef, g: popxl.Tensor, m: float, lr: float):
        momentum = self.add_input_tensor("momentum", partial(np.zeros, w.shape), popxl.float32, by_ref=True)
        ops.scaled_add_(momentum, g, a=m, b=1 - m)
        ops.scaled_add_(w, momentum, b=lr)


def graph_with_optimizer():
    np.random.seed(42)
    ir = popxl.Ir()
    main = ir.main_graph

    data = np.random.normal(0, 1, (2, 4)).astype(np.float32)

    with main, popxl.in_sequence():
        data, x_stream, x = host_load(data, popxl.float32, "x")

        args, graph = Linear(4).create_graph(x)
        dgraph = addons.autodiff(graph)
        opt_args, opt_graph = SGDM().create_graph(graph.args.weight, graph.args.weight, 0.5, 1.0)

        variables = [args.init(), args.init(), args.init()]
        opt_vars = [opt_args.init(), opt_args.init(), opt_args.init()]

        fwd1 = graph.bind(variables[0])
        call_info_1 = fwd1.call_with_info(x)
        x, = call_info_1.outputs

        fwd2 = graph.bind(variables[1])
        call_info_2 = fwd2.call_with_info(x)
        x, = call_info_2.outputs

        fwd3 = graph.bind(variables[2])
        call_info_3 = fwd3.call_with_info(x)
        x, = call_info_3.outputs

        dx: popxl.Tensor
        dw: popxl.Tensor

        dx, dw = dgraph.call(x, args=dgraph.grad_graph_info.inputs_dict(call_info_3))
        opt_graph.bind(opt_vars[2]).call(variables[2].weight, dw)

        dx, dw = dgraph.call(dx, args=dgraph.grad_graph_info.inputs_dict(call_info_2))
        opt_graph.bind(opt_vars[1]).call(variables[1].weight, dw)

        _, dw = dgraph.call(dx, args=dgraph.grad_graph_info.inputs_dict(call_info_1))
        opt_graph.bind(opt_vars[0]).call(variables[0].weight, dw)

    runner = addons.Runner(ir, device_num=1)
    runner.run({x_stream: data})

    results = [*(v.weight for v in variables), *(v.momentum for v in opt_vars)]
    weights = runner.read_weights(results)
    runner.detach()
    return tuple(weights[t] for t in results)


def test_phased_training_with_optimizer():
    def phased_graph():
        np.random.seed(42)
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.numIOTiles = 32
        main = ir.main_graph
        buffers = RemoteBuffers()

        data = np.random.normal(0, 1, (2, 4)).astype(np.float32)

        with main, popxl.in_sequence():
            data, x_stream, x = host_load(data, popxl.float32, "x")

            args, graph = Linear(4).create_graph(x)
            dgraph = addons.autodiff(graph)

            with popxl.io_tiles():
                opt_args, opt_graph = SGDM().create_graph(graph.args.weight, graph.args.weight, 0.5, 1.0)

            variables = [args.init(), args.init(), args.init()]
            r_variables = [remote_variables(v, buffers) for v in variables]

            opt_vars = [opt_args.init(), opt_args.init(), opt_args.init()]
            r_opt_vars = [remote_variables(v, buffers) for v in opt_vars]

            var_load1_io = load_to_io_tiles(r_variables[0])
            var_load1 = copy_from_io_tiles(var_load1_io)

            # Overlap
            with popxl.transforms.io_tile_exchange():
                var_load2_io = load_to_io_tiles(r_variables[1])

            fwd1 = graph.bind(var_load1)
            call_info_1 = fwd1.call_with_info(x)
            x, = call_info_1.outputs
            # -------

            acts1 = remote_activations(call_info_1, dgraph.grad_graph_info, buffers,
                                       var_load1.to_mapping(r_variables[0]))

            act_store1 = copy_to_io_tiles(acts1.to_store)
            var_load2 = copy_from_io_tiles(var_load2_io)

            # Overlap
            with popxl.transforms.io_tile_exchange():
                store_from_io_tiles(act_store1, acts1.buffers)
                var_load3_io = load_to_io_tiles(r_variables[2])

            fwd2 = graph.bind(var_load2)
            call_info_2 = fwd2.call_with_info(x)
            x, = call_info_2.outputs
            # -------

            acts2 = remote_activations(call_info_2, dgraph.grad_graph_info, buffers,
                                       var_load2.to_mapping(r_variables[1]))

            act_store2 = copy_to_io_tiles(acts2.to_store)
            var_load3 = copy_from_io_tiles(var_load3_io)

            # Overlap
            with popxl.transforms.io_tile_exchange():
                store_from_io_tiles(act_store2, acts2.buffers)
                opt_load3_io = load_to_io_tiles(r_opt_vars[2])

            fwd3 = graph.bind(var_load3)
            call_info_3 = fwd3.call_with_info(x)
            x, = call_info_3.outputs
            # ------

            dx: popxl.Tensor
            dw: popxl.Tensor

            # Overlap
            with popxl.transforms.io_tile_exchange():
                act_load2_io, var_load2_io, opt_load2_io = \
                    load_to_io_tiles(acts2.buffers, r_variables[1], r_opt_vars[1])

            dx, dw = dgraph.call(x, args=dgraph.grad_graph_info.inputs_dict(call_info_3))
            # ------
            dw = ops.io_tile_copy(dw)
            with popxl.io_tiles():
                opt_graph.bind(opt_load3_io).call(var_load3_io.weight, dw)

            act_load2 = copy_from_io_tiles(act_load2_io)

            # # Overlap
            with popxl.transforms.io_tile_exchange():
                store_from_io_tiles(var_load3_io, r_variables[2])
                store_from_io_tiles(opt_load3_io, r_opt_vars[2])
                act_load1_io, var_load1_io, opt_load1_io = \
                    load_to_io_tiles(acts1.buffers, r_variables[0], r_opt_vars[0])

            dx, dw = dgraph.call(dx, args=acts2.activation_map(act_load2))  # type: ignore
            # -------
            dw = ops.io_tile_copy(dw)
            with popxl.io_tiles():
                opt_graph.bind(opt_load2_io).call(var_load2_io.weight, dw)

            act_load1 = copy_from_io_tiles(act_load1_io)

            # Overlap
            with popxl.transforms.io_tile_exchange():
                store_from_io_tiles(var_load2_io, r_variables[1])
                store_from_io_tiles(opt_load2_io, r_opt_vars[1])

            _, dw = dgraph.call(dx, args=acts1.activation_map(act_load1))  # type: ignore
            # -----
            ops.io_tile_copy(dw)
            with popxl.io_tiles():
                opt_graph.bind(opt_load1_io).call(var_load1_io.weight, dw)

            with popxl.transforms.io_tile_exchange():
                store_from_io_tiles(var_load1_io, r_variables[0])
                store_from_io_tiles(opt_load1_io, r_opt_vars[0])

        runner = addons.Runner(ir, device_num=1)
        runner.run({x_stream: data})

        results = [*(v.weight for v in variables), *(v.momentum for v in opt_vars)]
        weights = runner.read_weights(results)
        runner.detach()
        return tuple(weights[t] for t in results)

    normal = graph_with_optimizer()
    phased = phased_graph()
    np.testing.assert_almost_equal(normal, phased)


class SGDM_RTS(addons.Module):
    @popxl.in_sequence()
    def build(self, w: popxl.TensorByRef, g: popxl.Tensor, m: float, lr: float):
        momentum = self.add_replica_sharded_input_tensor("momentum",
                                                         partial(np.zeros, w.meta_shape),
                                                         popxl.float32,
                                                         by_ref=True)
        ops.scaled_add_(momentum, g, a=m, b=1 - m)
        ops.scaled_add_(w, momentum, b=lr)


def rts_spec(t):
    spec = popxl.constant(0, t.dtype)
    shape = (int(np.prod(t.shape)) // popxl.gcg().ir._pb_ir.getSessionOptions().replicatedGraphCount, )
    info = spec._pb_tensor.info
    info.set(info.dataType(), shape, t.shape)
    return spec


def test_phased_training_with_rts():
    def phased_graph():
        np.random.seed(42)
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.numIOTiles = 32
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = 2
        main = ir.main_graph
        buffers = RemoteBuffers()

        data = np.random.normal(0, 1, (2, 4)).astype(np.float32)

        with main, popxl.in_sequence():
            _, x_stream, x = host_load(data[0].reshape(1, 4), popxl.float32, "x")

            args, graph = Linear(4).create_graph(x)
            dgraph = addons.autodiff(graph)

            with popxl.io_tiles():
                rts_ = rts_spec(graph.args.weight)
                opt_args, opt_graph = SGDM_RTS().create_graph(rts_, rts_, 0.5, 1.0)

            variables = [args.init(), args.init(), args.init()]
            r_variables = [remote_replica_sharded_variables(v, buffers, 0) for v in variables]

            opt_vars = [opt_args.init(), opt_args.init(), opt_args.init()]
            r_opt_vars = [remote_replica_sharded_variables(v, buffers, 0) for v in opt_vars]

            var_load1_io = load_to_io_tiles(r_variables[0])
            var_load1 = copy_from_io_tiles(all_gather_replica_sharded_tensors_io(var_load1_io))

            fwd1 = graph.bind(var_load1)

            with popxl.transforms.io_tile_exchange():
                var_load2_io = load_from_buffers(r_variables[1])
            call_info_1 = fwd1.call_with_info(x)
            x, = call_info_1.outputs
            acts1 = remote_activations(call_info_1, dgraph.grad_graph_info, buffers,
                                       var_load1.to_mapping(r_variables[0]))

            act_store1 = copy_to_io_tiles(acts1.to_store)
            var_load2 = copy_from_io_tiles(all_gather_replica_sharded_tensors_io(var_load2_io))

            fwd2 = graph.bind(var_load2)

            with popxl.transforms.io_tile_exchange():
                store_to_buffers(act_store1, acts1.buffers)
                var_load3_io = load_from_buffers(r_variables[2])
            call_info_2 = fwd2.call_with_info(x)
            x, = call_info_2.outputs
            acts2 = remote_activations(call_info_2, dgraph.grad_graph_info, buffers,
                                       var_load2.to_mapping(r_variables[1]))

            act_store2 = copy_to_io_tiles(acts2.to_store)
            var_load3 = copy_from_io_tiles(all_gather_replica_sharded_tensors_io(var_load3_io))

            fwd3 = graph.bind(var_load3)

            with popxl.transforms.io_tile_exchange():
                store_to_buffers(act_store2, acts2.buffers)
            call_info_3 = fwd3.call_with_info(x)
            x, = call_info_3.outputs

            dx: popxl.Tensor
            dw: popxl.Tensor

            to_load = NamedBuffers(acts=acts2.buffers, optim=r_opt_vars[2])
            with popxl.transforms.io_tile_exchange():
                loaded_io = load_from_buffers(to_load)
            dx, dw = dgraph.call(x, args=dgraph.grad_graph_info.inputs_dict(call_info_3))

            dw = ops.io_tile_copy(dw)
            with popxl.io_tiles():
                dw = ops.collectives.replicated_reduce_scatter(dw, 'add', None, True)
                opt_graph.bind(loaded_io.optim).call(var_load3_io.weight, dw)

            act_load2 = copy_from_io_tiles(all_gather_replica_sharded_tensors_io(loaded_io.acts))

            to_store = NamedTensors(fwd=var_load3_io, optim=loaded_io.optim)
            store_buffers = NamedBuffers(fwd=r_variables[2], optim=r_opt_vars[2])
            to_load = NamedBuffers(acts=acts1.buffers, fwd=r_variables[1], optim=r_opt_vars[1])

            with popxl.transforms.io_tile_exchange():
                store_to_buffers(to_store, store_buffers)
                loaded_io = load_from_buffers(to_load)
            dx, dw = dgraph.call(dx, args=acts2.activation_map(act_load2))

            dw = ops.io_tile_copy(dw)
            with popxl.io_tiles():
                dw = ops.collectives.replicated_reduce_scatter(dw, 'add', None, True)
                opt_graph.bind(loaded_io.optim).call(loaded_io.fwd.weight, dw)

            act_load1 = copy_from_io_tiles(all_gather_replica_sharded_tensors_io(loaded_io.acts))

            to_store = NamedTensors(fwd=loaded_io.fwd, optim=loaded_io.optim)
            store_buffers = NamedBuffers(fwd=r_variables[1], optim=r_opt_vars[1])
            to_load = NamedBuffers(fwd=r_variables[0], optim=r_opt_vars[0])

            with popxl.transforms.io_tile_exchange():
                store_to_buffers(to_store, store_buffers)
                loaded_io = load_from_buffers(to_load)
            dx, dw = dgraph.call(dx, args=acts1.activation_map(act_load1))

            dw = ops.io_tile_copy(dw)
            with popxl.io_tiles():
                dw = ops.collectives.replicated_reduce_scatter(dw, 'add', None, True)
                opt_graph.bind(loaded_io.optim).call(loaded_io.fwd.weight, dw)

            with popxl.transforms.io_tile_exchange():
                store_from_io_tiles(loaded_io.fwd, r_variables[0])
                store_from_io_tiles(loaded_io.optim, r_opt_vars[0])

        runner = addons.Runner(ir, device_num=2)
        runner.run({x_stream: data})

        results = [*(v.weight for v in variables), *(v.momentum for v in opt_vars)]
        weights = runner.read_weights(results)
        runner.detach()
        return tuple(weights[t] for t in results)

    normal = graph_with_optimizer()
    phased = phased_graph()

    # layer0
    np.testing.assert_almost_equal(normal[0], phased[0])
    np.testing.assert_almost_equal(normal[3], phased[3])
    # layer1
    np.testing.assert_almost_equal(normal[1], phased[1])
    np.testing.assert_almost_equal(normal[4], phased[4])
    # layer2
    np.testing.assert_almost_equal(normal[2], phased[2])
    np.testing.assert_almost_equal(normal[5], phased[5])
