# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np

import popxl
from popxl import ops

import popxl_addons.array_munging
from popxl_addons import Module, host_load, NamedTensors, named_variable_buffers, load_remote_graph, store_remote_graph
from popxl_addons.rts import all_gather_replica_sharded_graph, reduce_replica_sharded_graph


class Add(Module):
    def build(self, x: popxl.Tensor):
        w = self.add_variable_input("weight", partial(np.random.normal, 0, 0.02, x.shape), x.dtype, by_ref=True)
        ops.scaled_add_(w, x)


def test_phased_load_store():
    ir = popxl.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = 32
    main = ir.main_graph

    data = np.random.normal(0, 1, (1, 4)).astype(np.float32)

    with main, popxl.in_sequence():
        data, x_stream, x = host_load(data, popxl.float32, "x")

        args, graph = Add().create_graph(x)

        buffers = named_variable_buffers(args, 2)
        load, load_names = load_remote_graph(buffers)
        store = store_remote_graph(buffers)

        variables = NamedTensors(layer0=args.init_remote(buffers, 0), layer1=args.init_remote(buffers, 1))

        add0 = NamedTensors.pack(load_names, load.call(0))
        graph.bind(add0).call(x)
        store.bind(add0).call(0)

        add1 = NamedTensors.pack(load_names, load.call(1))
        graph.bind(add1).call(x)
        store.bind(add1).call(1)

    sess = popxl.Session(ir, "ipu_hw")
    before = {t: np_t.copy() for t, np_t in sess.get_tensors_data(variables.tensors).items()}
    with sess:
        sess.run({x_stream: data})
    after = sess.get_tensors_data(variables.tensors)
    for t in before.keys():
        np.testing.assert_almost_equal(before[t] + data, after[t])


def test_phased_rts():
    ir = popxl.Ir(replication=4)

    data = np.random.normal(0, 1, (1, 4)).astype(np.float32)

    with ir.main_graph, popxl.in_sequence():
        data, x_stream, x = host_load(data, popxl.float32, "x")

        args, graph = Add().create_graph(x)

        buffers = named_variable_buffers(args, 2, {k: 4 for k in args.keys_flat()})
        load, names = load_remote_graph(buffers)
        gather_, _ = all_gather_replica_sharded_graph(
            NamedTensors.pack(names, load.graph.outputs), replica_groups=args.replica_groupings
        )
        reduce_, _ = reduce_replica_sharded_graph(
            NamedTensors.pack(names, gather_.graph.outputs), shard_groups=args.replica_groupings
        )
        store = store_remote_graph(buffers)

        variables = NamedTensors(layer0=args.init_remote(buffers, 0), layer1=args.init_remote(buffers, 1))

        add0 = NamedTensors.pack(names, load.call(0))
        add0 = NamedTensors.pack(names, gather_.bind(add0).call())
        graph.bind(add0).call(x)
        add0 = NamedTensors.pack(names, reduce_.bind(add0).call())
        store.bind(add0).call(0)

        add1 = NamedTensors.pack(names, load.call(1))
        add1 = NamedTensors.pack(names, gather_.bind(add1).call())
        graph.bind(add1).call(x)
        add1 = NamedTensors.pack(names, reduce_.bind(add1).call())
        store.bind(add1).call(1)

    replicated_data = popxl_addons.array_munging.repeat(data, ir.replication_factor, axis=0)

    sess = popxl.Session(ir, "ipu_hw")
    before = {t: np_t.copy() for t, np_t in sess.get_tensors_data(variables.tensors).items()}
    with sess:
        sess.run({x_stream: replicated_data})
    after = sess.get_tensors_data(variables.tensors)
    for t in before.keys():
        np.testing.assert_almost_equal((before[t] + data) * 4, after[t])


if __name__ == "__main__":
    test_phased_load_store()
