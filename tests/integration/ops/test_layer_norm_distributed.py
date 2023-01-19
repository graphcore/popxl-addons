# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

import popxl
import popxl_addons as addons
from popxl_addons.layers.layer_norm import LayerNorm
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from popxl_addons.array_munging import shard, unshard


def test_layer_norm_distributed():
    def local(x_np, dy_np):
        np.random.seed(42)
        ir = popxl.Ir()
        with ir.main_graph:
            x = popxl.variable(x_np, name="x", dtype=popxl.float32)
            dy = popxl.variable(dy_np, name="dy", dtype=popxl.float32)

            facts, graph = LayerNorm().create_graph(x)

            dgraph = addons.autodiff(graph)

            vars = facts.init("ln")
            fwd_call = graph.bind(vars).call_with_info(x)
            y, *_ = fwd_call.outputs

            dx_x, dx_w, dx_b = dgraph.call(dy, args=dgraph.grad_graph_info.inputs_dict(fwd_call))

            y_d2h = addons.host_store(y)
            dx_x_d2h = addons.host_store(dx_x)
            dx_w_d2h = addons.host_store(dx_w)
            dx_b_d2h = addons.host_store(dx_b)

        with popxl.Session(ir, "ipu_hw") as session:
            outputs = session.run()
        return (outputs[y_d2h], outputs[dx_x_d2h], outputs[dx_w_d2h], outputs[dx_b_d2h])

    def distributed(x_np, dy_np):
        np.random.seed(42)

        shards = 4
        ir = popxl.Ir(replication=shards)
        rg = ir.replica_grouping(group_size=shards)

        x_sharded = shard(x_np, axis=1, n=shards)
        dy_sharded = shard(dy_np, axis=1, n=shards)

        with ir.main_graph:
            x = popxl.variable(x_sharded, name="x", dtype=popxl.float32, replica_grouping=rg.transpose())
            dy = popxl.variable(dy_sharded, name="dy", dtype=popxl.float32, replica_grouping=rg.transpose())

            facts, graph = LayerNormDistributed(replica_grouping=rg).create_graph(x)

            dgraph = addons.autodiff(graph)

            vars = facts.init("ln")
            fwd_call = graph.bind(vars).call_with_info(x)
            y, *_ = fwd_call.outputs

            dx_x, dx_w, dx_b = dgraph.call(dy, args=dgraph.grad_graph_info.inputs_dict(fwd_call))

            y_d2h = addons.host_store(y)
            dx_x_d2h = addons.host_store(dx_x)
            dx_w_d2h = addons.host_store(dx_w)
            dx_b_d2h = addons.host_store(dx_b)

        with popxl.Session(ir, "ipu_hw") as session:
            outputs = session.run()

        return (
            unshard(outputs[y_d2h], 1),
            unshard(outputs[dx_x_d2h], 1),
            unshard(outputs[dx_w_d2h], 0),
            unshard(outputs[dx_b_d2h], 0),
        )

    np.random.seed(42)
    x = np.random.random((100, 64)).astype("float32")
    dy = np.random.random((100, 64)).astype("float32")

    outputs_local = local(x, dy)
    outputs_distributed = distributed(x, dy)

    for l, d in zip(outputs_local, outputs_distributed):
        np.testing.assert_allclose(d, l, atol=1e-5, rtol=1e-5)
