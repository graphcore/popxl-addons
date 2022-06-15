# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
import pytest
import popxl
import popxl_addons as addons
from popxl import ops
from popxl_addons.ops.cross_entropy_sharded_loss import cross_entropy_sharded_loss


@pytest.mark.parametrize('dtype_size', [16, 32])
def test_cross_entropy_loss_fwd_and_grad(dtype_size):
    np.random.seed(42)

    if dtype_size == 32:
        float_dtype, np_float_dtype = popxl.float32, np.float32
        testing_decimals = 6
    else:
        float_dtype, np_float_dtype = popxl.float16, np.float16
        testing_decimals = 2

    batch = 1
    sequence = 17
    vocab = 20

    n_shards = 4

    ir = popxl.Ir()
    ir.replication_factor = n_shards
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)
    assert vocab % n_shards == 0

    logits_data = [np.random.rand(batch * sequence, vocab // n_shards).astype('float32') for i in range(n_shards)]
    labels_data = np.random.randint(0, vocab, (batch * sequence))
    labels_data[-1] = 0  # Make sure at least one element that's offsetted has value 0

    offsets_data = [i * (vocab // n_shards) for i in range(n_shards)]

    logits_torch = torch.tensor(np.concatenate(logits_data, axis=-1), requires_grad=True)
    loss_truth = torch.nn.functional.cross_entropy(logits_torch,
                                                   torch.tensor(labels_data),
                                                   ignore_index=0,
                                                   reduction='mean')

    loss_truth.backward(loss_truth.clone().detach())
    logits_grad_truth = logits_torch.grad

    class CrossEntropyLoss(addons.Module):
        def build(self, logits, labels, ignore_index):
            return cross_entropy_sharded_loss(logits,
                                              labels,
                                              reduction='mean',
                                              replica_grouping=replica_grouping.transpose(),
                                              ignore_index=ignore_index)

    with ir.main_graph:
        _, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(logits_data[0].astype(np_float_dtype), float_dtype, name="logits"),
            addons.host_load(labels_data, popxl.int32, name="labels"),
        ])
        logits, labels = inputs_tensors

        offset = popxl.variable(offsets_data, popxl.int32, name='offsets', replica_grouping=replica_grouping)
        ignore_index = -1 * offset

        _, graph = CrossEntropyLoss().create_graph(logits, labels, ignore_index)
        dgraph = addons.autodiff(graph, grads_provided=graph.graph.outputs, grads_required=graph.graph.inputs[:1])

        fwd_call = graph.call_with_info(logits, labels, ignore_index)
        loss, *_ = fwd_call.outputs
        loss_stream = addons.host_store(loss)

        logits_grad, *_ = dgraph.call(loss, args=dgraph.grad_graph_info.inputs_dict(fwd_call))
        logits_grad_stream = addons.host_store(logits_grad)

    # Adjust labels data
    labels_data_adjusted = [labels_data - e for e in offsets_data]

    inputs_data = [
        np.concatenate(logits_data, axis=0).reshape(
            (n_shards, batch * sequence, vocab // n_shards)).astype(np_float_dtype),
        np.concatenate(labels_data_adjusted, axis=0).reshape((n_shards, batch * sequence)),
    ]
    inputs = dict(zip(inputs_host_steam, inputs_data))
    with popxl.Session(ir, "ipu_hw") as session:
        output = session.run(inputs)

    # loss
    loss_popxl = output[loss_stream]

    # All replicas should have the same output
    for i in range(1, n_shards):
        np.testing.assert_equal(loss_popxl[0], loss_popxl[i])
    loss_popxl = loss_popxl[0]

    np.testing.assert_almost_equal(loss_truth.detach().numpy(), loss_popxl, decimal=testing_decimals)

    # logits_grad
    logits_grad_popxl = output[logits_grad_stream]
    logits_grad_popxl = np.concatenate(logits_grad_popxl, axis=-1)
    np.testing.assert_almost_equal(logits_grad_truth.detach().numpy(), logits_grad_popxl, decimal=testing_decimals)
