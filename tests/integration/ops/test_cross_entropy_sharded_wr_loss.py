# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

import popxl
import popxl_addons as addons
from popxl_addons.ops.cross_entropy_sharded_wr_loss import cross_entropy_sharded_wr_loss

np.random.seed(42)


def test_cross_entropy_loss_fwd_and_grad():
    batch = 1
    sequence = 17
    vocab = 53

    n_ipus = 4
    ipus = list(range(n_ipus))

    ir = popxl.Ir()
    ir.replication_factor = 1
    main = ir.main_graph

    assert vocab % n_ipus != 0, 'Not testing uneven vocab'

    labels_data = np.random.randint(0, vocab, (batch * sequence))
    logits_data = [
        np.random.rand(batch * sequence, vocab // n_ipus + int(i < vocab % n_ipus)).astype('float32')
        for i in range(n_ipus)
    ]
    loss_grad_data = np.random.rand(batch * sequence)

    logits_torch = torch.tensor(np.concatenate(logits_data, axis=-1), requires_grad=True)
    loss_truth = torch.nn.functional.cross_entropy(logits_torch, torch.tensor(labels_data), reduction='none')

    loss_truth.backward(torch.tensor(loss_grad_data))
    logits_grad_truth = logits_torch.grad

    class CrossEntropyLoss(addons.Module):
        def build(self, logits, labels, ipus):
            return cross_entropy_sharded_wr_loss(logits, labels, ipus=ipus)

    with main:
        # Logits are sharded
        # Labels are sharded copied
        logits = []
        labels = []
        with popxl.ipu(0):
            logits += [popxl.variable(logits_data[0]) + 0]
            labels += [popxl.variable(labels_data.astype('uint32')) + 0]
            loss_grad = popxl.variable(loss_grad_data) + 0
        for i in range(1, 4):
            with popxl.ipu(i):
                logits += [popxl.variable(logits_data[i]) + 0]
                labels += [labels[0].copy_to_ipu(source=0, destination=i)]

        _, graph = CrossEntropyLoss().create_graph(logits, labels, ipus=ipus)
        dgraph = addons.autodiff(graph,
                                 grads_provided=[graph.graph.outputs[0]],
                                 grads_required=graph.graph.inputs[:n_ipus])

        fwd_call = graph.call_with_info(logits, labels)
        loss, *_ = fwd_call.outputs
        y_streams = [addons.host_store(loss)]

        logits_grads = dgraph.call(loss_grad, args=dgraph.grad_graph_info.inputs_dict(fwd_call))

        for i, ipu in enumerate(ipus):
            with popxl.ipu(ipu):
                y_streams += [addons.host_store(logits_grads[i])]

    ir.num_host_transfers = 1
    output = popxl.Session(ir, "ipu_hw").run()
    outputs = [output[d2h] for d2h in y_streams]
    loss_popxl = outputs[0]
    logits_grad_popxl = np.concatenate(outputs[1:], axis=1)

    np.testing.assert_almost_equal(loss_truth.detach().numpy(), loss_popxl, decimal=6)
    np.testing.assert_almost_equal(logits_grad_truth.detach().numpy(), logits_grad_popxl, decimal=6)
