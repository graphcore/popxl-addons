# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np

import popxl

import popxl_addons as addons


def test_cross_entropy_with_grad():
    def inputs():
        torch.manual_seed(42)
        return torch.rand((8, 100), requires_grad=True), torch.randint(0, 100, (8, ))

    def pytorch():
        logits, target = inputs()
        loss = F.cross_entropy(logits, target)
        sloss = loss * 64
        sloss.backward()
        return loss.detach().numpy(), logits.grad.detach().numpy()

    def popxl_():
        logits, target = inputs()
        ir = popxl.Ir()
        with ir.main_graph:
            logits = popxl.variable(logits.detach().numpy().astype(np.float32))
            target = popxl.variable(target.detach().numpy().astype(np.uint32))
            loss, dlogits = addons.ops.cross_entropy_with_grad(logits, target, 64)
            loss_d2h = addons.host_store(loss)
            dlogits_d2h = addons.host_store(dlogits)

        ir.num_host_transfers = 1
        session = popxl.Session(ir, "ipu_hw")
        outputs = session.run()
        session.device.detach()
        return (outputs[loss_d2h], outputs[dlogits_d2h])

    for _t, _p in zip(pytorch(), popxl_()):
        np.testing.assert_almost_equal(_t, _p, 5)
