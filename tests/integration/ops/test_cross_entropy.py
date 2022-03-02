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

    def popart():
        logits, target = inputs()
        ir = popxl.Ir()
        with ir.main_graph:
            logits = popxl.variable(logits.detach().numpy().astype(np.float32))
            target = popxl.variable(target.detach().numpy().astype(np.uint32))
            loss, dlogits = addons.ops.cross_entropy_with_grad(logits, target, 64)
            outs = (
                addons.host_store(loss),
                addons.host_store(dlogits),
            )
        return addons.Runner(ir, outs).run()

    for _t, _p in zip(pytorch(), popart()):
        np.testing.assert_almost_equal(_t, _p, 5)
