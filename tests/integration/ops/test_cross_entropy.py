# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np

import popart.ir as pir

import popart_ir_extensions as pir_ext


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
        ir = pir.Ir()
        with ir.main_graph:
            logits = pir.variable(logits.detach().numpy().astype(np.float32))
            target = pir.variable(target.detach().numpy().astype(np.uint32))
            loss, dlogits = pir_ext.ops.cross_entropy_with_grad(logits, target, 64)
            outs = (
                pir_ext.host_store(loss),
                pir_ext.host_store(dlogits),
            )
        return pir_ext.Runner(ir, outs).run()

    for _t, _p in zip(pytorch(), popart()):
        np.testing.assert_almost_equal(_t, _p, 5)
