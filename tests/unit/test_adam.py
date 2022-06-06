# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
import torch.optim as optim
import popxl

from popxl_addons.optimizers.adam import AdamOptimizerStep


def test_adam_correctness():
    np.random.seed(42)
    torch.manual_seed(42)
    # pytorch
    var_data = np.random.normal(0, 1, (2, 2)).astype(np.float32)
    grad_data = np.random.normal(0, 1, (2, 2)).astype(np.float32)

    torch_var = torch.Tensor(var_data.copy())
    torch_var.grad = torch.Tensor(grad_data.copy())
    torch_adam = optim.AdamW([torch_var], lr=2, eps=1e-5, weight_decay=0.0)
    torch_adam.step()

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = 1

    main = ir.main_graph
    adam = AdamOptimizerStep()

    with main:
        var = popxl.variable(var_data)
        grad = popxl.variable(grad_data)
        state, optim_step = adam.create_graph(var, grad, lr=2, eps=1e-5, weight_decay=0.0)
        optim_step.bind(state.init()).call(var, grad)

    ir.num_host_transfers = 1
    with popxl.Session(ir, "ipu_hw") as session:
        session.run()
        result = session.get_tensor_data(var)

    np.testing.assert_almost_equal(torch_var, result, 4)
