# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from popxl_addons.layers import LayerNorm
from torch import nn
from popxl_addons.testing_utils import run_module, TensorInput
from popxl_addons import NamedTensors
import popxl
from popxl.utils import to_numpy
import numpy as np
import torch
from functools import partial

np.random.seed(42)
torch.manual_seed(42)


def test_layer_norm_4D():
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    # torch supports inputs with shape (N, *) and allows normalisation over multiple dimensions.
    # We achieve the same result by flattening all dimensions to the last dimension.

    c = 6
    bs = 2
    h = 8
    w = 8
    # (N x C x H x W)
    data = np.random.rand(bs, c, h, w).astype(np.float32)

    torch_layer = nn.LayerNorm([c, h, w])
    torch_output = torch_layer(torch.Tensor(data)).detach().numpy()
    popxl_layer = LayerNorm()

    (popxl_out,) = run_module(
        popxl_layer,
        TensorInput(data.reshape(bs, c * h * w)),
        weights=partial(LayerNorm.torch_mapping, nn_layer=torch_layer, dtype=popxl.float32),
    )
    np.testing.assert_allclose(popxl_out, torch_output.reshape(popxl_out.shape), rtol=10e-4)


def test_layer_norm_2D():
    bs = 2
    dim = 6
    data = np.random.rand(bs, dim).astype(np.float32)

    torch_layer = nn.LayerNorm(dim)
    torch_output = torch_layer(torch.Tensor(data)).detach().numpy()
    popxl_layer = LayerNorm()

    (popxl_out,) = run_module(
        popxl_layer,
        TensorInput(data),
        weights=partial(LayerNorm.torch_mapping, nn_layer=torch_layer, dtype=popxl.float32),
    )
    np.testing.assert_allclose(popxl_out, torch_output, rtol=10e-4)
