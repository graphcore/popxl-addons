# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from popxl_addons.layers import GroupNorm
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


def test_group_norm_4D():
    w = 16
    h = 16
    num_channels = 6
    bs = 2
    norm_num_groups = 2
    # (N x C x H x W)
    data = np.random.rand(bs, num_channels, h, w).astype(np.float32)

    torch_layer = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=1e-6, affine=True)
    torch_output = torch_layer(torch.Tensor(data)).detach().numpy()
    popxl_layer = GroupNorm(num_groups=norm_num_groups)

    (popxl_out,) = run_module(
        popxl_layer,
        TensorInput(data),
        weights=partial(GroupNorm.torch_mapping, nn_layer=torch_layer, dtype=popxl.float32),
    )
    np.testing.assert_allclose(popxl_out, torch_output, rtol=10e-4)


def test_group_norm_3D():
    w = 16
    h = 16
    num_channels = 6
    bs = 2
    norm_num_groups = 2
    # (N x C x HW)
    data = np.random.rand(bs, num_channels, h * w).astype(np.float32)

    torch_layer = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=1e-6, affine=True)
    torch_output = torch_layer(torch.Tensor(data)).detach().numpy()
    popxl_layer = GroupNorm(num_groups=norm_num_groups)

    (popxl_out,) = run_module(
        popxl_layer,
        TensorInput(data),
        weights=partial(GroupNorm.torch_mapping, nn_layer=torch_layer, dtype=popxl.float32),
    )
    np.testing.assert_allclose(popxl_out, torch_output, rtol=10e-4)


def test_group_norm_2D():
    num_channels = 6
    bs = 2
    norm_num_groups = 2
    # (N x C)
    data = np.random.rand(bs, num_channels).astype(np.float32)

    torch_layer = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=1e-6, affine=True)
    torch_output = torch_layer(torch.Tensor(data)).detach().numpy()
    popxl_layer = GroupNorm(num_groups=norm_num_groups)

    (popxl_out,) = run_module(
        popxl_layer,
        TensorInput(data),
        weights=partial(GroupNorm.torch_mapping, nn_layer=torch_layer, dtype=popxl.float32),
    )
    np.testing.assert_allclose(popxl_out, torch_output, rtol=10e-3)
