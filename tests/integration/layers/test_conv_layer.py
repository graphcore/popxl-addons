# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from popxl_addons.layers import Conv2D
from torch import nn
from functools import partial
from popxl_addons.testing_utils import run_module, TensorInput
from popxl_addons import NamedTensors
import popxl
from popxl.utils import to_numpy
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


def test_conv2D_layer():
    kernel_size = 2
    in_channels = 3
    out_channels = 3
    height = 16
    width = 16
    bs = 2
    strides = (1, 2)

    images = np.random.rand(bs, in_channels, height, width).astype(np.float32)

    # test no paddng
    torch_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides)
    torch_output = torch_layer(torch.Tensor(images)).detach().numpy()
    popxl_layer = Conv2D(out_channels, kernel_size, strides=strides)

    (popxl_out,) = run_module(
        popxl_layer,
        TensorInput(images),
        weights=partial(Conv2D.torch_mapping, nn_layer=torch_layer, dtype=popxl.float32),
    )
    np.testing.assert_allclose(popxl_out, torch_output, rtol=10e-4)


def test_conv2D_padding():
    kernel_size = 2
    in_channels = 3
    out_channels = 3
    height = 16
    width = 16
    bs = 2

    images = np.random.rand(bs, in_channels, height, width).astype(np.float32)

    # test same paddng
    print("padding = same")
    torch_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same")
    torch_output = torch_layer(torch.Tensor(images)).detach().numpy()
    popxl_layer = Conv2D(out_channels, kernel_size, paddings="same_upper")

    (popxl_out,) = run_module(
        popxl_layer,
        TensorInput(images),
        weights=partial(Conv2D.torch_mapping, nn_layer=torch_layer, dtype=popxl.float32),
    )
    np.testing.assert_allclose(popxl_out, torch_output, rtol=10e-4)

    # test explicit paddng
    print("explicit padding")
    torch_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(4, 4))
    torch_output = torch_layer(torch.Tensor(images)).detach().numpy()
    popxl_layer = Conv2D(out_channels, kernel_size, paddings=(4, 4, 4, 4))

    (popxl_out,) = run_module(
        popxl_layer,
        TensorInput(images),
        weights=partial(Conv2D.torch_mapping, nn_layer=torch_layer, dtype=popxl.float32),
    )
    np.testing.assert_allclose(popxl_out, torch_output, rtol=10e-4)
