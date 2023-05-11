# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from popxl_addons.layers.linear_fp8 import LinearFP8
from popxl_addons.testing_utils import run_module, TensorInput
import popxl
import numpy as np
import pytest
import torch
from functools import partial

np.random.seed(42)


@pytest.mark.parametrize("scale_metric", ["amax", "mse", "no_scale"])
def test_linear_fp8(scale_metric):
    in_features = 10
    out_features = 10
    bs = 2
    bias = True

    np.random.seed(0)
    torch.manual_seed(0)
    input = np.random.rand(bs, in_features).astype(np.float16)

    torch_layer = torch.nn.Linear(in_features, out_features, bias)
    torch_output = torch_layer(torch.Tensor(input)).detach().numpy()
    popxl_fp8_layer = LinearFP8(out_features, bias, scale_metric=scale_metric)

    (popxl_fp8_out,) = run_module(
        popxl_fp8_layer,
        TensorInput(input),
        weights=partial(LinearFP8.torch_mapping, nn_layer=torch_layer, dtype=popxl.float16),
    )
    print(f"torch_output = {torch_output}")
    print(f"popxl_fp8_out = {popxl_fp8_out}")
    np.testing.assert_allclose(popxl_fp8_out, torch_output, atol=0.1)
