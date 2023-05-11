# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from popxl_addons.layers.linear_gq import LinearGQ
from popxl_addons.testing_utils import run_module, TensorInput
from popxl_addons.ops.group_quantize_decompress import (
    group_quantize_decompress_numpy,
    group_quantize_compress_numpy,
)
import popxl
import numpy as np
import pytest
import torch
from functools import partial

np.random.seed(42)


class LinearGQTorch(torch.nn.Linear):
    def __init__(self, group_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(
            torch.from_numpy(
                group_quantize_decompress_numpy(
                    *group_quantize_compress_numpy(self.weight.T.detach().numpy(), group_size)
                )
            )
            .T.to(self.bias.dtype)
            .contiguous()
        )


def test_linear_gq():
    in_features = 128
    out_features = 256
    group_size = 16
    batch_size = 4
    bias = True

    np.random.seed(0)
    torch.manual_seed(0)
    input = np.random.rand(batch_size, in_features).astype(np.float16)

    torch_layer = LinearGQTorch(
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
    )
    torch_output = torch_layer(torch.from_numpy(input).to(torch.float)).detach().numpy().astype(np.float16)

    popxl_gq_layer = LinearGQ(out_features, bias, group_size=group_size)

    (popxl_gq_out,) = run_module(
        popxl_gq_layer,
        TensorInput(input),
        weights=partial(
            LinearGQ.torch_mapping,
            nn_layer=torch_layer,
            dtype=popxl.float16,
            group_size=group_size,
        ),
    )

    print(f"torch_output = {torch_output}")
    print(f"popxl_gq_out = {popxl_gq_out}")
    np.testing.assert_allclose(popxl_gq_out, torch_output, atol=0.1)


if __name__ == "__main__":
    test_linear_gq()
