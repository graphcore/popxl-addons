# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
import popxl
from popxl.utils import to_numpy
from popxl_addons.layers import Linear
from popxl_addons import NamedTensors
from popxl_addons.testing_utils import TensorInput, run_module
from popxl_addons import Module


def test_run_module():
    input_np = np.random.rand(4)
    input_torch = torch.Tensor(input_np)

    torch_linear = torch.nn.Linear(4, 2)
    output_torch = torch_linear(input_torch)
    output_torch = output_torch.detach().numpy()

    input_t = TensorInput(input_np, popxl.float32)
    popxl_linear = Linear(2)

    def weight_mapping(variables: NamedTensors):
        return {variables.weight: to_numpy(torch_linear.weight.data.T), variables.bias: to_numpy(torch_linear.bias)}

    (output_popxl,) = run_module(popxl_linear, input_t, weights=weight_mapping)

    np.testing.assert_allclose(output_popxl, output_torch.reshape(output_popxl.shape), rtol=10e-3)


def test_run_module_in_place():
    input_np = np.random.rand(4)
    add_np = np.ones(4)

    input_torch = torch.Tensor(input_np)
    input_torch += torch.Tensor(add_np)

    input_t = TensorInput(input_np, popxl.float32)
    add_t = TensorInput(add_np, popxl.float32)

    class InPlaceAdd(Module):
        @popxl.in_sequence(True)
        def build(self, x: popxl.TensorByRef, a: popxl.Tensor):
            popxl.ops.add_(x, a)
            return x

    popxl_layer = InPlaceAdd()

    (output_popxl,) = run_module(popxl_layer, input_t, add_t)

    np.testing.assert_allclose(output_popxl, input_torch, rtol=10e-3)


if __name__ == "__main__":
    test_run_module()
    test_run_module_in_place()
