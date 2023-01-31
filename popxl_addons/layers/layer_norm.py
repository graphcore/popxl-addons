# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import numpy as np
import popxl
from popxl import ops
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict


class LayerNorm(addons.Module):
    """
    Apply layer normalisation to a tensor.

    Args:
        x (Tensor): The tensor to be normalized. It must be a 2 dimensional tensor of shape (N, C).
                    All dimensions you want to normalise over must be flattened to the last one.
        eps (float): The small value to use to avoid division by zero
    Returns:
        Tensor: The layer normalised tensor.
    """

    def build(self, x: popxl.Tensor, eps: float = 1e-5) -> popxl.Tensor:
        if len(x.shape) != 2:
            raise ValueError("The input tensor must be of shape N x C. Please flatten all extra dimensions.")

        w = self.add_variable_input("weight", partial(np.ones, x.shape[-1]), x.dtype)
        b = self.add_variable_input("bias", partial(np.zeros, x.shape[-1]), x.dtype)
        return ops.layer_norm(x, w, b, eps=eps)

    @staticmethod
    def torch_mapping(variables: NamedTensors, nn_layer, dtype: popxl.dtype = popxl.float32) -> WeightsDict:
        """
        Returns a mapping from the layer variables to the corresponding torch nn.LayerNorm parameters.
        """
        weight = nn_layer.weight.data
        bias = nn_layer.bias.data
        if len(weight.shape) > 1:
            # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
            # torch supports inputs with shape (N, *) and allows normalisation over multiple dimensions.
            # We achieve the same result by flattening all dimensions to the last dimension.
            weight = weight.reshape(variables.weight.shape)
            bias = bias.reshape(variables.bias.shape)

        state_dict = WeightsDict(
            {
                variables.weight: to_numpy(weight, dtype),
                variables.bias: to_numpy(bias, dtype),
            }
        )
        return state_dict
