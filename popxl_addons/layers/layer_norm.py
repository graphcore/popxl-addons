# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import numpy as np
import popxl
from popxl import ops
import popxl_addons as addons


class LayerNorm(addons.Module):
    """
    Apply layer normalisation to a tensor.

    Args:
        x (Tensor): The tensor to be normalized.
        eps (float): The small value to use to avoid division by zero
    Returns:
        Tensor: The layer normalised tensor.
    """

    def build(self, x: popxl.Tensor, eps: float = 1e-5) -> popxl.Tensor:
        w = self.add_variable_input("weight", partial(np.ones, x.shape[-1]), x.dtype)
        b = self.add_variable_input("bias", partial(np.zeros, x.shape[-1]), x.dtype)
        return ops.layer_norm(x, w, b, eps=eps)
