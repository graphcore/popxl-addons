# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import numpy as np
import popxl
from popxl import ops, ReplicaGrouping
from typing import Optional

import popxl_addons as addons
from popxl_addons.ops.layer_norm_distributed import layer_norm_distributed


class LayerNormDistributed(addons.Module):
    """
    Apply layer normalisation to a tensor that is sharded across it's hidden dimension within the group `replica_grouping`.

    Args:
        x (Tensor): The tensor to be normalized.
        eps (float): The small value to use to avoid division by zero
    Returns:
        Tensor: The layer normalised tensor.
    """

    def __init__(self, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.replica_grouping = replica_grouping

    def build(self, x: popxl.Tensor, eps: float = 1e-5) -> popxl.Tensor:
        w = self.add_variable_input("weight",
                                    partial(np.ones, x.shape[-1]),
                                    x.dtype,
                                    replica_grouping=self.replica_grouping.transpose())
        b = self.add_variable_input("bias",
                                    partial(np.zeros, x.shape[-1]),
                                    x.dtype,
                                    replica_grouping=self.replica_grouping.transpose())
        return layer_norm_distributed(x, w, b, epsilon=eps, group=self.replica_grouping)
