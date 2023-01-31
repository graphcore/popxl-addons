# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
from math import ceil

import numpy as np
from scipy.stats import truncnorm
import popxl
from popxl.utils import to_numpy
from popxl import ops, ReplicaGrouping
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict


class Linear(addons.Module):
    def __init__(self, out_features: int, bias: bool = True, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self.replica_grouping = replica_grouping

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        w = self.add_variable_input(
            "weight",
            partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=(x.shape[-1], self.out_features)),
            x.dtype,
            replica_grouping=self.replica_grouping,
        )
        y = x @ w
        if self.bias:
            b = self.add_variable_input(
                "bias", partial(np.zeros, y.shape[-1]), x.dtype, replica_grouping=self.replica_grouping
            )
            y = y + b
        return y

    @staticmethod
    def torch_mapping(variables: NamedTensors, nn_layer, dtype: popxl.dtype = popxl.float32) -> WeightsDict:
        """
        Returns a mapping from the layer variables to the corresponding torch nn.LayerNorm parameters.
        """
        state_dict = {
            variables.weight: to_numpy(nn_layer.weight.data, dtype).T,
        }
        if "bias" in variables.keys():
            state_dict[variables.bias] = to_numpy(nn_layer.bias.data, dtype)

        return WeightsDict(state_dict)
