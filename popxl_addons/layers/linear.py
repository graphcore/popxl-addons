# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
from math import ceil

import numpy as np
from scipy.stats import truncnorm
import popxl
from popxl import ops, ReplicaGrouping
from typing import Optional

import popxl_addons as addons


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
