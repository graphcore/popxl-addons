# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Optional
import numpy as np

import popxl
from popxl import ReplicaGrouping
from popxl import ops
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict


class GroupNorm(addons.Module):
    """
    Apply group normalisation to a tensor.
    """

    def __init__(
        self,
        num_groups: int,
        replica_grouping: Optional[ReplicaGrouping] = None,
        eps: float = 1e-5,
        cache: bool = False,
    ):
        """
        Args:
            num_groups (int):  Number of groups to separate the channels into.
            replica_grouping (ReplicaGrouping, optional): ReplicaGrouping for the variables
            eps (float): The small value to use to avoid division by zero.
            cache (bool): Re-use graphs where possible when calling `create_graph`. Defaults to False.
        """
        super().__init__(cache)
        self.num_groups = num_groups
        self.replica_grouping = replica_grouping
        self.eps = eps

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        """
        Args:
            x (Tensor): The tensor to be normalized, with shape (N x C x *)
                        The channel dimension is the second. Parameters have shape (C,).
        Returns:
            Tensor: The group normalised tensor, with shape (N x C x *).
        """
        channel_axis = 1
        self.weight = self.add_variable_input(
            "weight", partial(np.ones, x.shape[channel_axis]), x.dtype, replica_grouping=self.replica_grouping
        )
        self.bias = self.add_variable_input(
            "bias", partial(np.zeros, x.shape[channel_axis]), x.dtype, replica_grouping=self.replica_grouping
        )
        return ops.group_norm(x, self.weight, self.bias, num_groups=self.num_groups, eps=self.eps)

    @staticmethod
    def torch_mapping(variables: NamedTensors, nn_layer, dtype: popxl.dtype = popxl.float32) -> WeightsDict:
        """
        Returns a mapping from the layer variables to the corresponding torch nn.LayerNorm parameters.
        """
        state_dict = WeightsDict(
            {
                variables.weight: to_numpy(nn_layer.weight.data, dtype),
                variables.bias: to_numpy(nn_layer.bias.data, dtype),
            }
        )
        return state_dict
