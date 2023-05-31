# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import numpy as np
import popxl
from popxl.utils import to_numpy
from popxl import ops, ReplicaGrouping
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict
from popxl_addons.ops.group_quantize_decompress import (
    group_quantize_decompress,
    group_quantize_compress_numpy,
)


class LinearGQ(addons.Module):
    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        replica_grouping: Optional[ReplicaGrouping] = None,
        group_size: int = 0,
        dim: int = -1,
    ):
        """
        Group Quantised Linear layer

        Stores quantised weights as three tensor variables:
            w_compressed: packed integer ids with four 4-bit integers per 16-bit
                          integer element
            w_scale: scaling factor for decompressing unpacked 4-bit integers to
                     float16
            w_bias: bias term for decompressing unpacked 4-bit integer to float16

        Default behaviour is to initialize variables with zeros. Recommended pattern
        is to overwrite variable data before use. This can be done using the
        torch_mapping static method that automatically compresses a float16 weight
        tensor.

        Arguments:
        - out_features (int): output dimension of layer
        - bias (bool): flag for including additive bias term in layer graph (default= True)
        - replica_grouping (ReplicaGrouping): information required for sharding graph variables across multiple IPUs (optional)
        - group_size (int): size of groups within weight matrix to calculate scaling factors for quantization (default=0, not quantised)
        - dim (int): dimension of weight matrix to quantize (default=-1)

        """
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self.replica_grouping = replica_grouping
        self.group_size = group_size
        self.dim = dim

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        in_features = x.shape[-1]
        if self.dim in [1, -1]:
            compressed_dim = self.out_features
            uncompressed_dim = in_features
        elif self.dim == 0:
            uncompressed_dim = self.out_features
            compressed_dim = in_features

        w_compressed = self.add_variable_input(
            "weight_compressed",
            partial(
                np.zeros,
                shape=(
                    uncompressed_dim,
                    compressed_dim // self.group_size,
                    self.group_size // 4,
                ),
            ),
            popxl.uint16,
            replica_grouping=self.replica_grouping,
        )
        w_scale = self.add_variable_input(
            "weight_decompression_scale",
            partial(
                np.zeros,
                shape=(uncompressed_dim, compressed_dim // self.group_size, 1),
            ),
            x.dtype,
            replica_grouping=self.replica_grouping,
        )
        w_bias = self.add_variable_input(
            "weight_decompression_bias",
            partial(
                np.zeros,
                shape=(uncompressed_dim, compressed_dim // self.group_size, 1),
            ),
            x.dtype,
            replica_grouping=self.replica_grouping,
        )

        w = group_quantize_decompress(w_compressed, w_scale, w_bias, self.dim)

        y = x @ w
        if self.bias:
            b = self.add_variable_input(
                "bias",
                partial(np.zeros, y.shape[-1]),
                x.dtype,
                replica_grouping=self.replica_grouping,
            )
            y = y + b
        return y

    @staticmethod
    def torch_mapping(
        variables: NamedTensors,
        nn_layer,
        dtype: popxl.dtype = popxl.float32,
        group_size: int = 64,
        dim: int = -1,
    ) -> WeightsDict:
        """
        Returns a mapping from the layer variables to the corresponding torch nn.Linear parameters.
        """
        state_dict = dict(
            zip(
                (
                    variables.weight_compressed,
                    variables.weight_decompression_scale,
                    variables.weight_decompression_bias,
                ),
                group_quantize_compress_numpy(nn_layer.weight.T.detach().numpy(), group_size, dim=dim),
            )
        )
        if "bias" in variables.keys():
            state_dict[variables.bias] = to_numpy(nn_layer.bias.data, dtype)

        return WeightsDict(state_dict)
