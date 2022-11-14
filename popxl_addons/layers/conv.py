# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
from math import ceil

import numpy as np
from scipy.stats import truncnorm
import popxl
from popxl import ops, ReplicaGrouping
from typing import Optional, Tuple, List, Union
from popxl.ops.conv import PadType
import popxl_addons as addons


class Conv2D(addons.Module):
    """
    Apply 2D convolution.
    The input tensor has shape (N, C, H, W), where:
        - N is the batch size,
        - C is the number of input channels,
        - H and W are the height and width

    See also `popxl.ops.conv`

    Args:
        out_channels (int):
            number of output channels
        kernel_size (int or Tuple[int]):
            dimensions of the convolution kernel
        strides (Tuple[int], optional):
            Strides of the convolution. Default: (1,1)
        paddings (Tuple[int] or str, optional):
            Can be a tuple specifying explicit padding to be added to all four sides of the input,
            or a string identifying a padding type: "same_upper", "same_lower" or "valid".
        dilations (Tuple[int], optional):
            Spacing between kernel elements. Default: (1,1)
        groups (int, optional):
            Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional):
            If ``True``, adds a learnable bias to the output. Default: ``True``
        available_memory_proportions (List[float], optional):
            The available memory proportions per conv, each [0, 1).
            It is the percentage of memory to reserve for computation.
        partials_types (List[str], optional):
            The partials type per convolution, choose between half and float.
        enable_conv_dithering (List[int], optional):
            Enable convolution dithering per convolution.
            If true, then convolutions with different parameters will be laid out from different tiles
            in an effort to improve tile balance in models.
        replica_grouping (ReplicaGrouping, optional):
            replica grouping for the variables. 
    Returns:
        Tensor: The result of the convolution
    """

    def __init__(self,
                 out_channels: int,
                 kernel_size: Union[Tuple[int], int],
                 strides: Optional[Tuple[int]] = (1, 1),
                 paddings: Optional[Union[str, Tuple[int]]] = (0, 0, 0, 0),
                 dilations: Optional[Tuple[int]] = (1, 1),
                 groups: Optional[int] = 1,
                 bias: bool = True,
                 available_memory_proportions: Optional[List[float]] = None,
                 partials_types: Optional[List[str]] = None,
                 enable_conv_dithering: Optional[List[int]] = None,
                 replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()

        self.out_channels = out_channels

        if isinstance(kernel_size, tuple):
            self.k_h, self.k_w = kernel_size
        else:
            self.k_h, self.k_w = kernel_size, kernel_size

        self.strides = strides
        self.dilations = dilations
        self.groups = groups

        if isinstance(paddings, str):
            self.pad_type = paddings
            self.paddings = (0, 0, 0, 0)
        else:
            # explicit padding
            self.pad_type = "not_set"
            self.paddings = paddings

        self.replica_grouping = replica_grouping
        self.bias = bias

        self.available_memory_proportions = available_memory_proportions
        self.partial_types = partials_types
        self.enable_conv_dithering = enable_conv_dithering

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        assert len(x.shape) == 4
        n, in_channel, h, w = x.shape
        kernel_shape = (self.out_channels, int(in_channel / self.groups), self.k_h, self.k_w)
        self.weight = self.add_variable_input("weight",
                                              partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=kernel_shape),
                                              x.dtype,
                                              replica_grouping=self.replica_grouping)

        x = ops.conv(t=x,
                     weight=self.weight,
                     stride=self.strides,
                     padding=self.paddings,
                     dilation=self.dilations,
                     groups=self.groups,
                     pad_type=self.pad_type,
                     available_memory_proportions=self.available_memory_proportions,
                     partials_types=self.partial_types,
                     enable_conv_dithering=self.enable_conv_dithering)

        if self.bias:
            bias_shape = [1] * len(x.shape)
            bias_shape[1] = self.out_channels
            self.bias = self.add_variable_input("bias",
                                                partial(np.zeros, bias_shape),
                                                x.dtype,
                                                replica_grouping=self.replica_grouping)
            x = x + self.bias
        return x
