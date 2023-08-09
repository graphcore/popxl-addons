# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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
from popxl_addons.ops.fp8_mse_scale import fp8_mse_scale
from popxl_addons.ops.fp8_amax_scale import fp8_amax_scale


class LinearFP8(addons.Module):
    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        scale_metric: str = "amax",
        scale: int = 0,
        replica_grouping: Optional[ReplicaGrouping] = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self.replica_grouping = replica_grouping
        self.scale_metric = scale_metric
        self.scale = scale
        self.forward_fp8_type = popxl.float8_143
        self.grad_fp8_type = popxl.float8_152

        # Let the user pass a fp8_scale function
        if self.scale_metric == "amax":
            self.fp8_scale = lambda x, y: fp8_amax_scale(x, y)
        elif self.scale_metric == "mse":
            self.fp8_scale = lambda x, y: fp8_mse_scale(x, y)
        elif self.scale_metric == "manual_scale":
            self.fp8_scale = lambda x, y: popxl.constant(self.scale)
        else:
            raise ValueError(
                f"The scale_metric option {self.scale_metric} is not available. "
                "Please provide one of the following options: amax, mse or manual_scale."
            )

    def build(self, x: popxl.Tensor):
        w = self.add_variable_input(
            "weight",
            partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=(x.shape[-1], self.out_features)),
            x.dtype,
            replica_grouping=self.replica_grouping,
        )

        # Compute w_log2scale to cast weights from F16 to FP8 following scale_metric
        w_log2scale = self.fp8_scale(w, popxl.float8_143)

        # Cast weights from FP16 to FP8 using w_log2scale
        w_fp8 = ops.pow2scale_cast_to_fp8(w, data_type=self.forward_fp8_type, log2_scale=w_log2scale)

        # Compute x_log2scale to cast activations from F16 to FP8 following scale_metric
        x_log2scale = self.fp8_scale(x, popxl.float8_143)

        # Cast activations from FP16 to FP8 using x_log2scale
        x_fp8 = ops.pow2scale_cast_to_fp8(x, data_type=self.forward_fp8_type, log2_scale=x_log2scale)

        # Save log2scales, weights and activations to be used in bwd pass
        self.aux["w_log2scale"] = w_log2scale
        self.aux["w_fp8"] = w_fp8
        self.aux["x_log2scale"] = x_log2scale
        self.aux["x_fp8"] = x_fp8

        # Do the FP8 matmul, which accumulates in F16
        y = ops.matmul_pow2scaled(x_fp8, w_fp8, log2_scale=-w_log2scale - x_log2scale)

        # Add a cast in case bias are fp32, since the output of matmul_pow2scaled is fp16
        if x.dtype == popxl.float32:
            y = ops.cast(y, popxl.float32)

        if self.bias:
            b = self.add_variable_input(
                "bias", partial(np.zeros, y.shape[-1]), x.dtype, replica_grouping=self.replica_grouping
            )
            y = y + b
        return y

    def build_grad(self, dLdy: popxl.Tensor):
        """
        Build the backward pass graph using the FP8 functions.
        """
        # Compute optimal scale for incoming F16 dLdy following scale_metric
        # The scale is substracted -3 to leave some margin and prevent overflow in the matmul_pow2scaled
        dLdy_log2scale = self.fp8_scale(dLdy, popxl.float8_152) - popxl.constant(3)
        dLdy_log2scale = ops.clip(dLdy_log2scale, -31, 31)
        # Cast activation grad dLdy from FP16 to FP8 using dLdy_log2scale
        dLdy_fp8 = ops.pow2scale_cast_to_fp8(dLdy, data_type=self.grad_fp8_type, log2_scale=dLdy_log2scale)

        # Compute dLdx
        dLdx = ops.matmul_pow2scaled(
            dLdy_fp8, self.aux["w_fp8"].T, log2_scale=-self.aux["w_log2scale"] - dLdy_log2scale
        )

        # Compute dLdw
        dLdw = ops.matmul_pow2scaled(
            dLdy_fp8.T, self.aux["x_fp8"], log2_scale=-self.aux["x_log2scale"] - dLdy_log2scale
        ).T

        # Associate grads with weights
        self.var_grad["weight"] = dLdw

        # Associate grads with bias
        if self.bias:
            dLdb = ops.sum(dLdy, 0)
            self.var_grad["bias"] = dLdb

        return dLdx

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
