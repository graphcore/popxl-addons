# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Optional, Union

import numpy as np

import popxl
from popxl import ops
import popxl_addons as addons

__all__ = ["AdamOptimizerStep"]


class AdamOptimizerStep(addons.Module):
    @popxl.in_sequence()
    def build(self,
              var: popxl.TensorByRef,
              grad: popxl.Tensor,
              *,
              lr: Union[float, popxl.Tensor],
              beta1: Union[float, popxl.Tensor] = 0.9,
              beta2: Union[float, popxl.Tensor] = 0.999,
              eps: Union[float, popxl.Tensor] = 1e-5,
              weight_decay: Union[float, popxl.Tensor] = 1e-2,
              first_order_dtype: popxl.dtype = popxl.float16,
              bias_correction: bool = True,
              loss_scaling: Union[float, popxl.Tensor] = 1,
              global_norm: Optional[popxl.Tensor] = None,
              global_norm_max: Union[None, float, popxl.Tensor] = None):

        scale = 1
        if loss_scaling != 1:
            scale = scale / loss_scaling

        if global_norm is not None:
            assert global_norm_max is not None, "global_norm_max must be specified along with global_norm"
            if not isinstance(global_norm_max, popxl.Tensor):
                global_norm_max = popxl.constant(global_norm_max, popxl.float32)
            scale = (scale * global_norm_max) / ops.maximum(global_norm, global_norm_max)

        if scale != 1:
            grad = ops.cast(grad, popxl.float32) * scale

        if var.meta_shape:
            first_order = self.add_replica_sharded_variable_input("first_order",
                                                                  partial(np.zeros, var.meta_shape),
                                                                  first_order_dtype,
                                                                  by_ref=True)
        else:
            first_order = self.add_variable_input("first_order",
                                                  partial(np.zeros, var.shape),
                                                  first_order_dtype,
                                                  by_ref=True)
        ops.var_updates.accumulate_moving_average_(first_order, grad, f=beta1)

        if var.meta_shape:
            second_order = self.add_replica_sharded_variable_input("second_order",
                                                                   partial(np.zeros, var.meta_shape),
                                                                   popxl.float32,
                                                                   by_ref=True)
        else:
            second_order = self.add_variable_input("second_order",
                                                   partial(np.zeros, var.shape),
                                                   popxl.float32,
                                                   by_ref=True)
        ops.var_updates.accumulate_moving_average_square_(second_order, grad, f=beta2)

        step = None
        if bias_correction:
            step = self.add_variable_input("step", partial(np.zeros, ()), popxl.float32, by_ref=True)

        # Fused operator that performs:
        # 1. Bias correction (if needed)
        #  m = first_order / (1 - beta1 ** step)
        #  v = second_order / (1 - beta2 ** step)
        # 2. Calculate updater
        #   updater = (m / (sqrt(v) + eps)) + weight_decay * var
        updater = ops.var_updates.adam_updater(first_order,
                                               second_order,
                                               weight=var,
                                               weight_decay=weight_decay,
                                               time_step=step,
                                               beta1=beta1,
                                               beta2=beta2,
                                               epsilon=eps)

        # Important to use `scaled_add` operation to ensure correct
        # handling of float16 values and stochastic rounding.
        ops.scaled_add_(var, updater, b=-lr)
