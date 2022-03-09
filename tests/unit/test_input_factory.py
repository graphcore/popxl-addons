# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
import pytest
import numpy as np

import popxl
import popxl_addons as addons
from popxl_addons.input_factory import InputFactory


def test_bad_callable():

    bad_callable = lambda: sum(1)

    with pytest.raises(ValueError):
        input_factory = InputFactory(data_iter=bad_callable)


class Linear(addons.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def build(self, x: popxl.Tensor):
        w = self.add_input_tensor("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
        b = self.add_input_tensor("b", partial(np.zeros, (self.features, )), popxl.float32)
        return (x @ w) + b


def test_init_zero():
    ir = popxl.Ir()
    with ir.main_graph:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 4)), popxl.float32)

        args, _ = Linear(10).create_graph(x)
        zero_args = args.init_zero()

        assert set(zero_args.to_dict().keys()) == set(args.to_dict().keys())


def test_init_undef():
    ir = popxl.Ir()
    with ir.main_graph:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 4)), popxl.float32)

        args, _ = Linear(10).create_graph(x)
        undef_args = args.init_undef()

        assert set(undef_args.to_dict().keys()) == set(args.to_dict().keys())
