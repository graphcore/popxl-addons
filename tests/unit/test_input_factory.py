# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from popart_ir_extensions.input_factory import InputFactory
import pytest


def test_bad_callable():

    bad_callable = lambda: sum(1)

    with pytest.raises(ValueError):
        input_factory = InputFactory(data_iter=bad_callable)
