# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import numpy as np
import popxl

import popxl_addons as addons
from popxl_addons.ops.fp8_mse_scale import fp8_mse_scale
from popxl_addons.ops.fp8_amax_scale import fp8_amax_scale
from popxl_addons.ops.fp8_utils import float8_143_def, float8_152_def
from popxl.fp8_utils import host_fp8_mse_scale


@pytest.mark.parametrize("format", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("from_dtype", [np.float16, np.float32])
def test_fp8_scale_mse(format, from_dtype):
    def inputs():
        np.random.seed(0)
        return np.random.uniform(-(2**5), -(2**6), 10).astype(from_dtype)

    def python():
        src = inputs()
        return host_fp8_mse_scale(src, format)

    def popxl_():
        src = inputs()
        ir = popxl.Ir()
        with ir.main_graph:
            src = popxl.variable(src)
            bias = fp8_mse_scale(src, format)
            bias_d2h = addons.host_store(bias)

        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run()
        return (outputs[bias_d2h],)

    python_scale = python()[1]
    ipu_scale = popxl_()[0]
    print(f"python scale on host: {python_scale}")
    print(f"popxl scale on IPU: {ipu_scale}")
    assert python_scale == ipu_scale


@pytest.mark.parametrize("format", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("from_dtype", [np.float16, np.float32])
def test_fp8_scale_amax(format, from_dtype):
    def inputs():
        np.random.seed(0)
        return np.random.uniform(-(2**5), -(2**6), 10).astype(from_dtype)

    def python():
        src = inputs()
        if format is popxl.float8_143:
            dest_ftype = float8_143_def
        if format is popxl.float8_152:
            dest_ftype = float8_152_def
        log2_max = np.log(dest_ftype.max_value / max(abs(src))) / np.log(2)
        print(f"np log2 max is {log2_max}")
        return np.floor(log2_max)

    def popxl_():
        src = inputs()
        ir = popxl.Ir()
        with ir.main_graph:
            src = popxl.variable(src)
            bias = fp8_amax_scale(src, format)
            bias_d2h = addons.host_store(bias)

        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run()
        return (outputs[bias_d2h],)

    python_scale = python()
    ipu_scale = popxl_()
    print(f"python scale on host: {python_scale}")
    print(f"popxl scale on IPU: {ipu_scale}")
    assert python_scale == ipu_scale
    assert ipu_scale[0] >= -31 and ipu_scale[0] <= 31


test_fp8_scale_amax(popxl.float8_152, np.float32)
