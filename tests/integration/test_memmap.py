# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from tempfile import TemporaryDirectory
import os

import popxl
from popxl import ops
from popxl_addons.remote import load_remote, named_variable_buffers, store_remote
from popxl_addons.ops.streams import host_load, host_store
import pytest
import numpy as np
from popxl_addons.layers import Linear
from popxl_addons.module import Module
from popxl_addons.task_session import TaskSession


class TestModel(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(30, bias=False)

    def build(self, x: popxl.Tensor):
        x = self.l1(x)
        return x


def test_init_memmap():
    with TemporaryDirectory() as memmap_dir:
        ir = popxl.Ir()

        with ir.main_graph, popxl.in_sequence():
            x_np, x_h2d, x = host_load(np.random.normal(0, 1, size=100), popxl.float32, "x")

            facts, g = TestModel().create_graph(x)

            variables = facts.init("test_model", memmap_dir=memmap_dir)

            (y,) = g.bind(variables).call(x)
            outputs = [host_store(y)]

            delta = np.random.normal(0, 1, size=variables.l1.weight.shape)
            dw = popxl.constant(delta, popxl.float32)
            ops.var_updates.accumulate_(variables.l1.weight, dw)

        session = TaskSession(inputs=[x_h2d], outputs=outputs, state=variables, ir=ir, device_desc="ipu_hw")

        weight = session.get_tensor_data(session.state.l1.weight)
        assert isinstance(weight, np.memmap)
        assert weight is session.state.l1.weight._memmap_arr
        before = weight.copy()
        with session:
            y = session.run({x_h2d: x_np})[session.outputs[0]]
        np.testing.assert_almost_equal(x_np @ before, y)
        np.testing.assert_almost_equal(before + delta, session.get_tensor_data(session.state.l1.weight), 6)


def test_reuse_init_memmap():
    with TemporaryDirectory() as memmap_dir:
        ir0 = popxl.Ir()
        with ir0.main_graph:
            facts0, _ = TestModel().create_graph(popxl.TensorSpec((100,), popxl.float32))
            variables0 = facts0.init("test_model", memmap_dir=memmap_dir)
            # Flush values
            variables0.l1.weight._memmap_arr.base.flush()

        ir = popxl.Ir()
        with ir.main_graph, popxl.in_sequence():
            x_np, x_h2d, x = host_load(np.random.normal(0, 1, size=100), popxl.float32, "x")

            facts, g = TestModel().create_graph(x)

            variables = facts.init("test_model", memmap_dir=memmap_dir)
            (y,) = g.bind(variables).call(x)
            outputs = [host_store(y)]

        session = TaskSession(inputs=[x_h2d], outputs=outputs, state=variables, ir=ir, device_desc="ipu_hw")

        weight = session.get_tensor_data(session.state.l1.weight)
        assert isinstance(weight, np.memmap)
        assert weight is session.state.l1.weight._memmap_arr
        # Check they match
        np.testing.assert_allclose(variables0.l1.weight._memmap_arr, weight)


def test_init_remote_memmap():
    with TemporaryDirectory() as memmap_dir:
        ir = popxl.Ir()

        with ir.main_graph, popxl.in_sequence():
            x_np, x_h2d, x = host_load(np.random.normal(0, 1, size=100), popxl.float32, "x")

            facts, g = TestModel().create_graph(x)

            buffers = named_variable_buffers(facts)

            variables = facts.init_remote(buffers, 0, "test_model", memmap_dir=memmap_dir)

            loaded = load_remote(buffers)
            (y,) = g.bind(loaded).call(x)
            outputs = [host_store(y)]

            delta = np.random.normal(0, 1, size=loaded.l1.weight.shape)
            dw = popxl.constant(delta, popxl.float32)
            ops.var_updates.accumulate_(loaded.l1.weight, dw)

            store_remote(buffers, loaded)

        session = TaskSession(inputs=[x_h2d], outputs=outputs, state=variables, ir=ir, device_desc="ipu_hw")

        weight = session.get_tensor_data(session.state.l1.weight)
        assert isinstance(weight, np.memmap)
        assert weight is session.state.l1.weight._memmap_arr
        before = weight.copy()
        with session:
            y = session.run({x_h2d: x_np})[session.outputs[0]]
        np.testing.assert_almost_equal(x_np @ before, y)
        np.testing.assert_almost_equal(before + delta, session.get_tensor_data(session.state.l1.weight), 6)


if __name__ == "__main__":
    test_init_memmap()
    test_reuse_init_memmap()
    test_init_remote_memmap()
