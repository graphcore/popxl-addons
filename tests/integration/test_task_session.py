# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from tempfile import TemporaryDirectory
import popxl
import numpy as np
from popxl_addons.layers import Linear
from popxl_addons.module import Module
from popxl_addons.task_session import TaskSession
import os
import glob


class MockModel(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(30)
        self.l2 = Linear(10)
        self.l3 = Linear(2)

    def build(self, x: popxl.Tensor):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def build_session():
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.constant(np.random.rand(100))
        facts, g = MockModel().create_graph(x)
        vars = facts.init()
        (x,) = g.bind(vars).call(x)

    session = TaskSession(inputs=[], outputs=[], state=vars, ir=ir, device_desc="ipu_hw")
    return session


def test_save_load_ckpt():
    with TemporaryDirectory() as ckpt_dir:
        session = build_session()

        with session:
            session.run()
            session.save_checkpoint(ckpt_dir)

        files = glob.glob(os.path.join(ckpt_dir, "model", "*.npz"))
        assert len(files) == len(
            session.state.keys_flat()
        ), f"expected {len(session.state.keys_flat())} found {len(files)}"
        session2 = build_session()
        try:
            session2.load_checkpoint(ckpt_dir, report_missing="error")
        except ValueError as e:
            assert e is None


def test_missing_key():
    with TemporaryDirectory() as ckpt_dir:
        session = build_session()

        with session:
            session.run()
            session.save_checkpoint(ckpt_dir)

        files = glob.glob(os.path.join(ckpt_dir, "model", "*.npz"))
        assert len(files) == len(
            session.state.keys_flat()
        ), f"expected {len(session.state.keys_flat())} found {len(files)}"
        os.remove(os.path.join(ckpt_dir, "model", "l2.weight.npz"))

        session2 = build_session()
        try:
            session2.load_checkpoint(ckpt_dir, report_missing="error")
        except ValueError as e:
            print(e)
            assert e is not None


if __name__ == "__main__":
    test_save_load_ckpt()
    test_missing_key()
