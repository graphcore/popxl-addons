# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import popxl
import pytest
import numpy as np
from popxl_addons.layers import Linear
from popxl_addons.module import Module
from popxl_addons.task_session import TaskSession
from pathlib import Path
import os
import glob
import shutil


class TestModel(Module):
    def __init__(self):
        super().__init__()
        self.l1_baz = Linear(30)
        self.l2 = Linear(10)
        self.l3 = Linear(2)

    def build(self, x: popxl.Tensor):
        x = self.l1_baz(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def build_session():
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.constant(np.random.rand(100))
        facts, g = TestModel().create_graph(x)
        vars = facts.init()
        x, = g.bind(vars).call(x)

    session = TaskSession(inputs=[], outputs=[], state=vars, ir=ir, device_desc="ipu_hw")
    return session


def test_save_load_ckpt():
    test_dir = Path(__file__).parent.resolve()
    ckpt_dir = os.path.join(test_dir, "test_ckpts")
    session = build_session()

    with session:
        session.run()
        session.save_checkpoint(ckpt_dir)

    files = glob.glob(os.path.join(ckpt_dir, "model", "*.npz"))
    assert len(files) == len(session.state.keys_flat()), f"expected {len(session.state.keys_flat())} found {len(files)}"
    session2 = build_session()
    try:
        session2.load_checkpoint(ckpt_dir, report_missing='error')
    except ValueError as e:
        assert e is None

    shutil.rmtree(ckpt_dir)


def test_missing_key():
    test_dir = Path(__file__).parent.resolve()
    ckpt_dir = os.path.join(test_dir, "test_ckpts")
    session = build_session()

    with session:
        session.run()
        session.save_checkpoint(ckpt_dir)

    files = glob.glob(os.path.join(ckpt_dir, "model", "*.npz"))
    assert len(files) == len(session.state.keys_flat()), f"expected {len(session.state.keys_flat())} found {len(files)}"
    os.remove(os.path.join(ckpt_dir, "model", "l2_weight.npz"))

    session2 = build_session()
    try:
        session2.load_checkpoint(ckpt_dir, report_missing='error')
    except ValueError as e:
        print(e)
        assert e is not None

    shutil.rmtree(ckpt_dir)


if __name__ == '__main__':
    test_save_load_ckpt()
    test_missing_key()
