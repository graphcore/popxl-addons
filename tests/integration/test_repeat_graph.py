# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import popxl
import pytest
import numpy as np
import popxl_addons as addons
from popart._internal import ir as _ir
from popxl_addons.transforms.repeat_graph import repeat_graph


@pytest.mark.parametrize('repeat_count', [1, 5])
def test_basic(repeat_count):
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.constant(0) + 1
        addons.host_store(x)

    repeat_graph(ir.main_graph, repeat_count)

    ir.num_host_transfers = repeat_count
    output = popxl.Session(ir, 'ipu_hw').run()
    assert np.sum(list(output.values())) == repeat_count


def test_variables():
    repeat_count = 5
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(0) + 1
        addons.host_store(x)

    repeat_graph(ir.main_graph, repeat_count)

    ir.num_host_transfers = repeat_count
    output = popxl.Session(ir, 'ipu_hw').run()
    assert np.sum(list(output.values())) == repeat_count


def test_topological_constraints():
    repeat_count = 5
    ir = popxl.Ir()

    with ir.main_graph:
        small = popxl.variable(1, name='small')
        big = popxl.variable(np.ones((2, 2), np.float32), name='big')

        # From a liveness perspective,
        #   these two Ops are in the wrong order.
        with popxl.in_sequence():
            b = big + 1
            a = small * 1

    new_graph = repeat_graph(ir.main_graph, repeat_count)

    ops_sorted = new_graph._pb_graph.getOpSchedule()
    assert isinstance(ops_sorted[0], _ir.op.AddOp)
    assert isinstance(ops_sorted[1], _ir.op.MulOp)
