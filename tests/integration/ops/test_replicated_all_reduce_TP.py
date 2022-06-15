# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from popart import popart_exception
import popxl
import popxl.ops as ops
import numpy as np

from popxl_addons.ops.replicated_all_reduce_TP.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs, replicated_all_reduce_identical_grad_inputs)
from popxl_addons.patterns import apply_pre_alias_patterns


def test_all_reduce_identical_inputs_op():
    n_ipus = 4

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    ir.replication_factor = n_ipus
    main = ir.main_graph
    with main:

        x_h2d = popxl.h2d_stream((2, 3), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name='x')

        y = replicated_all_reduce_identical_inputs(x)

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level='default')

    with popxl.Session(ir, device_desc="ipu_hw") as session:
        y_host = session.run({x_h2d: inputs})
    y_host = y_host[y_d2h]

    # Outputs should be identical to inputs
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(inputs[i], y_host_i)


def test_all_reduce_identical_inputs_op_backwards():
    n_ipus = 4

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    ir.replication_factor = 4
    main = ir.main_graph
    with main:

        x_h2d = popxl.h2d_stream((2, 3), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name='x')

        all_reduce_graph = ir.create_graph(replicated_all_reduce_identical_inputs, x)

    # Auto diff
    all_reduce_graph_grad_info = popxl.transforms.autodiff(all_reduce_graph)
    all_reduce_graph_grad = all_reduce_graph_grad_info.graph

    with main:
        # call backwards
        y, = ops.call(all_reduce_graph_grad, x)

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level='default')

    with popxl.Session(ir, device_desc="ipu_hw") as session:
        y_host = session.run({x_h2d: inputs})
    y_host = y_host[y_d2h]

    # Outputs should be sum of inputs
    target = inputs.sum(axis=0)
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target, y_host_i)


def test_all_reduce_identical_grad_inputs_op():
    n_ipus = 4

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    ir.replication_factor = n_ipus
    main = ir.main_graph
    with main:

        x_h2d = popxl.h2d_stream((2, 3), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name='x')

        y = replicated_all_reduce_identical_grad_inputs(x)

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level='default')

    with popxl.Session(ir, device_desc="ipu_hw") as session:
        y_host = session.run({x_h2d: inputs})
    y_host = y_host[y_d2h]

    # Outputs should be sum of inputs
    target = inputs.sum(axis=0)
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target, y_host_i)


def test_all_reduce_identical_grad_inputs_op_backwards():
    n_ipus = 4

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    ir.replication_factor = 4
    main = ir.main_graph
    with main:

        x_h2d = popxl.h2d_stream((2, 3), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name='x')

        all_reduce_graph = ir.create_graph(replicated_all_reduce_identical_grad_inputs, x)

    # Auto diff
    all_reduce_graph_grad_info = popxl.transforms.autodiff(all_reduce_graph)
    all_reduce_graph_grad = all_reduce_graph_grad_info.graph

    with main:
        # call backwards
        y, = ops.call(all_reduce_graph_grad, x)

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level='default')

    with popxl.Session(ir, device_desc="ipu_hw") as session:
        y_host = session.run({x_h2d: inputs})
    y_host = y_host[y_d2h]

    # Outputs should be identical to inputs
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(inputs[i], y_host_i)


def test_error_need_to_run_pattern():
    n_ipus = 4

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    ir.replication_factor = 4
    main = ir.main_graph
    with main:

        x_h2d = popxl.h2d_stream((2, 3), popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, name='x')

        all_reduce_graph = ir.create_graph(replicated_all_reduce_identical_grad_inputs, x)

    # Auto diff
    all_reduce_graph_grad_info = popxl.transforms.autodiff(all_reduce_graph)
    all_reduce_graph_grad = all_reduce_graph_grad_info.graph

    with main:
        # call backwards
        y, = ops.call(all_reduce_graph_grad, x)

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    with pytest.raises(popart_exception):
        with popxl.Session(ir, device_desc="ipu_hw") as session:
            y_host = session.run({x_h2d: inputs})
