# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
import pytest

import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons import Module, GraphWithNamedArgs
from popxl.transforms.autodiff import GradGraphInfo


np.random.seed(42)

## Utils
def equal_grad_graph_info(true: GradGraphInfo, test: GradGraphInfo, true_ir: popxl.Ir, test_ir: popxl.Ir):
    """Test if two GradGraphInfo are the same. You also need to pass the Irs so they are not garbage collected before"""
    assert isinstance(true, GradGraphInfo)
    assert isinstance(test, GradGraphInfo)

    for group in ("expected_inputs", "expected_outputs"):
        true_info = []
        for connection in getattr(true, group):
            t = connection.fwd_tensor
            true_info += [(t.name, t.dtype, t.shape, connection.connection_type.name)]
        test_info = []
        for connection in getattr(test, group):
            t = connection.fwd_tensor
            test_info += [(t.name, t.dtype, t.shape, connection.connection_type.name)]
        true_info = tuple(sorted(true_info, key=lambda x: x[0]))
        test_info = tuple(sorted(test_info, key=lambda x: x[0]))

        assert true_info == test_info, f"{group} do not match"


## Fixtures
@pytest.fixture
def x_() -> np.ndarray:
    x = np.random.rand(4, 3)
    return x


@pytest.fixture
def dy_() -> np.ndarray:
    dy = np.random.rand(4, 3)
    return dy


class Linear(Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def build(self, x: popxl.Tensor):
        self.w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
        self.b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
        return (x @ self.w) + self.b


@pytest.fixture
def linear_result(x_: np.ndarray, dy_: np.ndarray):
    """Construct linear grad graph using normal autodiff. Call forward and grad to obtain results"""
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)

        facts, fwd_graph = Linear(x.shape[-1]).create_graph(x)

        grad_graph = addons.autodiff(fwd_graph)

        vars = facts.init()
        fwd_call = fwd_graph.bind(vars).call_with_info(x)
        y, *_ = fwd_call.outputs

        dLdx, dLdw, dLdb = grad_graph.call(dy, args=grad_graph.grad_graph_info.inputs_dict(fwd_call))

        y_d2h = addons.host_store(y)
        dLdx_d2h = addons.host_store(dLdx)
        dLdw_d2h = addons.host_store(dLdw)
        dLdb_d2h = addons.host_store(dLdb)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run()

    # Ir needs to be returned so its not garbage collected
    return outputs[y_d2h], outputs[dLdx_d2h], outputs[dLdw_d2h], outputs[dLdb_d2h], ir, grad_graph.grad_graph_info


## Tests
class LinearBuildGrad(Module):
    def __init__(self, features: int, grad_inlining_error="error", cache=False):
        super().__init__(grad_inlining_error=grad_inlining_error, cache=cache)
        self.features = features

    def build(self, x: popxl.Tensor):
        self.w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
        self.b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
        self.aux["x"] = x
        self.aux["w"] = self.w
        y = (x @ self.w) + self.b
        return y

    def build_grad(self, dLdy: popxl.Tensor):
        dLdx = dLdy @ self.aux["w"].T
        dLdw = (dLdy.T @ self.aux["x"]).T
        dLdb = ops.sum(dLdy, 0)
        self.var_grad["b"] = dLdb
        self.var_grad["w"] = dLdw
        return dLdx


def test_linear_build_grad(x_, dy_, linear_result):
    """Construct linear grad graph using normal build_grad method"""
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        facts, fwd_graph = LinearBuildGrad(x.shape[-1]).create_graph(x)

        grad_graph = fwd_graph.autodiff(method="build_grad")

        assert len(grad_graph.graph.outputs) == len(fwd_graph.graph.inputs)
        assert len(grad_graph.graph.inputs) == len(fwd_graph.graph.outputs)

        vars = facts.init()
        fwd_call = fwd_graph.bind(vars).call_with_info(x)
        y, *_ = fwd_call.outputs

        outputs = grad_graph.call(dy, args=grad_graph.grad_graph_info.inputs_dict(fwd_call))
        dLdx, dLdw, dLdb = outputs

        assert len(outputs) == len(grad_graph.graph.outputs)

        # Check grad_graph correspond
        expected_output = grad_graph.grad_graph_info.expected_outputs
        assert len(outputs) == len(expected_output)
        for g, e in zip(outputs, expected_output):
            assert g.spec == e.fwd_tensor.spec

        expected_inputs = grad_graph.grad_graph_info.expected_inputs
        assert len(grad_graph.graph.inputs) == len(expected_inputs)
        for g, e in zip(grad_graph.graph.inputs, expected_inputs):
            assert g.spec == e.fwd_tensor.spec

        y_d2h = addons.host_store(y)
        dLdx_d2h = addons.host_store(dLdx)
        dLdw_d2h = addons.host_store(dLdw)
        dLdb_d2h = addons.host_store(dLdb)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run()

    y, dLdx, dLdw, dLdb = outputs[y_d2h], outputs[dLdx_d2h], outputs[dLdw_d2h], outputs[dLdb_d2h]
    y_linear, dLdx_linear, dLdw_linear, dLdb_linear, ir_linear, grad_graph_info_linear = linear_result

    for true, actual, name in zip(
        (y, dLdx, dLdw, dLdb), (y_linear, dLdx_linear, dLdw_linear, dLdb_linear), ("y", "dLdx", "dLdw", "dLdb")
    ):
        np.testing.assert_equal(actual, true, err_msg=f"{name} not equal")

    # Test if grad_graph_info the same
    equal_grad_graph_info(grad_graph.grad_graph_info, grad_graph_info_linear, ir, ir_linear)


def test_linear_force_autodiff(x_, dy_, linear_result):
    """Construct linear grad graph using autodiff transform method on graph"""
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        facts, fwd_graph = LinearBuildGrad(x.shape[-1]).create_graph(x)

        grad_graph = fwd_graph.autodiff(method="autodiff")

        vars = facts.init()
        fwd_call = fwd_graph.bind(vars).call_with_info(x)
        y, *_ = fwd_call.outputs

        outputs = grad_graph.call(dy, args=grad_graph.grad_graph_info.inputs_dict(fwd_call))
        dLdx, dLdw, dLdb = outputs

        y_d2h = addons.host_store(y)
        dLdx_d2h = addons.host_store(dLdx)
        dLdw_d2h = addons.host_store(dLdw)
        dLdb_d2h = addons.host_store(dLdb)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run()

    y, dLdx, dLdw, dLdb = outputs[y_d2h], outputs[dLdx_d2h], outputs[dLdw_d2h], outputs[dLdb_d2h]
    y_linear, dLdx_linear, dLdw_linear, dLdb_linear, ir_linear, grad_graph_info_linear = linear_result

    for true, actual, name in zip(
        (y, dLdx, dLdw, dLdb), (y_linear, dLdx_linear, dLdw_linear, dLdb_linear), ("y", "dLdx", "dLdw", "dLdb")
    ):
        np.testing.assert_equal(actual, true, err_msg=f"{name} not equal")


def test_error_with_no_from_module(x_, dy_):
    """Error when try to use autodiff when GraphWithNamedArgs has been constructed manually and does not have `from_module`"""
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        facts, fwd_graph = LinearBuildGrad(x.shape[-1]).create_graph(x)

        # Create new GraphWithNamedArgs manually without specifying `from_module`
        new_graph = GraphWithNamedArgs(fwd_graph.graph, args=fwd_graph.args)

        with pytest.raises(Exception, match="`from_module` is specified"):
            grad_graph = new_graph.autodiff(method="build_grad")


def test_inlining_behaviour(x_):
    """Error or warning when user inlines code"""

    class LinearInlined(Module):
        def __init__(self, features: int, grad_inlining_error="error"):
            super().__init__()
            self.linear = LinearBuildGrad(features=features, grad_inlining_error=grad_inlining_error)

        def build(self, x: popxl.Tensor):
            x = 2 * x
            x = self.linear(x)
            return x

    ir = popxl.Ir()
    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)

        # Print error as invalid value for grad_inlining_error
        with pytest.raises(ValueError, match="grad_inlining_error should be one of"):
            LinearInlined(x.shape[-1], grad_inlining_error="").create_graph(x)

        # Should throw an error
        with pytest.raises(Exception, match="`build_grad` has been implemented"):
            LinearInlined(x.shape[-1], grad_inlining_error="error").create_graph(x)

        # Print warning
        with pytest.warns(UserWarning, match="`build_grad` has been implemented"):
            LinearInlined(x.shape[-1], grad_inlining_error="warning").create_graph(x)

        # Works without error
        LinearInlined(x.shape[-1], grad_inlining_error="none").create_graph(x)


class LinearOutlined(Module):
    def __init__(self):
        super().__init__()

    def build(self, x: popxl.Tensor):
        x = 1 * x
        (y,) = self.call_module(LinearBuildGrad(x.shape[-1]))(x)
        return y


def test_outlining(x_, dy_, linear_result):
    """Test outlining behaviour (module being created into a graph that calls another graph)"""

    ir = popxl.Ir()
    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)

        facts, fwd_graph = LinearOutlined().create_graph(x)

        # Test error without called_graphs_grad_info
        called_graphs_grad_info = fwd_graph.called_graphs_grad_info
        fwd_graph.called_graphs_grad_info = {}

        with pytest.raises(Exception, match="marked as unsafe to autodiff"):
            grad_graph = fwd_graph.autodiff()

        # Test with called_graphs_grad_info
        fwd_graph.called_graphs_grad_info = called_graphs_grad_info

        grad_graph = fwd_graph.autodiff()

        vars = facts.init()
        fwd_call = fwd_graph.bind(vars).call_with_info(x)
        y, *_ = fwd_call.outputs

        outputs = grad_graph.call(dy, args=grad_graph.grad_graph_info.inputs_dict(fwd_call))
        dLdx, dLdw, dLdb = outputs

        assert len(outputs) == len(grad_graph.graph.outputs)

        # Check grad_graph correspond
        expected_output = grad_graph.grad_graph_info.expected_outputs
        assert len(outputs) == len(expected_output)
        for g, e in zip(outputs, expected_output):
            assert g.spec == e.fwd_tensor.spec

        expected_inputs = grad_graph.grad_graph_info.expected_inputs
        assert len(grad_graph.graph.inputs) == len(expected_inputs)
        for g, e in zip(grad_graph.graph.inputs, expected_inputs):
            assert g.spec == e.fwd_tensor.spec

        y_d2h = addons.host_store(y)
        dLdx_d2h = addons.host_store(dLdx)
        dLdw_d2h = addons.host_store(dLdw)
        dLdb_d2h = addons.host_store(dLdb)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run()

    y, dLdx, dLdw, dLdb = outputs[y_d2h], outputs[dLdx_d2h], outputs[dLdw_d2h], outputs[dLdb_d2h]
    y_linear, dLdx_linear, dLdw_linear, dLdb_linear, ir_linear, grad_graph_info_linear = linear_result

    for true, actual, name in zip(
        (y, dLdx, dLdw, dLdb), (y_linear, dLdx_linear, dLdw_linear, dLdb_linear), ("y", "dLdx", "dLdw", "dLdb")
    ):
        np.testing.assert_equal(actual, true, err_msg=f"{name} not equal")


def test_outlining_inline_nesting(x_, dy_, linear_result):
    """Test inlining a module which itself calls a build_grad graph (outlined)"""

    class LinearOutlinedInlined(Module):
        def __init__(self):
            super().__init__()
            self.linear = LinearOutlined()

        def build(self, x: popxl.Tensor):
            x = x * 1
            y = self.linear(x)
            return y

    ir = popxl.Ir()
    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)

        module = LinearOutlinedInlined()
        facts, fwd_graph = module.create_graph(x)

        assert len(fwd_graph.called_graphs_grad_info) > 0

        # Empty called_graphs_grad_info and should error
        called_graphs_grad_info = fwd_graph.called_graphs_grad_info
        fwd_graph.called_graphs_grad_info = {}

        with pytest.raises(Exception, match="marked as unsafe to autodiff"):
            grad_graph = fwd_graph.autodiff(method="autodiff")

        # Test with called_graphs_grad_info
        fwd_graph.called_graphs_grad_info = called_graphs_grad_info

        grad_graph = fwd_graph.autodiff(method="autodiff")

        vars = facts.init()
        fwd_call = fwd_graph.bind(vars).call_with_info(x)
        y, *_ = fwd_call.outputs

        outputs = grad_graph.call(dy, args=grad_graph.grad_graph_info.inputs_dict(fwd_call))
        dLdx, dLdw, dLdb = outputs

        assert len(outputs) == len(grad_graph.graph.outputs)

        # Check grad_graph correspond
        expected_output = grad_graph.grad_graph_info.expected_outputs
        assert len(outputs) == len(expected_output)
        for g, e in zip(outputs, expected_output):
            assert g.spec == e.fwd_tensor.spec

        expected_inputs = grad_graph.grad_graph_info.expected_inputs
        assert len(grad_graph.graph.inputs) == len(expected_inputs)
        for g, e in zip(grad_graph.graph.inputs, expected_inputs):
            assert g.spec == e.fwd_tensor.spec

        y_d2h = addons.host_store(y)
        dLdx_d2h = addons.host_store(dLdx)
        dLdw_d2h = addons.host_store(dLdw)
        dLdb_d2h = addons.host_store(dLdb)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run()

    y, dLdx, dLdw, dLdb = outputs[y_d2h], outputs[dLdx_d2h], outputs[dLdw_d2h], outputs[dLdb_d2h]
    y_linear, dLdx_linear, dLdw_linear, dLdb_linear, ir_linear, grad_graph_info_linear = linear_result

    for true, actual, name in zip(
        (y, dLdx, dLdw, dLdb), (y_linear, dLdx_linear, dLdw_linear, dLdb_linear), ("y", "dLdx", "dLdw", "dLdb")
    ):
        np.testing.assert_equal(actual, true, err_msg=f"{name} not equal")


def test_outlining_outlining_nesting(x_, dy_, linear_result):
    """Test inlining a module which itself calls a build_grad graph (outlined)"""

    class LinearOutlinedOutlined(Module):
        def __init__(self):
            super().__init__()

        def build(self, x: popxl.Tensor):
            x = x * 1
            y = self.call_module(LinearOutlined())(x)
            return y

    ir = popxl.Ir()
    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)

        module = LinearOutlinedOutlined()
        facts, fwd_graph = module.create_graph(x)

        # Test error without called_graphs_grad_info
        assert len(fwd_graph.called_graphs_grad_info) > 0

        # Empty called_graphs_grad_info and should error
        called_graphs_grad_info = fwd_graph.called_graphs_grad_info
        fwd_graph.called_graphs_grad_info = {}

        with pytest.raises(Exception, match="marked as unsafe to autodiff"):
            grad_graph = fwd_graph.autodiff(method="autodiff")

        # Test with called_graphs_grad_info
        fwd_graph.called_graphs_grad_info = called_graphs_grad_info

        grad_graph = fwd_graph.autodiff(method="autodiff")

        vars = facts.init()
        fwd_call = fwd_graph.bind(vars).call_with_info(x)
        y, *_ = fwd_call.outputs

        outputs = grad_graph.call(dy, args=grad_graph.grad_graph_info.inputs_dict(fwd_call))
        dLdx, dLdw, dLdb = outputs

        assert len(outputs) == len(grad_graph.graph.outputs)

        # Check grad_graph correspond
        expected_output = grad_graph.grad_graph_info.expected_outputs
        assert len(outputs) == len(expected_output)
        for g, e in zip(outputs, expected_output):
            assert g.spec == e.fwd_tensor.spec

        expected_inputs = grad_graph.grad_graph_info.expected_inputs
        assert len(grad_graph.graph.inputs) == len(expected_inputs)
        for g, e in zip(grad_graph.graph.inputs, expected_inputs):
            assert g.spec == e.fwd_tensor.spec

        y_d2h = addons.host_store(y)
        dLdx_d2h = addons.host_store(dLdx)
        dLdw_d2h = addons.host_store(dLdw)
        dLdb_d2h = addons.host_store(dLdb)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run()

    y, dLdx, dLdw, dLdb = outputs[y_d2h], outputs[dLdx_d2h], outputs[dLdw_d2h], outputs[dLdb_d2h]
    y_linear, dLdx_linear, dLdw_linear, dLdb_linear, ir_linear, grad_graph_info_linear = linear_result

    for true, actual, name in zip(
        (y, dLdx, dLdw, dLdb), (y_linear, dLdx_linear, dLdw_linear, dLdb_linear), ("y", "dLdx", "dLdw", "dLdb")
    ):
        np.testing.assert_equal(actual, true, err_msg=f"{name} not equal")


def test_inline_module_to_build_grad_module(x_, dy_):
    """Test inlining a module without the build_grad into a build_grad. The namespace of the grads will be nested"""

    class LinearWithLinearInlined(Module):
        def __init__(self, features: int):
            super().__init__()
            self.linear = Linear(features)  # Normal Linear with no build_grad

        def build(self, x: popxl.Tensor):
            y = self.linear(x)
            self.aux["x"] = x
            self.aux["w"] = self.linear.w
            return y

        def build_grad(self, dLdy: popxl.Tensor):
            dLdx = dLdy @ self.aux["w"].T
            dLdw = (dLdy.T @ self.aux["x"]).T
            dLdb = ops.sum(dLdy, 0)
            # Nested in linear namespace
            self.var_grad["linear.b"] = dLdb
            self.var_grad["linear.w"] = dLdw
            return dLdx

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = LinearWithLinearInlined(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        fwd_graph.autodiff(method="build_grad")


def test_var_grad(x_, dy_):
    """Test error if use var_grad outside the `build_grad` context and error if missing a gradient for a variable"""

    class Linear(Module):
        def __init__(self, features: int):
            super().__init__()
            self.features = features

        def build(self, x: popxl.Tensor):
            w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
            b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
            self.aux["x"] = x
            self.aux["w"] = w
            y = (x @ w) + b

            # Cannot use var_grad outside build_grad context
            with pytest.raises(Exception, match=" within the `build_graph` method"):
                module.var_grad["q"] = 0

            return y

        def build_grad(self, dLdy: popxl.Tensor):
            dLdx = dLdy @ self.aux["w"].T
            dLdw = (dLdy.T @ self.aux["x"]).T
            dLdb = ops.sum(dLdy, 0)
            self.var_grad["b"] = dLdb

            # Can only store tensors
            with pytest.raises(TypeError, match="Not a tensor"):
                self.var_grad["q"] = 0

            with pytest.raises(KeyError, match="Not a variable"):
                self.var_grad["Z"] = dLdb  # Z does not exist as a variable

            # dLdw missing
            # self.var_grad['w'] = dLdw

            # Contains in works
            assert "b" in self.var_grad
            assert "h" not in self.var_grad

            return dLdx

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = Linear(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        # dLdw missing
        with pytest.raises(Exception, match="Not all variable gradients have been specified"):
            grad_graph = fwd_graph.autodiff(method="build_grad")

        # Cannot use var_grad outside build_grad context
        with pytest.raises(Exception, match=" within the `build_graph` method"):
            module.var_grad["q"] = 0

        # Cannot use var_grad outside build_grad context
        with pytest.raises(Exception, match=" within the `build_graph` method"):
            _ = module.var_grad["q"]


def test_aux(x_, dy_):
    """Test error obtaining tensors outside `build_grad` content and other `aux` features"""

    class Linear(Module):
        def __init__(self, features: int):
            super().__init__()
            self.features = features

        def build(self, x: popxl.Tensor):
            w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
            b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
            self.aux["x"] = x
            self.aux["w"] = w

            # Its ok to set non-tensors
            self.aux["string"] = "string"

            # Cannot get items in build method
            with pytest.raises(Exception, match="get items within the `build_graph`"):
                _ = self.aux["w"]

            y = (x @ w) + b
            return y

        def build_grad(self, dLdy: popxl.Tensor):
            # Its ok to retrieve non-tensors
            s = self.aux["string"]

            # Contains in works
            assert "w" in self.aux
            assert "h" not in self.aux

            # If you fetch twice you should get the same value
            # Test as fist call to `self.aux["w"]` has a different effect to the second call
            w = self.aux["w"]
            w_new = self.aux["w"]
            assert w == w_new

            dLdx = dLdy @ self.aux["w"].T
            dLdw = (dLdy.T @ self.aux["x"]).T
            dLdb = ops.sum(dLdy, 0)

            self.var_grad["b"] = dLdb
            self.var_grad["w"] = dLdw
            return dLdx

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = Linear(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        grad_graph = fwd_graph.autodiff(method="build_grad")

        # Cannot get aux outside build_grad context
        with pytest.raises(Exception, match=" within the `build_graph` method"):
            module.var_grad["q"]


def test_check_output_shapes(x_, dy_):
    """Test error when output shapes are wrong"""

    class Linear(Module):
        def __init__(self, features: int):
            super().__init__()
            self.features = features

        def build(self, x: popxl.Tensor):
            w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
            b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
            self.aux["x"] = x
            self.aux["w"] = w
            y = (x @ w) + b
            return y

        def build_grad(self, dLdy: popxl.Tensor):
            dLdx = dLdy @ self.aux["w"].T
            dLdw = (dLdy.T @ self.aux["x"]).T
            dLdb = ops.sum(dLdy, 0)
            self.var_grad["b"] = dLdb
            with pytest.raises(Exception, match="Shape or meta-shape of Tensor does not match variable."):
                self.var_grad["w"] = dLdb
            self.var_grad["w"] = dLdw
            return dLdb  # Output wrong shape

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = Linear(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        # Output wrong shape
        with pytest.raises(Exception, match="Shape or meta-shape of output Tensor at position 0"):
            fwd_graph.autodiff(method="build_grad")


def test_output_too_many(x_, dy_):
    """Test error when output too many tensors"""

    class Linear(Module):
        def __init__(self, features: int):
            super().__init__()
            self.features = features

        def build(self, x: popxl.Tensor):
            w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
            b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
            self.aux["x"] = x
            self.aux["w"] = w
            y = (x @ w) + b
            return y

        def build_grad(self, dLdy: popxl.Tensor):
            dLdx = dLdy @ self.aux["w"].T
            dLdw = (dLdy.T @ self.aux["x"]).T
            dLdb = ops.sum(dLdy, 0)
            self.var_grad["b"] = dLdb
            self.var_grad["w"] = dLdw
            return dLdx, dLdx  # Too many outputs

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = Linear(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        # Output wrong shape
        with pytest.raises(Exception, match="Too many outputs in the grad graph"):
            fwd_graph.autodiff(method="build_grad")


def test_output_not_enough(x_, dy_):
    """Test error when don't output all required gradients"""

    class Linear(Module):
        def __init__(self, features: int):
            super().__init__()
            self.features = features

        def build(self, x: popxl.Tensor):
            w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
            b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
            self.aux["x"] = x
            self.aux["w"] = w
            y = (x @ w) + b
            return y

        def build_grad(self, dLdy: popxl.Tensor):
            dLdx = dLdy @ self.aux["w"].T
            dLdw = (dLdy.T @ self.aux["x"]).T
            dLdb = ops.sum(dLdy, 0)
            self.var_grad["b"] = dLdb
            self.var_grad["w"] = dLdw
            return  # 1 few output

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = Linear(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        # Output wrong shape
        with pytest.raises(Exception, match="Not all forward graph inputs have been specified"):
            fwd_graph.autodiff(method="build_grad")


def test_build_grad_includes_variable_inputs(x_, dy_):
    """Test error when `build_grad` method signature has a `*args`"""

    class Linear(Module):
        def __init__(self, features: int):
            super().__init__()
            self.features = features

        def build(self, x: popxl.Tensor):
            w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
            b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
            self.aux["x"] = x
            self.aux["w"] = w
            y = (x @ w) + b
            return y

        def build_grad(self, *args):  # Method signature has variable
            return

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = Linear(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        # Output wrong shape
        with pytest.raises(
            TypeError, match="method signature must only have a fixed number of positional parameters as inputs."
        ):
            fwd_graph.autodiff(method="build_grad")


def test_grads_provided(x_, dy_):
    """Test grads_provided errors"""

    class Linear(Module):
        def __init__(self, features: int):
            super().__init__()
            self.features = features

        def build(self, x: popxl.Tensor):
            w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
            b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
            self.aux["x"] = x
            self.aux["w"] = w
            y = (x @ w) + b
            y2 = 0.5 * y
            return y, y2  # Outputs two

        def build_grad(self, dLdy):  # Only require one grad
            dLdx = dLdy @ self.aux["w"].T
            dLdw = (dLdy.T @ self.aux["x"]).T
            dLdb = ops.sum(dLdy, 0)
            self.var_grad["b"] = dLdb
            self.var_grad["w"] = dLdw
            return dLdx

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = Linear(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        # Not a tensor of the forward graph
        with pytest.raises(ValueError, match="A Tensor specified in `grads_provided` is not a output of `fwd_graph`"):
            fwd_graph.autodiff(method="build_grad", grads_provided=[dy])

        # Not enough grads_provided (require one)
        with pytest.raises(ValueError, match="do not match the signature"):
            fwd_graph.autodiff(method="build_grad", grads_provided=[])

        # Correct usage
        fwd_graph.autodiff(method="build_grad", grads_provided=[fwd_graph.graph.outputs[0]])


def test_grads_required(x_, dy_):
    """Test grads_required errors"""

    class Linear(Module):
        def __init__(self, features: int):
            super().__init__()
            self.features = features

        def build(self, x: popxl.Tensor, x2: popxl.Tensor):
            w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
            b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
            self.aux["x"] = x
            self.aux["w"] = w
            y = (x * x2 @ w) + b
            return y

        def build_grad(self, dLdy):
            dLdx = dLdy @ self.aux["w"].T
            dLdw = (dLdy.T @ self.aux["x"]).T
            dLdb = ops.sum(dLdy, 0)
            self.var_grad["b"] = dLdb
            # self.var_grad['w'] = dLdw # Don't require variable grad
            return dLdx  # Only require grad of one input

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = Linear(x.shape[-1])
        facts, fwd_graph = module.create_graph(x, x)

        # Not a tensor of the forward graph
        with pytest.raises(ValueError, match="A Tensor specified in `grads_required` is not a input of `fwd_graph`"):
            fwd_graph.autodiff(method="build_grad", grads_required=[dy])

        # Specified too many grads in output
        with pytest.raises(Exception, match="Too many outputs in the grad graph"):
            fwd_graph.autodiff(method="build_grad", grads_required=[fwd_graph.args.b])

        # Specify all gradients provided in `build_grad`
        grad_graph_info = fwd_graph.autodiff(
            method="build_grad", grads_required=[fwd_graph.args.b, fwd_graph.graph.inputs[1]]
        )
        assert len(grad_graph_info.grad_graph_info.expected_outputs) == 2
        assert len(grad_graph_info.graph.outputs) == 2

        # Specify a subset of gradients provided in `build_grad`
        facts, fwd_graph = module.create_graph(x, x)
        grad_graph_info = fwd_graph.autodiff(method="build_grad", grads_required=[fwd_graph.graph.inputs[1]])
        assert len(grad_graph_info.grad_graph_info.expected_outputs) == 1
        assert len(grad_graph_info.graph.outputs) == 1


def test_graph_caching(x_, dy_):
    """Test if error is produced if used in conjunction with graph caching"""

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)

        with pytest.raises(Exception, match="You cannot use graph caching if you have implemented `build_grad`"):
            module = LinearBuildGrad(x.shape[-1], cache=True)


def test_autodiff_multiple_calls(x_, dy_):
    """Applying autodiff twice should raise an error"""

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(x_, popxl.float32)
        dy = popxl.variable(dy_, popxl.float32)
        module = LinearBuildGrad(x.shape[-1])
        facts, fwd_graph = module.create_graph(x)

        # First call
        fwd_graph.autodiff()

        # Second call
        with pytest.raises(Exception, match="Autodiff has already been applied"):
            fwd_graph.autodiff()


# refractor
# docs
