# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import popart.ir as pir
import popart.ir.ops as ops
import pytest

import popart_ir_extensions as pir_ext
from popart_ir_extensions.graphs import GenericGraphList


class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        scale = self.add_input_tensor("scale", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return x * scale


def test_variable_explictly_created_with_subgraph():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        scale_graph = Scale().to_concrete(x)

        # Variable has not been created
        with pytest.raises(popart.popart_exception):
            main._pb_graph.getTensor("scale")

        # Construct variables for the graph.
        scale = scale_graph.to_callable(create_inputs=True)
        scale.call(x)

    variable = main._pb_graph.getTensor("scale")
    assert variable
    assert variable.tensor_type() == "Variable"
    assert variable.id == scale.scale.id


def test_add_variables_post_construction():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        scale_graph = Scale().to_concrete(x)

        # Do some transformations...
        with scale_graph as g:
            g.add_input_tensor(lambda: np.ones(1, np.float32), "shift")

        # Create variables for each input including our new one.
        scale = scale_graph.to_callable(create_inputs=True)
        scale.call(x)

    variable = main._pb_graph.getTensor("shift")
    assert variable
    assert variable.tensor_type() == "Variable"
    assert variable.id == scale.shift.id
    assert scale.call_input() == {
        scale_graph.scale: scale.scale,
        scale_graph.shift: scale.shift
    }


def test_variables_no_variable_conflict():
    ir = pir.Ir()
    main = ir.main_graph()
    scale_fn = Scale()
    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        scale_graph = scale_fn.to_concrete(x)

        # Construct variables for the graph.
        variables_1 = scale_graph.to_callable(create_inputs=True)
        # PopART should handle the collision of 'scale' already existing.
        variables_2 = scale_graph.to_callable(create_inputs=True)
    assert variables_1.scale != variables_2.scale


class ScaleAndShift(pir_ext.GenericGraph):
    def __init__(self):
        super().__init__()
        self.scale = Scale()

    def build(self, x: pir.Tensor) -> pir.Tensor:
        shift = self.add_input_tensor("shift", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return self.scale.build(x) + shift


def test_inline_child_variables():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        graph = ScaleAndShift().to_concrete(x)
        scale = graph.to_callable(create_inputs=True)
        scale.call(x)

    assert scale.shift
    assert scale.scale.scale
    assert scale.call_input() == {
        graph.shift: scale.shift,
        graph.scale.scale: scale.scale.scale}


class OutlinedScaleAndInlineShift(pir_ext.GenericGraph):
    def __init__(self, scale_graph: pir_ext.ConcreteGraph):
        super().__init__()
        self.scale_graph = scale_graph

    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = self.scale_graph.to_callable(create_inputs=False)
        shift = self.add_input_tensor("shift", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return self.scale.call(x) + shift


def test_outline_child_variables():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale_graph = Scale().to_concrete(x)

        scale_n_shift_graph = OutlinedScaleAndInlineShift(scale_graph).to_concrete(x)
        scale_n_shift = scale_n_shift_graph.to_callable(create_inputs=True)
        scale_n_shift.call(x)

    assert scale_n_shift.shift
    assert scale_n_shift.scale.scale
    assert scale_n_shift.call_input() == {
        scale_n_shift_graph.shift: scale_n_shift.shift,
        scale_n_shift_graph.scale.scale: scale_n_shift.scale.scale}


def test_graph_decorator():
    @pir_ext.graph
    def scale(x: pir.Tensor, scale: pir.Tensor) -> pir.Tensor:
        return x * scale

    assert isinstance(scale, pir_ext.GenericGraph)


def test_generic_graph_list():
    gl = GenericGraphList([Scale(), Scale()])
    assert isinstance(gl.get(0), Scale)
    assert isinstance(gl.get(1), Scale)
    assert gl.get(0) == gl.i0
    assert gl.get(1) == gl.i1
    assert len(gl) == 2
    assert gl.get(0) != gl.get(1)

    with pytest.raises(IndexError):
        gl.get(2)

    with pytest.raises(AttributeError):
        gl.i2


class ScaleTwice(pir_ext.GenericGraph):
    def __init__(self):
        super().__init__()
        self.scales = GenericGraphList([Scale(), Scale()])

    def build(self, x: pir.Tensor) -> pir.Tensor:
        x = self.scales.get(0).build(x)
        x = self.scales.get(1).build(x)
        return x


def test_generic_graph_list_nested():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        st_graph = ScaleTwice().to_concrete(x)

        assert st_graph.scales.get(0)
        assert st_graph.scales.get(1)

        st = st_graph.to_callable(create_inputs=True)
        st.call(x)

    assert st.scales
    assert st.scales.i0
    assert st.scales.i1

    assert st.scales.get(0)
    assert st.scales.get(1)
