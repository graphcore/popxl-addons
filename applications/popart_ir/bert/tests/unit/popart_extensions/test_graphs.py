# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import popart
import popart.ir as pir
import popart.ir.ops as ops
import popart_extensions as pir_ext


class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = pir_ext.variable_def(np.ones(x.shape, x.dtype.as_numpy()), "scale")
        return x * self.scale


class ScaleAndShift(pir_ext.GenericGraph):
    def __init__(self):
        super().__init__()
        self.scale = Scale()

    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.shift = pir_ext.variable_def(np.ones(x.shape, x.dtype.as_numpy()), "shift")
        return self.scale.build(x) + self.shift


class OutlinedScaleAndInlineShift(pir_ext.GenericGraph):
    def __init__(self, scale_graph: pir_ext.ConcreteGraph):
        super().__init__()
        self.scale_graph = scale_graph

    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = self.scale_graph.to_callable()
        self.shift = pir_ext.variable_def(np.ones(x.shape, x.dtype.as_numpy()), "shift")
        return self.scale.call(x) + self.shift


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
        scale = scale_graph.to_callable(create_variables=True)
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
        with scale_graph.graph:
            new_variable = pir_ext.variable_def(np.ones(1, np.float32), "shift")
            new_input = new_variable.create_input()
        scale_graph.insert("shift", (new_variable, new_input))

        # Create variables for each input including our new one.
        scale = scale_graph.to_callable(create_variables=True)
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
        variables_1 = scale_graph.to_callable(create_variables=True)
        # PopART should handle the collision of 'scale' already existing.
        variables_2 = scale_graph.to_callable(create_variables=True)
    assert variables_1.scale != variables_2.scale


class ScaleAndShift(pir_ext.GenericGraph):
    def __init__(self):
        super().__init__()
        self.scale = Scale()

    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.shift = pir_ext.variable_def(np.ones(x.shape, x.dtype.as_numpy()), "shift")
        return self.scale.build(x) + self.shift


def test_inline_child_variables():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        graph = ScaleAndShift().to_concrete(x)
        scale = graph.to_callable(create_variables=True)
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
        self.scale = self.scale_graph.to_callable(create_variables=False)
        self.shift = pir_ext.variable_def(np.ones(x.shape, x.dtype.as_numpy()), "shift")
        return self.scale.call(x) + self.shift


def test_outline_child_variables():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale_graph = Scale().to_concrete(x)

        scale_n_shift_graph = OutlinedScaleAndInlineShift(scale_graph).to_concrete(x)
        scale_n_shift = scale_n_shift_graph.to_callable(create_variables=True)
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
