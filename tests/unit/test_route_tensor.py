# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart.ir as pir
import popart.ir.ops as ops
from popart_ir_extensions import is_subgraph, route_tensor_into_graph


def test_is_subgraph():
    ir = pir.Ir()
    g = ir.main_graph()
    sg1 = ir.create_empty_graph("sg1")
    sg2 = ir.create_empty_graph("sg2")
    sg3 = ir.create_empty_graph("sg3")

    with g:
        ops.call(sg1)
        assert is_subgraph(g, sg1)

    with sg1:
        ops.call(sg2)
        assert is_subgraph(g, sg2)
        assert is_subgraph(sg1, sg2)

    with g:
        ops.call(sg3)
        assert is_subgraph(g, sg3)
        assert not is_subgraph(sg1, sg3)
        assert not is_subgraph(sg3, sg2)


def test_route_tensor_into_graph():
    ir = pir.Ir()
    g = ir.main_graph()
    sg = ir.create_empty_graph("sg1")

    with g:
        a = pir.variable(1)
        ops.call(sg)

    with sg:
        sg_a = route_tensor_into_graph(a)
        b = sg_a + 1

    assert len(sg.get_input_tensors()) == 1


def test_route_tensor_into_graph_many_calls():
    ir = pir.Ir()
    g = ir.main_graph()
    sg = ir.create_empty_graph("sg1")

    with g:
        a = pir.variable(1)
        ops.call(sg)
        ops.call(sg)

    with sg:
        sg_a = route_tensor_into_graph(a)
        b = sg_a + 1

    assert len(sg.get_input_tensors()) == 1
    for op in g._pb_graph.getOps():
        assert len(op.getInputTensors()) == 1


def test_route_tensor_into_graph_multiple_moves():
    ir = pir.Ir()
    g = ir.main_graph()
    sg = ir.create_empty_graph("sg1")

    with g:
        a = pir.variable(1, name="var")
        ops.call(sg)

    with sg:
        b = route_tensor_into_graph(a) + 1
        c = route_tensor_into_graph(a) + 2

    assert len(sg.get_input_tensors()) == 1


def test_route_tensor_into_graph_nested():
    ir = pir.Ir()
    g = ir.main_graph()
    sg1 = ir.create_empty_graph("sg1")
    sg2 = ir.create_empty_graph("sg2")

    with g:
        a = pir.variable(1)
        ops.call(sg1)
    with sg1:
        ops.call(sg2)
    with sg2:
        b = route_tensor_into_graph(a) + 1


def test_route_tensor_into_graph_many_paths():
    ir = pir.Ir()
    g = ir.main_graph()
    sg1 = ir.create_empty_graph("sg1")
    sg2 = ir.create_empty_graph("sg2")
    sg3 = ir.create_empty_graph("sg3")

    with g:
        a = pir.variable(1)
        ops.call(sg1)
    with sg1:
        ops.call(sg2)
        ops.call(sg3)
    with sg3:
        ops.call(sg2)
    with sg2:
        b = route_tensor_into_graph(a) + 1

    assert len(g.get_variables()) == 1
    assert len(sg1.get_input_tensors()) == 1
    assert len(sg2.get_input_tensors()) == 1
    assert len(sg3.get_input_tensors()) == 1


def test_route_tensor_into_graph_error():
    ir = pir.Ir()
    g = ir.main_graph()
    sg = ir.create_empty_graph("sg1")

    with g:
        a = pir.variable(1)

    with pytest.raises(ValueError) as excinfo:
        sg_a = route_tensor_into_graph(a, sg)
    message = str(excinfo.value)
    assert "is not in the graph" in message
    assert "is not in a known parent graph" in message


def test_route_tensor_into_graph_loop():
    ir = pir.Ir()
    g = ir.main_graph()
    sg = ir.create_empty_graph("sg1")
    with g:
        a = pir.variable(1)
        ops.repeat(sg, 5)
        with sg:
            b = route_tensor_into_graph(a) + 1
            pir.subgraph_output(b)
