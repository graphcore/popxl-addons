# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np

import popart.ir as pir

from popart_ir_extensions import InputFactory
from popart_ir_extensions.module import Module
from popart_ir_extensions.transforms.autodiff import autodiff_with_accumulation
from popart_ir_extensions.transforms.recomputation import recompute_graph


class Linear(Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def build(self, x: pir.Tensor):
        w = self.add_input_tensor("w", partial(np.zeros, (x.shape[-1], self.features)), pir.float32)
        b = self.add_input_tensor("b", partial(np.zeros, (self.features, )), pir.float32)
        return (x @ w) + b


def test_linear_module():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (2, 4)), pir.float32)

        args, graph = Linear(10).create_graph(x)

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 2


def test_linear_reuse_module():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (2, 4)), pir.float32)

        args, graph = Linear(4).create_graph(x)

        layer1 = graph.bind(args.init())
        layer2 = graph.bind(args.init())

        y, = layer1.call(x)
        z, = layer2.call(y)

    assert len(ir._pb_ir.getAllGraphs()) == 2


def test_linear_autodiff():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (2, 4)), pir.float32)

        args, graph = Linear(10).create_graph(x)

        grad_args, grad_graph = autodiff_with_accumulation(graph, graph.args.tensors, [graph.graph.inputs[0]])

        layer = graph.bind(args.init())
        grad_layer = grad_graph.bind(grad_args.init())

        call_info = layer.call_with_info(x)
        y, = call_info.outputs

        dx, = grad_layer.call(y, args=grad_graph.grad_graph_info.inputs_dict(call_info))

    assert len(ir._pb_ir.getAllGraphs()) == 3


def test_linear_recompute():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (2, 4)), pir.float32)

        args, graph = Linear(10).create_graph(x)

        grad_args, grad_graph = autodiff_with_accumulation(graph, graph.args.tensors, [graph.graph.inputs[0]])

        grad_graph = recompute_graph(grad_graph)

        layer = graph.bind(args.init())
        grad_layer = grad_graph.bind(grad_args.init())

        call_info = layer.call_with_info(x)
        y, = call_info.outputs

        dx, = grad_layer.call(y, args=grad_graph.grad_graph_info.inputs_dict(call_info))

    assert len(ir._pb_ir.getAllGraphs()) == 4


class DoubleLinear(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(10)
        self.linear2 = Linear(10)

    def build(self, x: pir.Tensor):
        return self.linear2(self.linear1(x))


def test_nested_module():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (2, 4)), pir.float32)

        args, graph = DoubleLinear().create_graph(x)

        assert len(args.to_dict()) == 4
        assert args.linear1.w
        assert args.linear2.w

        assert len(graph.args.to_dict()) == 4
        assert graph.args.linear1.w in graph.graph
        assert graph.args.linear2.w in graph.graph

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 2


class DoubleLinearOutlined(Module):
    def build(self, x: pir.Tensor):
        args, graph = Linear(10).create_graph(x)

        args1 = self.add_inputs("linear1", args)
        x, = graph.bind(args1).call(x)

        args2 = self.add_inputs("linear2", args)
        return graph.bind(args2).call(x)


def test_outlined_module():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (2, 10)), pir.float32)

        args, graph = DoubleLinearOutlined().create_graph(x)

        assert len(args.to_dict()) == 4
        assert args.linear1.w
        assert args.linear2.w

        assert len(graph.args.to_dict()) == 4
        assert graph.args.linear1.w in graph.graph
        assert graph.args.linear2.w in graph.graph

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 3


class Linears(Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.linears = Module.from_list([Linear(10) for i in range(self.n)])

    def build(self, x):
        for i in range(self.n):
            x = self.linears[i](x)
        return x


def test_module_list_inlined():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (2, 10)), pir.float32)

        n = 3
        linear = Linears(n)
        args, graph = linear.create_graph(x)

        for i in range(n):
            assert graph.args.linears[i].w in graph.graph

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 2


class LinearsOutlined(Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def build(self, x):
        args, graph = Linear(10).create_graph(x)

        for i in range(self.n):
            args_nt = self.add_inputs(i, args)
            x, = graph.bind(args_nt).call(x)
        return x


def test_module_list_outlined():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (2, 10)), pir.float32)

        n = 3
        linear = LinearsOutlined(n)
        args, graph = linear.create_graph(x)

        for i in range(n):
            assert graph.args[i].w in graph.graph

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 3


class MultiMatMul(Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def build(self, x):
        self.weights = Module.from_input_factories(
            self.n * [InputFactory(lambda: np.random.normal(0, 0.02, (10, 10)), pir.float16)])

        for i in range(self.n):
            x = x @ self.weights[i]

        return x


def test_from_input_factories():
    ir = pir.Ir()
    main = ir.main_graph

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (10, 10)), pir.float32)

        n = 3
        args, graph = MultiMatMul(n).create_graph(x)

        for i in range(n):
            assert graph.args.weights[i] in graph.graph

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 2
