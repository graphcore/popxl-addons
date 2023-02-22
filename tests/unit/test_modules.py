# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np

import popxl

from popxl_addons import VariableFactory, NamedTensors, Module
from popxl_addons.transforms.autodiff import autodiff_with_accumulation
from popxl_addons.transforms.recomputation import recompute_graph


class Linear(Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def build(self, x: popxl.Tensor):
        w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
        b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
        return (x @ w) + b


def test_linear_module():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 4)), popxl.float32)

        args, graph = Linear(10).create_graph(x)

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 2


def test_linear_reuse_module():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 4)), popxl.float32)

        args, graph = Linear(4).create_graph(x)

        layer1 = graph.bind(args.init())
        layer2 = graph.bind(args.init())

        (y,) = layer1.call(x)
        (z,) = layer2.call(y)

    assert len(ir._pb_ir.getAllGraphs()) == 2


def test_linear_autodiff():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 4)), popxl.float32)

        args, graph = Linear(10).create_graph(x)

        grad_args, grad_graph = autodiff_with_accumulation(
            graph, graph.args.tensors, grads_required=[graph.graph.inputs[0]]
        )

        layer = graph.bind(args.init())
        grad_layer = grad_graph.bind(grad_args.init())

        call_info = layer.call_with_info(x)
        (y,) = call_info.outputs

        (dx,) = grad_layer.call(y, args=grad_graph.grad_graph_info.inputs_dict(call_info))

    assert len(ir._pb_ir.getAllGraphs()) == 3


def test_linear_recompute():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 4)), popxl.float32)

        args, graph = Linear(10).create_graph(x)

        grad_args, grad_graph = autodiff_with_accumulation(
            graph, graph.args.tensors, grads_required=[graph.graph.inputs[0]]
        )

        grad_graph = recompute_graph(grad_graph)

        layer = graph.bind(args.init())
        grad_layer = grad_graph.bind(grad_args.init())

        call_info = layer.call_with_info(x)
        (y,) = call_info.outputs

        (dx,) = grad_layer.call(y, args=grad_graph.grad_graph_info.inputs_dict(call_info))

    assert len(ir._pb_ir.getAllGraphs()) == 4


class DoubleLinear(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(10)
        self.linear2 = Linear(10)

    def build(self, x: popxl.Tensor):
        return self.linear2(self.linear1(x))


def test_nested_module():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 4)), popxl.float32)

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
    def build(self, x: popxl.Tensor):
        args, graph = Linear(10).create_graph(x)

        args1 = self.add_variable_inputs("linear1", args)
        (x,) = graph.bind(args1).call(x)

        args2 = self.add_variable_inputs("linear2", args)
        return graph.bind(args2).call(x)


def test_outlined_module():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 10)), popxl.float32)

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
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 10)), popxl.float32)

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
            args_nt = self.add_variable_inputs(i, args)
            (x,) = graph.bind(args_nt).call(x)
        return x


def test_module_list_outlined():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 10)), popxl.float32)

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
        self.weights = Module.from_variable_factories(
            self.n * [VariableFactory(lambda: np.random.normal(0, 0.02, (10, 10)), popxl.float16)]
        )

        for i in range(self.n):
            x = x @ self.weights[i]

        return x


def test_from_variable_factories():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (10, 10)), popxl.float32)

        n = 3
        args, graph = MultiMatMul(n).create_graph(x)

        for i in range(n):
            assert graph.args.weights[i] in graph.graph

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 2


class SubModuleWithMerge(Module):
    def build(self, x):
        lin_facts, lin_graph = Linear(10).create_graph(x)

        lin_vars = self.add_variable_inputs("fwd.linear", lin_facts)

        self.offset = self.add_variable_input(
            "fwd.offset",
            lambda: np.array([1.0]),
            popxl.float32,
            overwrite=True,  # You have to merge the DotTrees as both are in the `fwd` namespace
        )

        (y,) = lin_graph.bind(lin_vars).call(x)
        z = y + self.offset

        return z


def test_merging_variables():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (10, 10)), popxl.float32)

        facts, graph = SubModuleWithMerge().create_graph(x)
        vars = facts.init()
        z = graph.bind(vars).call(x)

        assert vars.fwd.linear.w
        assert vars.fwd.linear.b
        assert vars.fwd.offset

    assert len(ir._pb_ir.getAllGraphs()) == 3


class IterableModule(Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.linears = Module.from_list([Linear(10) for _ in range(self.n)])

    def build(self, x):
        for m in self.linears:
            x = m(x)
        return x


def test_iterable_module_inlined():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 10)), popxl.float32)

        n = 3
        linear = IterableModule(n)
        args, graph = linear.create_graph(x)

        for m in graph.args.linears.values():
            assert m.w in graph.graph
            assert m.b in graph.graph

        layer = graph.bind(args.init())

        layer.call(x)

    assert len(ir._pb_ir.getAllGraphs()) == 2


def test_iterable_module_grad_inlined():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (2, 10)), popxl.float32)

        n = 3
        linear = IterableModule(n)
        args, graph = linear.create_graph(x)
        grad_args, grad_graph = autodiff_with_accumulation(
            graph, graph.args.tensors, grads_required=[graph.graph.inputs[0]]
        )

        for m in graph.args.linears.values():
            assert m.w in graph.graph
            assert m.b in graph.graph

        for m_acc in grad_graph.args.accum.linears.values():
            assert m_acc.w in grad_graph.graph
            assert m_acc.b in grad_graph.graph

        layer = graph.bind(args.init())
        grad_layer = grad_graph.bind(grad_args.init())

        call_info = layer.call_with_info(x)
        y = call_info.outputs[0]

        _ = grad_layer.call(y, args=grad_graph.grad_graph_info.inputs_dict(call_info))

    assert len(ir._pb_ir.getAllGraphs()) == 3
