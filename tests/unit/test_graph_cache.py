# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
from popxl_addons.graph_cache import GraphCache
from popxl_addons.module import Module


def layer_fn(x: popxl.Tensor, y: popxl.Tensor) -> popxl.Tensor:
    return x + y


class LayerCls(Module):
    def __init__(self, config: int):
        self.config = config

    def build(self, x: popxl.Tensor, y: popxl.Tensor):
        return x + y

    def __hash__(self):
        return hash(self.config)


def test_graph_cache():
    ir = popxl.Ir()
    main = ir.main_graph
    cache = GraphCache()
    with main, popxl.ipu(0):
        x = popxl.variable(1)
        y = popxl.variable(2)

        layer = LayerCls(0)
        g1 = cache.create_graph(layer, x, y)
        g2 = cache.create_graph(layer, x, y)
        assert g1 is g2

        # Different function
        g3 = cache.create_graph(lambda x, y: x + y, x, y)
        assert g1 is not g3

        # Different instance, same hash
        layer0 = LayerCls(0)
        g4 = cache.create_graph(layer0, x, y)
        assert g1 is g4

        # Different instance, different hash
        layer1 = LayerCls(1)
        g5 = cache.create_graph(layer1, x, y)
        assert g1 is not g5

        # Different variables, same spec
        a = popxl.variable(5)
        b = popxl.variable(6)
        g6 = cache.create_graph(layer, a, b)
        assert g1 is g6

        # Different variables, different spec
        j = popxl.variable(7, popxl.float32)
        k = popxl.variable(8, popxl.float32)
        g7 = cache.create_graph(layer, j, k)
        assert g1 is not g7

        # Different IPU
        with popxl.ipu(1):
            g8 = cache.create_graph(layer, x, y)
            assert g1 is not g8
