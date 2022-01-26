# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart.ir as pir
from popart_ir_extensions.graph_cache import GraphCache
from popart_ir_extensions.module import Module


def layer_fn(x: pir.Tensor, y: pir.Tensor) -> pir.Tensor:
    return x + y


class LayerCls(Module):
    def __init__(self, config: int):
        self.config = config

    def build(self, x: pir.Tensor, y: pir.Tensor):
        return x + y

    def __hash__(self):
        return hash(self.config)


def test_graph_cache():
    ir = pir.Ir()
    main = ir.main_graph()
    cache = GraphCache()
    with main, pir.ipu(0):
        x = pir.variable(1)
        y = pir.variable(2)

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
        a = pir.variable(5)
        b = pir.variable(6)
        g6 = cache.create_graph(layer, a, b)
        assert g1 is g6

        # Different variables, different spec
        j = pir.variable(7, pir.float32)
        k = pir.variable(8, pir.float32)
        g7 = cache.create_graph(layer, j, k)
        assert g1 is not g7

        # Different IPU
        with pir.ipu(1):
            g8 = cache.create_graph(layer, x, y)
            assert g1 is not g8
