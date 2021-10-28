# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart.ir as pir
import pytest
from popart.ir import ops

import popart_extensions as pir_ext
from popart_extensions.seeds import SeedBank


class ScaleAndDrop(pir_ext.GenericGraph):
    def __init__(self, seed_bank: SeedBank):
        super().__init__()
        self.seed_bank = seed_bank

    def build(self, x: pir.Tensor) -> pir.Tensor:
        scale = self.add_input_tensor("scale", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        x = x * scale
        x = ops.dropout(x, self.seed_bank.next(self), p=0.5)
        return x


def test_seed_bank_init():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        seed = pir.constant(np.array([0, 0]), dtype=pir.uint32, name='seed')

    # Must create init in graph
    with pytest.raises(Exception):
        SeedBank(seed)

    with main:
        # Must init in main graph (not subgraph)
        sg = ir.create_empty_graph("sg1")
        with sg:
            with pytest.raises(Exception):
                SeedBank(seed)

        # Ok
        seed_bank = SeedBank(seed)


def test_seed_bank_next():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        seed = pir.constant(np.array([0, 0]), dtype=pir.uint32, name='seed')
        x = pir.constant(0, dtype=pir.uint32, name='seed')
        seed_bank = SeedBank(seed)

        # Generate seed in main graph
        seed_1 = seed_bank.next()
        assert seed_bank.i == 1

        # Call and output
        y = ScaleAndDrop(seed_bank)(x)
        assert seed_bank.i == 2

        # Create concrete graph
        concrete_graph = ScaleAndDrop(seed_bank).to_concrete(x)
        assert concrete_graph.seed == seed
        assert seed_bank.i == 3

        # Create callable graph
        callable_graph_1 = concrete_graph.to_callable(create_inputs=True)
        assert callable_graph_1.seed == seed
        assert seed_bank.i == 3

        callable_graph_2 = concrete_graph.to_callable(create_inputs=True)
        assert callable_graph_2.seed == seed
        assert seed_bank.i == 3

        # Have the same base seed but different increments
        assert callable_graph_1.seed == callable_graph_2.seed
        assert callable_graph_1.seed_inc != callable_graph_2.seed_inc

        callable_graph_2.call(x)
