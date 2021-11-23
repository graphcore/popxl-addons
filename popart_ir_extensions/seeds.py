# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Dict, Optional

import numpy as np
import popart.ir as pir
from popart.ir import get_current_graph

import popart_ir_extensions as pir_ext

MAX_UINT32 = 2**32 - 1

__all__ = ["SeedBank"]


class SeedBank:
    def __init__(self, seed: pir.Tensor, offset: int = 0):
        """
        Generates new seeds from a input seed. The outputs are incremented from the input seed.
        This object holds a global index to ensure each random op gets a different seed.
        The `seed` is also automatically added as a input to a graph if required.

        Args:
            seed: parent seed
            offset: can offset seed generation
        """
        cg = get_current_graph()
        mg = cg.ir().main_graph()

        if mg != cg:
            raise Exception("Please initialise the SeedBank in the main graph.")
        if seed not in cg:
            raise Exception("`seed` should be a member of the main graph.")
        self.seed = seed
        self.subgraph_tensors: Dict[pir.Graph, pir.Tensor] = {mg: self.seed}
        self.i = offset

    def next(self, graph: Optional['pir_ext.GenericGraph'] = None) -> pir.Tensor:
        """
        Generate a new seed.
        Manges adding the parent seed to the current graph as a input.

        Args:
            graph: Current `GenericGraph`

        Returns:
            pir.Tensor: new seed generated from parent seed
        """
        cg = get_current_graph()
        mg = cg.ir().main_graph()

        if cg == mg:
            seed_parent = self.seed
        elif graph is not None:
            if cg not in self.subgraph_tensors:
                # Each subgraph has the same seed input and increments it with a different random number
                seed_cg = graph.add_static_input_tensor(self.seed.name, self.seed)
                seed_inc = graph.add_input_tensor('seed_inc',
                                                  lambda: np.random.randint(0, MAX_UINT32, (2, ), dtype='uint32'),
                                                  constant=True)
                self.subgraph_tensors[cg] = seed_cg + seed_inc
            seed_parent = self.subgraph_tensors[cg]
        else:
            ValueError("You must call `next` either in the main graph or pass the `GenericGraph` you are "
                       "currently working with")

        self.i += 1
        seed = seed_parent + self.i
        return seed

    def _next(self, seed_parent) -> pir.Tensor:
        self.i += 1
        seed = seed_parent + self.i
        return seed
