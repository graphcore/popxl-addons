# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Optional, Tuple
import popart._internal.ir as _ir
import popxl
from popxl import ops
from popxl.tensor import Tensor, Variable
from popxl.context import _execution_context

__all__ = ["seed_variable"]


def seed_variable(seed: int, replica_grouping: Optional[popxl.ReplicaGrouping] = None) -> Tuple[Variable, Tensor]:
    """Initialise a seed variable.
        Similar to `replica_sharded_variable` this method returns two Tensors.
        The first is a Variable that can be used as a reference with popxl.Session to get/set the current seed value.
        The second is a Tensor that should be passed to other operations in the main_graph.

        Note: this is a workaround to enable compatibility between onchip variables and ReplicaGrouping 
              when using distributed replication"""
    replica_grouping = replica_grouping or popxl.gcg().ir.replica_grouping(group_size=1)
    initial_seed_values = popxl.create_seeds(seed, replicas=replica_grouping.num_groups)
    buffer = popxl.remote_buffer((2, ), popxl.uint32)
    seed_var = popxl.remote_variable(initial_seed_values, buffer, replica_grouping=replica_grouping, name="random_seed")

    with popxl.gmg():
        with _execution_context(_ir.ExecutionContext.WeightsFromHostFragment):
            seed_t = ops.remote_load(buffer, 0, "random_seed_h2d")

        with _execution_context(_ir.ExecutionContext.WeightsToHostFragment):
            ops.remote_store(buffer, 0, seed_t)

    return seed_var, seed_t
