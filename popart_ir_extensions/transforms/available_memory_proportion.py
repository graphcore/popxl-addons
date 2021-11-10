# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable

import popart.ir as pir
from popart.ir.ops.utils import convert_optional_float

__all__ = ["set_available_memory_proportion_by_ipu"]


def set_available_memory_proportion_by_ipu(ir: pir.Ir, proportions: Iterable[float]):
    """For all ops in the `ir`, if `availableMemoryProportion` can be set, set it as
        specified by `proportions`.
        
        If the the available memory proportion has been set on a op site it will be overridden.

    Args:
        ir (pir.Ir)
        proportions (List[float]): The availableMemoryProportion to be set on each ipu
                                   proportions[N] == 'proportion for ipu N'
    """
    ipu_to_prop = dict(enumerate(proportions))
    for g in ir._pb_ir.getAllGraphs():
        for op in g.getOps():
            if hasattr(op, "setAvailableMemoryProportion"):
                prop = ipu_to_prop[op.getVirtualGraphId()]
                op.setAvailableMemoryProportion(convert_optional_float(prop))
