# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popxl

__all__ = ["disable_fc_pass", "enable_fc_pass"]


def disable_fc_pass(ir: popxl.Ir):
    """Transform to disable automatic transposition of weights in matmuls.  This 
    is enabled by default as transposed weights are typically used in the backwards 
    graph, so are prepared in advanced. However, for inference-only, this consumes 
    extra memory and can be disabled with `disable_fc_pass`.

    Args:
        ir (popxl.Ir)
    """
    for g in ir._pb_ir.getAllGraphs():
        for op in g.getOps():
            if hasattr(op, "useFullyConnectedPass"):
                op.setUseFullyConnectedPass(False)


def enable_fc_pass(ir: popxl.Ir):
    """Transform to renable automatic weight transposition in matmuls after a prior
    call to `disable_fc_pass`. See `disable_fc_pass` for further detail.

    Args:
        ir (popxl.Ir)
    """
    for g in ir._pb_ir.getAllGraphs():
        for op in g.getOps():
            if hasattr(op, "useFullyConnectedPass"):
                op.setUseFullyConnectedPass(True)
