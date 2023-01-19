# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing_extensions import Literal

import popart._internal.ir as _ir
from popxl import Ir

patterns_level = {
    "no_patterns": _ir.patterns.PatternsLevel.NoPatterns,
    "minimal": _ir.patterns.PatternsLevel.Minimal,
    "default": _ir.patterns.PatternsLevel.Default,
    "all": _ir.patterns.PatternsLevel.All,
}


def apply_pre_alias_patterns(ir: Ir, level: Literal["no_patterns", "minimal", "default", "all"] = "default"):
    """Apply pre-alias patterns to all graphs in the Ir inplace

    Args:
        ir: apply patterns to all graph contained in the `ir`
        level:
            - `no_patterns`: run nothing
            - `minimal`: run only some patterns deemed necessary
            - `default`: run some sensible set of patterns
            - `all`: run all patterns
    """
    level = patterns_level[level]
    ir._pb_ir.setPatterns(_ir.patterns.Patterns(level))
    for g in ir._pb_ir.getAllGraphs():
        ir._pb_ir.applyPreAliasPatterns(g)
