# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Optional
from typing_extensions import Literal
import os
import logging

import popxl
from popxl import ops

__all__ = ["print_tensor", "print_text"]

logger = logging.getLogger(__name__)

LEVELS_TYPE = Literal["TRACE", "DEBUG", "INFO", "WARN"]
LEVELS = {
    "TRACE": 0,
    "DEBUG": 1,
    "INFO": 2,
    "WARN": 3,
}
MODEL_LOG_LEVEL = os.environ.get("MODEL_LOG_LEVEL", "INFO")
MODEL_LOG_LEVEL_VAL = LEVELS[MODEL_LOG_LEVEL]

if MODEL_LOG_LEVEL_VAL <= LEVELS["DEBUG"]:
    logger.info(f"MODEL_LOG_LEVEL set to: {MODEL_LOG_LEVEL}")


def print_tensor(t: popxl.Tensor, *args, level: LEVELS_TYPE = "DEBUG", **kwargs):
    """Equivalent to `popxl.ops.print_tensor` but can be switched on/off using
    the `MODEL_LOG_LEVEL` environment variable in a similar manner to Python logging.
    """
    level_val = LEVELS[level]
    if MODEL_LOG_LEVEL_VAL <= level_val:
        return ops.print_tensor(t, *args, **kwargs)
    else:
        return t


def print_text(text: str, num: Optional[popxl.Tensor] = None, level: LEVELS_TYPE = "DEBUG"):
    """Print `text` during runtime. Does not work on auto-diff grad graph. Need to be in `in_sequence(True)` to ensure position.
    Use `num` to include a tensor with the text."""
    level_val = LEVELS[level]
    if MODEL_LOG_LEVEL_VAL <= level_val:
        t = num if num else popxl.constant(tuple())
        ops.print_tensor(t, text)
