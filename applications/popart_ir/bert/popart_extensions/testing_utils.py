# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Type

from popart._internal import ir as _ir


def ops_of_type(ops: Iterable[_ir.Op], op_type: Type[_ir.Op]) -> int:
    return len(list(filter(lambda op: isinstance(op, op_type), ops)))
