# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Mapping, Tuple, Union, Dict
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.context import debug_context_frame_offset
from popart.ir.tensor import Variable
from popart_ir_extensions.dot_tree import DotTree

TensorMap = Mapping[pir.Tensor, pir.Tensor]


class NamedTensors(DotTree[pir.Tensor]):
    """A `DotTree` collection of Tensors"""

    @property
    def tensors(self) -> Tuple[pir.Tensor, ...]:
        return tuple(t for t in self.to_dict().values())

    @property
    def named_tensors(self) -> Dict[str, pir.Tensor]:
        return {name: t for name, t in self.to_dict().items()}

    @property
    def variables(self) -> Tuple[pir.Tensor, ...]:
        return tuple(t for t in self.tensors if isinstance(t, Variable))

    @property
    def named_variables(self) -> Dict[str, pir.Tensor]:
        return {name: t for name, t in self.named_tensors.items() if isinstance(t, Variable)}

    def remap(self, mapping: TensorMap):
        """Map each value to a new value from `mapping`. `mapping` must contain all keys in self."""
        remapped = self.copy()
        for attr, item in self._map.items():
            if isinstance(item, pir.Tensor):
                remapped.insert(attr, mapping[item], True)
            else:
                remapped.insert(attr, item.remap(mapping), True)
        return remapped


@debug_context_frame_offset(1)
def print_named_tensors(named_tensors: NamedTensors) -> NamedTensors:
    printed = {}
    for key, tensor in named_tensors.to_dict().items():
        printed[key] = ops.print_tensor(tensor, key)
    return NamedTensors.from_dict(printed)
