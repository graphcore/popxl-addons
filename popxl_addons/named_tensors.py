# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Callable, List, Mapping, Tuple, Dict
import popxl
from popxl import ops
from popxl.context import debug_context_frame_offset
from popxl.tensor import Variable
from popxl_addons.dot_tree import DotTree

if TYPE_CHECKING:
    from popxl_addons import GraphWithNamedArgs

TensorMap = Mapping[popxl.Tensor, popxl.Tensor]


class NamedTensors(DotTree[popxl.Tensor]):
    """A `DotTree` collection of Tensors"""

    @property
    def tensors(self) -> Tuple[popxl.Tensor, ...]:
        return tuple(t for t in self.to_dict().values())

    @property
    def named_tensors(self) -> Dict[str, popxl.Tensor]:
        return {name: t for name, t in self.to_dict().items()}

    @property
    def variables(self) -> Tuple[popxl.Tensor, ...]:
        return tuple(t for t in self.tensors if isinstance(t, Variable))

    @property
    def named_variables(self) -> Dict[str, popxl.Tensor]:
        return {name: t for name, t in self.named_tensors.items() if isinstance(t, Variable)}

    def remap(self, mapping: TensorMap):
        """Map each value to a new value from `mapping`. `mapping` must contain all keys in self."""
        remapped = self.copy()
        for attr, item in self._map.items():
            if isinstance(item, popxl.Tensor):
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
