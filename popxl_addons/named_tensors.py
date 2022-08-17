# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Callable, List, Mapping, Tuple, Dict
import popxl
from popxl import ops
from popxl.context import debug_context_frame_offset
from popxl.tensor import Variable
from popxl_addons.dot_tree import DotTree
import numpy as np

if TYPE_CHECKING:
    from popxl_addons import GraphWithNamedArgs

TensorMap = Mapping[popxl.Tensor, popxl.Tensor]


class NamedTensorData(DotTree[np.ndarray]):
    pass


class NamedTensors(DotTree[popxl.Tensor]):
    """A `DotTree` collection of Tensors"""

    @property
    def tensors(self) -> Tuple[popxl.Tensor, ...]:
        return tuple(sorted(self.to_dict().values(), key=lambda t: t.name))

    @property
    def named_tensors(self) -> Dict[str, popxl.Tensor]:
        return {name: t for name, t in self.to_dict().items()}

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
