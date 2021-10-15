# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Union
import numpy as np

import torch

HostTensor = Union[np.ndarray, torch.Tensor]


def to_numpy(a: HostTensor) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a.copy()
    if isinstance(a, torch.Tensor):
        return a.detach().numpy().copy()
    else:
        raise ValueError(f"Do not recognise type: {a}")
