# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from popxl.tensor import HostTensor

try:
    import torch
    torch_imported = True
except ModuleNotFoundError:
    torch_imported = False


def to_numpy(x: HostTensor, dtype=None) -> np.ndarray:
    if torch_imported and isinstance(x, torch.Tensor):
        x = x.detach().numpy()
        if dtype:
            x = x.astype(dtype)
    else:
        x = np.array(x, dtype=dtype)

    return x.copy()
