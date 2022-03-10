# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
import numpy as np
from popxl.tensor import HostTensor

try:
    import torch
    torch_imported = True
except ModuleNotFoundError:
    torch_imported = False


def to_numpy(x: HostTensor, dtype=None) -> np.ndarray:
    if torch_imported and isinstance(x, torch.Tensor):
        x = x.detach().numpy().copy()
        if dtype:
            x = x.astype(dtype)
    else:
        x = np.array(x, dtype=dtype)

    return x


def suffix_graph_name(name: str, suffix: str):
    removed_subgraph = re.sub(r"_subgraph\(\d+\)", "", name)
    return f"{removed_subgraph}_{suffix}"
