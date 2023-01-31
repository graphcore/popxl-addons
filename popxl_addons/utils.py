# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import collections
import re
import logging
import time
import typing
import collections
import typing_extensions
from contextlib import contextmanager
from popxl.tensor import Variable
from typing import MutableMapping
import numpy as np
import popxl

WANDB_IMPORTED = False
try:
    import wandb

    WANDB_IMPORTED = True
except ImportError:
    pass


# Backported to python3.6
@contextmanager
def null_context():
    yield


def suffix_graph_name(name: str, suffix: str):
    removed_subgraph = re.sub(r"_subgraph\(\d+\)", "", name)
    return f"{removed_subgraph}_{suffix}"


@contextmanager
def timer(title=None, log_to_wandb=True):
    """Timer util which times contents in a context and logs the value to W&B (if `log_to_wandb==True`)

    Example:
    ```
    with timer('Process X'):
        long_process()
    ```
    """
    t = time.perf_counter()
    logging.info(f'Starting {title or ""}')
    yield
    duration_in_mins = (time.perf_counter() - t) / 60
    prefix = f"{title} duration" if title else "Duration"
    logging.info(f"{prefix}: {duration_in_mins:.2f} mins")
    if log_to_wandb and title and WANDB_IMPORTED and wandb.run is not None:
        # `wandb.run` is not None when `wandb.init` been called
        title_ = title.lower().replace(" ", "_") + "_mins"
        wandb.run.summary[title_] = duration_in_mins


_KT = typing.TypeVar("_KT")  # key type
_VT = typing.TypeVar("_VT")  # value type


class OrderedDict(collections.OrderedDict, typing.MutableMapping[_KT, _VT]):
    """An OrderedDict with a nicer string representation and access via index."""

    def idx(self, idx: int) -> _VT:
        """Return value at index `idx`"""
        return list(self.values())[idx]

    def __repr__(self):
        if len(self):
            out = "{\n"
        else:
            out = "{"
        for k, v in self.items():
            out += f"{k}: {v},\n"
        out += "}"
        return out


class WeightsDict(MutableMapping[Variable, np.ndarray]):
    def __init__(self, *args, **kwargs):
        """A dictionary that maps `popxl.Variable`s to `numpy.ndarry`s and validates the corresponding items have
        the equivalent dtypes and shape."""
        super().__init__()
        self.dict = dict()
        if len(args) + len(kwargs) > 0:
            self.update(*args, **kwargs)

    def __getitem__(self, key: Variable) -> np.ndarray:
        return self.dict.__getitem__(key)

    def __setitem__(self, key: Variable, value: np.ndarray):
        # Validate
        value_dtype = popxl.dtype.as_dtype(value.dtype)
        if key.dtype != value_dtype:
            raise ValueError(f"{key}: tensor and weight dtypes do not equal {key.dtype} != {value_dtype}")
        if key.shape_on_host != value.shape:
            raise ValueError(f"{key}: tensor and weight shapes do not equal {key.shape_on_host} != {value.shape}")

        return self.dict.__setitem__(key, value)

    def __delitem__(self, key: Variable):
        return self.dict.__delitem__(key)

    def __len__(self):
        return self.dict.__len__()

    def __iter__(self):
        return self.dict.__iter__()
