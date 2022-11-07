# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import collections
import re
import logging
import time
import typing
import collections
import typing_extensions
from contextlib import contextmanager

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
    prefix = f'{title} duration' if title else 'Duration'
    logging.info(f'{prefix}: {duration_in_mins:.2f} mins')
    if log_to_wandb and title and WANDB_IMPORTED and wandb.run is not None:
        # `wandb.run` is not None when `wandb.init` been called
        title_ = title.lower().replace(' ', '_') + '_mins'
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
            out = '{\n'
        else:
            out = '{'
        for k, v in self.items():
            out += f'{k}: {v},\n'
        out += '}'
        return out
