# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import re
from contextlib import contextmanager


# Backported to python3.6
@contextmanager
def null_context():
    yield


def suffix_graph_name(name: str, suffix: str):
    removed_subgraph = re.sub(r"_subgraph\(\d+\)", "", name)
    return f"{removed_subgraph}_{suffix}"
