# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import re


def suffix_graph_name(name: str, suffix: str):
    removed_subgraph = re.sub(r"_subgraph\(\d+\)", "", name)
    return f"{removed_subgraph}_{suffix}"
