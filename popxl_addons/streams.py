# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Tuple
from collections import OrderedDict

import popxl

__all__ = ["InputStreams", "OutputStreams"]


class InputStreams(OrderedDict):
    def __init__(self, **streams: Tuple[Tuple[int, ...], popxl.dtype]):
        """Helper class to manage popxl.h2d_streams.

            Usage:
                inputs = InputStreams(x=((2, 4), popxl.float32))
                ...
                x = ops.host_load(inputs.x)
        """
        super().__init__()
        for name, args in streams.items():
            self[name] = popxl.h2d_stream(*args, name=name)

    def __getattr__(self, __name: str) -> popxl.HostToDeviceStream:
        try:
            return super().__getitem__(__name)
        except AttributeError as ae:
            keys = "    \n".join(self.keys())
            raise AttributeError(f"No attribute '{__name}'. Available Keys:\n{keys}") from ae


class OutputStreams(OrderedDict):
    def __init__(self, **streams: Tuple[Tuple[int, ...], popxl.dtype]):
        """Helper class to manage popxl.d2h_streams

            Usage:
                outputs = OutputStreams(y=((2, 4), popxl.float32))
                ...
                y = ...
                ops.host_store(outputs.y, y)
        """
        super().__init__()
        for name, args in streams.items():
            self[name] = popxl.d2h_stream(*args, name=name)

    def __getattr__(self, __name: str) -> popxl.DeviceToHostStream:
        try:
            return super().__getitem__(__name)
        except AttributeError as ae:
            keys = "    \n".join(self.keys())
            raise AttributeError(f"No attribute '{__name}'. Available Keys:\n{keys}") from ae
