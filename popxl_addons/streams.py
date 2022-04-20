# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Tuple, Union, Iterable, TypeVar, Generic
from collections import OrderedDict
from popxl import HostToDeviceStream, DeviceToHostStream
import popxl

__all__ = ["InputStreams", "OutputStreams"]

StreamType = TypeVar('StreamType', bound=popxl.streams.Stream)


class _StreamsBase(OrderedDict, Generic[StreamType]):
    def __init__(self, **streams: Tuple[Tuple[int, ...], popxl.dtype]):
        super().__init__()
        for name, (shape, dtype) in streams.items():
            self[name] = self.stream_init(shape=shape, dtype=dtype, name=name)

    @classmethod
    def from_streams(cls, streams: Iterable[StreamType]):
        self = cls()
        for stream in streams:
            self[stream.tensor_id] = stream
        return self

    def __getattr__(self, key: str) -> StreamType:
        try:
            return super().__getitem__(key)
        except KeyError as ae:
            keys = "    \n".join(self.keys())
            raise AttributeError(f"No attribute '{key}'. Available Keys:\n{keys}") from ae

    def __getitem__(self, key: Union[str, int, slice]) -> StreamType:
        if isinstance(key, (int, slice)):
            return list(self.values())[key]
        try:
            return super().__getitem__(key)
        except KeyError as e:
            keys = "    \n".join(self.keys())
            raise KeyError(f"No key '{key}'. Available Keys:\n{keys}") from e

    def __iter__(self):
        return iter(self.values())


class InputStreams(_StreamsBase[HostToDeviceStream]):
    """Helper class to manage `popxl.h2d_streams`.

    Usage:
    .. code-block:: python
        inputs = InputStreams(x=((2, 4), popxl.float32))
        ...
        x = ops.host_load(inputs.x)
    """
    stream_init = staticmethod(popxl.h2d_stream)


class OutputStreams(_StreamsBase[DeviceToHostStream]):
    """Helper class to manage `popxl.d2h_streams`

    Usage:
    .. code-block:: python
        outputs = OutputStreams(y=((2, 4), popxl.float32))
        ...
        y = ...
        ops.host_store(outputs.y, y)
    """
    stream_init = staticmethod(popxl.d2h_stream)
