# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Tuple

import popxl
from popxl import ops
from popxl.streams import HostToDeviceStream, DeviceToHostStream
from popxl.tensor import HostTensor
from popxl.utils import to_numpy

__all__ = ["host_load", "host_store"]


def host_load(t: HostTensor, dtype: popxl.dtype, name: str) -> Tuple[HostTensor, HostToDeviceStream, popxl.Tensor]:
    """Create a HostToDeviceStream and HostLoadOp at the same time."""
    x_h2d = popxl.h2d_stream(t.shape, dtype, name=f"{name}_stream")
    return to_numpy(t, dtype=dtype), x_h2d, ops.host_load(x_h2d, name)


def host_store(t: popxl.Tensor) -> DeviceToHostStream:
    """Create a DeviceToHostStream and HostStoreOp at the same time."""
    t_d2h = popxl.d2h_stream(t.shape, t.dtype, name=f"{t.name}_stream")
    ops.host_store(t_d2h, t)
    return t_d2h
