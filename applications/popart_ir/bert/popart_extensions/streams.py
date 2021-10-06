# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Tuple
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.streams import HostToDeviceStream, DeviceToHostStream


def host_load(shape: Iterable[int], dtype: pir.dtype, name: str) -> Tuple[HostToDeviceStream, pir.Tensor]:
    """Create a HostToDeviceStream and HostLoadOp at the same time."""
    x_h2d = pir.h2d_stream(shape, dtype, name=f"{name}_stream")
    return x_h2d, ops.host_load(x_h2d, name)


def host_store(t: pir.Tensor) -> DeviceToHostStream:
    """Create a DeviceToHostStream and HostStoreOp at the same time."""
    t_d2h = pir.d2h_stream(t.shape, t.dtype, name=f"{t.name}_stream")
    ops.host_store(t_d2h, t)
    return t_d2h
