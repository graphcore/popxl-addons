# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import popart.ir as pir
from popart.ir.context import get_current_context
from popart.ir.tensor import Variable, remote_replica_sharded_variable, remote_variable

__all__ = ["RemoteBuffers"]


class RemoteBuffers:
    """Container to track buffer entries. 
        New buffer can be created by using method `get_buffer`. This will reuse buffers where possible.
       Remote variable can also be constructed to be stored in the buffers
    """

    def __init__(self):
        self.buffers: Dict[int, pir.RemoteBuffer] = {}
        self.hash_to_id: Dict[int, int] = {}

    def get_buffer(self, shape: Iterable[int], dtype: pir.dtype,
                   meta_shape: Optional[Iterable[int]] = None) -> pir.RemoteBuffer:
        """Fetch or create a buffer for the current virtual graph with a specified shape, dtype, meta_shape.

        Args:
            shape (Iterable[int]): Shape of the entries of the Buffer
            dtype (pir.dtype): Datatype of the entries of the Buffer
            meta_shape (Optional[Iterable[int]], optional): Meta shape of the entries of the Buffer. Defaults to None.
        """
        shape = tuple(shape)
        meta_shape = tuple(meta_shape) if meta_shape else None
        buffer_hash = hash((get_current_context().ipu_id, shape, dtype, meta_shape))
        if buffer_hash not in self.hash_to_id:
            buffer = pir.remote_buffer(tuple(shape), dtype, 1)
            self.hash_to_id[buffer_hash] = buffer.remote_buffer_id
            self.buffers[buffer.remote_buffer_id] = buffer
        return self.buffers[self.hash_to_id[buffer_hash]]

    def remote_variable(self, var: Variable) -> Tuple[pir.RemoteBuffer, int]:
        """Set `var` as a remote Variable, with storage in a new entry in the remote buffers.

        Args:
            var (Variable): Variable to be marked as remote

        Returns:
            Tuple[pir.RemoteBuffer, int]: RemoteBuffer and offset of the remote variable storage
        """
        buffer = self.get_buffer(var.shape, var.dtype)
        offset = buffer.entries
        buffer.entries += 1
        remote_variable(var, buffer, offset)
        return buffer, offset

    def replica_sharded_variable(self, var: Variable) -> Tuple[pir.RemoteBuffer, int]:
        """Set `var` as a remote replica sharded Variable, with storage in a new entry in the remote buffers.

        Args:
            var (Variable): Variable to be marked as remote

        Returns:
            Tuple[pir.RemoteBuffer, int]: RemoteBuffer and offset of the remote variable storage
        """
        shape = (int(np.prod(var.shape)) // pir.gcg().ir._pb_ir.getSessionOptions().replicatedGraphCount, )
        buffer = self.get_buffer(shape, var.dtype, var.shape)
        offset = buffer.entries
        buffer.entries += 1
        remote_replica_sharded_variable(var, buffer, offset)
        return buffer, offset
