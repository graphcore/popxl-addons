# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Union, Tuple
import os
import numpy as np
import logging
from mpi4py import MPI

from more_itertools import peekable
import popxl
import popdist
from popxl import ops, ReplicaGrouping
from popxl import dtypes
from popxl.tensor import HostTensor, host_tensor_types
from popxl.utils import to_numpy

from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.dot_tree import DotTree
from popxl_addons.named_replica_grouping import NamedReplicaGrouping
from popxl_addons.named_tensors import NamedTensors

if TYPE_CHECKING:
    from popxl_addons.remote import NamedRemoteBuffers

host_scalar_tensor_types = tuple([float, int, bool, np.number, np.bool_, *host_tensor_types])

logger = logging.getLogger(__name__)


class VariableFactory:
    def __init__(self,
                 data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                 dtype: Optional[dtypes.dtype] = None,
                 name: Optional[str] = None,
                 by_ref: bool = False,
                 replica_grouping: Optional[ReplicaGrouping] = None,
                 shard_over: int = 1):
        """
        Generates variable tensors for a subgraph from a host tensor data iterator.

        Args:
            data_iter:
                Either a function or iterable that generates data for each instance of the variable tensor. Each element of
                data should be a HostTensor type (numpy, pytorch, ect.) with the same shape and data type (this is not
                checked at runtime). If you want your data to be the same for all tensor instances wrap it in a
                lambda function e.g. `lambda: data`.
            name (str):
                The name of the variable tensor - by default 't'
            by_ref (bool = False):
                If true the graph_input's created for this tensor will be flagged as pass by reference.
            replica_grouping (Optional[ReplicaGrouping]):
                The replica group of the variable. Determines which replicas of the variable will have identical data or
                not when written to. On variable initialisation it will fill a tensor with the replica grouping shape
                `(n_groups, *data_shape)`
            shard_over (int):
                Number of replicas in `replica_grouping` to shard the variable over. Defaults to 1, meaning that the variable is not sharded. If you want to create a replica_sharded_variable, specify shard_over > 1 (to shard over all replicas, `shard_over=replica_grouping.group_size`). See also `add_replica_sharded_variable_input`.
        """

        if callable(data_iter):
            # Test callable
            try:
                data_iter()
            except Exception as e:
                raise ValueError("Passed callable via `data_iter` throws error.") from e

            def data_iter_():
                while True:
                    yield data_iter()

            data_iter_ = data_iter_()
        else:
            data_iter_ = data_iter

        if not isinstance(data_iter_, Iterable):
            raise ValueError("`data_iter` must be an iterable or callable. "
                             "If you want your data to be the same for all tensor instances "
                             "wrap it in a lambda function e.g. `lambda: data`")

        self.data_iter: peekable[HostTensor] = peekable(data_iter_)
        self.name = name
        self.by_ref = by_ref
        self.replica_sharded = shard_over > 1
        self.replica_grouping = replica_grouping or popxl.gcg().ir.replica_grouping()

        data_peek = self.data_iter.peek()
        if not isinstance(data_peek, tuple(host_scalar_tensor_types)):
            raise ValueError(f"`data_iter` must be of a numpy array, torch tensor or iterable. "
                             f"It provided: {data_peek}.")

        self.dtype = dtype or dtypes.dtype.as_dtype(data_peek)

        self.meta_shape = None
        self.shape = data_peek.shape
        if self.replica_sharded:
            self.meta_shape = self.shape
            assert shard_over <= self.replica_grouping.group_size
            self.shape = (int(np.prod(self.meta_shape)) // shard_over, )

    def create_input(self, prefix: Optional[str] = None) -> popxl.Tensor:
        """Create a subgraph input for the current graph."""
        name = self.name if prefix is None else f"{prefix}.{self.name}"
        return popxl.graph_input(shape=self.shape,
                                 dtype=self.dtype,
                                 name=name,
                                 by_ref=self.by_ref,
                                 meta_shape=self.meta_shape)

    def create_tensor(self,
                      name: Optional[str] = None,
                      empty: bool = False,
                      memmap_dir: Optional[str] = None,
                      read_only_if_exists: bool = False):
        """
        Create a new tensor for the current graph.

        Args:
            name: Name of the tensor
            empty: Don't use data and use numpy empty

        Returns:
        """
        name = name or self.name
        if memmap_dir:
            data: HostTensor = self.next_memmap(memmap_dir, name, empty, read_only_if_exists)
        else:
            data: HostTensor = self.next_data(empty)

        return popxl.variable(data, self.dtype, name, replica_grouping=self.replica_grouping)

    def create_remote_tensor(self,
                             buffer: popxl.RemoteBuffer,
                             entry: int,
                             name: Optional[str] = None,
                             empty: bool = False,
                             memmap_dir: Optional[str] = None,
                             read_only_if_exists: bool = False):
        name = name or self.name
        if memmap_dir:
            data: HostTensor = self.next_memmap(memmap_dir, name, empty, read_only_if_exists)
        else:
            data: HostTensor = self.next_data(empty)

        if buffer.meta_shape:
            return popxl.remote_replica_sharded_variable(data,
                                                         buffer,
                                                         entry,
                                                         self.dtype,
                                                         name,
                                                         replica_grouping=self.replica_grouping)
        else:
            return popxl.remote_variable(data, buffer, entry, self.dtype, name, replica_grouping=self.replica_grouping)

    def next_data(self, empty: bool = False) -> HostTensor:
        def next_():
            if not empty:
                return next(self.data_iter)
            else:
                return np.empty(self.meta_shape or self.shape, self.dtype.as_numpy())

        if self.replica_grouping.num_groups == 1:
            return next_()
        else:
            return np.concatenate(
                [to_numpy(next_(), copy=False)[np.newaxis, ...] for _ in range(self.replica_grouping.num_groups)])

    def next_memmap(self, memmap_dir: str, name: str, empty: bool = False, read_only_if_exists: bool = False):
        # Protect against instances racing to create directory
        # (popdist.getInstanceIndex defaults to 0 if popdist.isPopdistEnvSet() == False)
        if popdist.getInstanceIndex() == 0:
            os.makedirs(memmap_dir, exist_ok=True)

        path = os.path.join(memmap_dir, name + ".npy")
        shape = self.meta_shape or self.shape
        if self.replica_grouping.num_groups != 1:
            shape = (self.replica_grouping.num_groups, *shape)

        # Initialise
        # When using poprun, only rank 0 should create the memmap file if it does not already exist
        # All other ranks will wait for this to finish before reading the file
        if not os.path.exists(path) and popdist.getInstanceIndex() == 0:
            logger.debug(f"Creating new memmaped variable file (w+): {path}")
            data = np.memmap(path, dtype=self.dtype.as_numpy(), shape=shape, mode='w+')
            if not empty:
                np.copyto(data, self.next_data(empty))
            if popdist.isPopdistEnvSet():
                MPI.COMM_WORLD.Barrier()
        else:
            if popdist.isPopdistEnvSet():
                MPI.COMM_WORLD.Barrier()
            mode = 'r' if read_only_if_exists else 'r+'
            logger.debug(f"Using existing memmaped variable file ({mode}): {path}")
            data = np.memmap(path, dtype=self.dtype.as_numpy(), shape=shape, mode=mode)

        return data


class NamedVariableFactories(DotTree[VariableFactory]):
    """A `DotTree` collection of VariableFactories """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_zero_graph: Optional[GraphWithNamedArgs] = None
        self._init_zero_names: Optional[List[str]] = None

    def init(self,
             prefix: Optional[str] = None,
             empty: bool = False,
             memmap_dir: Optional[str] = None,
             read_only_if_exists: bool = False) -> NamedTensors:
        """Construct tensors for each VariableFactory.

        The tensors are created in alphabetical order to generate the data deterministically.

        Pseudo example:
        .. code-block:: python
            nif = NamedVariableFactories(a=VariableFactory(lambda: 1), b=VariableFactory(lambda: 2))
            nt = nif.init()
            nt.dict() == {'a': popxl.variable(1), 'b': popxl.variable(2)}

        Args:
            prefix (Optional[str], optional): Prefix the tensor name of the created tensors. Defaults to None.

        Returns:
            NamedTensors: A named Tensor collection. The keys are the same with values being Tensors generated from the
                variable factories.
        """
        inputs = {}
        for name, value in sorted(self.to_dict().items()):
            prefixed = f"{prefix}.{name}" if prefix else name
            inputs[name] = value.create_tensor(prefixed, empty, memmap_dir, read_only_if_exists)
        return NamedTensors.from_dict(inputs)

    def init_remote(self,
                    buffers: "NamedRemoteBuffers",
                    entry: int = 0,
                    prefix: Optional[str] = None,
                    empty: bool = False,
                    memmap_dir: Optional[str] = None,
                    read_only_if_exists: bool = False) -> NamedTensors:
        """Construct remote variables for each VariableFactory using the buffer with a matching name in `buffers`.

        The tensors are created in alphabetical order to generate the data deterministically.

        Args:
            buffers (NamedRemoteBuffers): Buffers to store the variables in.
            entry (int, optional): Entry into the remote buffer to store the variable. Defaults to 0.
            prefix (Optional[str], optional): Prefix the tensor name of the created tensors. Defaults to None.
            empty (bool): If True, create an array of the right shape and dtype with garbage data
            memmap_dir (Optional[str], optional): TODO

        Returns:
            NamedTensors: A named Tensor collection. The keys are the same with values being Tensors generated from the
                variable factories.
        """
        variables = {}
        buffers_ = buffers.to_dict()
        for name, factory in sorted(self.to_dict().items()):
            prefixed = f"{prefix}.{name}" if prefix else name
            variables[name] = factory.create_remote_tensor(buffers_[name], entry, prefixed, empty, memmap_dir,
                                                           read_only_if_exists)
        return NamedTensors.from_dict(variables)

    def init_zero(self) -> NamedTensors:
        """Zero initialise a Tensor using `ops.init` for each VariableFactory in the current Graph scope. (Can be non-main graphs)

        Returns:
            NamedTensors: A named Tensor collection. The keys are the same with values being Tensors generated from the
                variable factories.
        """
        ts = {}
        for name, factory in self.to_dict().items():
            ts[name] = ops.init(factory.shape, factory.dtype, name, "zero")
        return NamedTensors.from_dict(ts)

    def init_undef(self) -> NamedTensors:
        """Undefined initialise a Tensor using `ops.init` for each VariableFactory in the current Graph scope. (Can be non-main graphs)

        Returns:
            NamedTensors: A named Tensor collection. The keys are the same with values being Tensors generated from the
                variable factories.
        """
        ts = {}
        for name, factory in self.to_dict().items():
            ts[name] = ops.init(factory.shape, factory.dtype, name, "undef")
        return NamedTensors.from_dict(ts)

    @property
    def replica_groupings(self) -> NamedReplicaGrouping:
        """DotTree that maps tensor to replica grouping"""
        groups = {name: t.replica_grouping for name, t in self.to_dict().items()}
        return NamedReplicaGrouping.from_dict(groups)


def add_variable_input(
        name: str,
        data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
        dtype: Optional[dtypes.dtype] = None,
        by_ref: bool = False,
        replica_grouping: Optional[ReplicaGrouping] = None,
) -> Tuple[popxl.Tensor, VariableFactory]:
    """Create a VariableFactory and graph_input in the current graph."""
    input_f = VariableFactory(data_iter, dtype, name, by_ref, replica_grouping=replica_grouping)
    tensor = input_f.create_input()
    return tensor, input_f


def add_replica_sharded_variable_input(name: str,
                                       data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                                       dtype: Optional[dtypes.dtype] = None,
                                       by_ref: bool = False,
                                       replica_grouping: Optional[ReplicaGrouping] = None,
                                       shard_over: Optional[int] = None) -> Tuple[popxl.Tensor, VariableFactory]:
    """Create a VariableFactory and replica sharded graph_input in the current graph. 
        If no `replica_grouping` is specified, assume the variable is the same on all replicas. 
        If `shard_over` is not provided, all replicas in replica_grouping will be used for sharding. 
    """
    group = replica_grouping or popxl.gcg().ir.replica_grouping()
    shard_over = shard_over or group.group_size
    input_f = VariableFactory(data_iter, dtype, name, by_ref, replica_grouping, shard_over)
    tensor = input_f.create_input()
    return tensor, input_f
