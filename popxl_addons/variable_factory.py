# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Union, Tuple
import numpy as np

from more_itertools import peekable
import popxl
from popxl import ops
from popxl import dtypes
from popxl.tensor import HostTensor, host_tensor_types
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.dot_tree import DotTree
from popxl_addons.named_tensors import NamedTensors

if TYPE_CHECKING:
    from popxl_addons.transforms.phased import NamedRemoteBuffers


class VariableFactory:
    def __init__(self,
                 data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                 dtype: Optional[dtypes.dtype] = None,
                 name: Optional[str] = None,
                 by_ref: bool = False,
                 replica_sharded: bool = False):
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
        self.replica_sharded = replica_sharded

        data_peek = self.data_iter.peek()
        if not isinstance(data_peek, tuple(host_tensor_types)):
            raise ValueError(f"`data_iter` must be of a numpy array, torch tensor or iterable. "
                             f"It provided: {data_peek}.")

        self.dtype = dtype or dtypes.dtype.as_dtype(data_peek)

        if self.replica_sharded:
            self.meta_shape = data_peek.shape
            self.shape = (int(np.prod(self.meta_shape)) // popxl.gcg().ir.replication_factor, )
        else:
            self.meta_shape = None
            self.shape = data_peek.shape

    def create_input(self, prefix: Optional[str] = None) -> popxl.Tensor:
        """Create a subgraph input for the current graph."""
        name = self.name if prefix is None else f"{prefix}.{self.name}"
        return popxl.graph_input(shape=self.shape,
                                 dtype=self.dtype,
                                 name=name,
                                 by_ref=self.by_ref,
                                 meta_shape=self.meta_shape)

    def create_tensor(self, name: Optional[str] = None):
        """
        Create a new tensor for the current graph.

        Args:
            name: Name of the tensor

        Returns:
        """
        name = name or self.name
        data: HostTensor = next(self.data_iter)
        dtype = self.dtype or None

        return popxl.variable(data, dtype, name)

    def create_remote_tensor(self, buffer: popxl.RemoteBuffer, entry: int, name: Optional[str] = None):
        name = name or self.name
        data: HostTensor = next(self.data_iter)
        dtype = self.dtype or None

        if buffer.meta_shape:
            return popxl.remote_replica_sharded_variable(data, buffer, entry, dtype, name)
        else:
            return popxl.remote_variable(data, buffer, entry, dtype, name)


class NamedVariableFactories(DotTree[VariableFactory]):
    """A `DotTree` collection of VariableFactories """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_zero_graph: Optional[GraphWithNamedArgs] = None
        self._init_zero_names: Optional[List[str]] = None

    def init(self, prefix: Optional[str] = None) -> NamedTensors:
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
            inputs[name] = value.create_tensor(prefixed)
        return NamedTensors.from_dict(inputs)

    def init_remote(self, buffers: "NamedRemoteBuffers", entry: int = 0, prefix: Optional[str] = None) -> NamedTensors:
        """Construct remote variables for each VariableFactory using the buffer with a matching name in `buffers`.

        The tensors are created in alphabetical order to generate the data deterministically.

        Args:
            buffers (NamedRemoteBuffers): Buffers to store the variables in.
            entry (int, optional): Entry into the remote buffer to store the variable. Defaults to 0.
            prefix (Optional[str], optional): Prefix the tensor name of the created tensors. Defaults to None.

        Returns:
            NamedTensors: A named Tensor collection. The keys are the same with values being Tensors generated from the
                variable factories.
        """
        variables = {}
        buffers_ = buffers.to_dict()
        for name, factory in sorted(self.to_dict().items()):
            prefixed = f"{prefix}.{name}" if prefix else name
            variables[name] = factory.create_remote_tensor(buffers_[name], entry, prefixed)
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


def add_variable_input(name: str,
                       data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                       dtype: Optional[dtypes.dtype] = None,
                       by_ref: bool = False) -> Tuple[popxl.Tensor, VariableFactory]:
    """Create a VariableFactory and graph_input in the current graph."""
    input_f = VariableFactory(data_iter, dtype, name, by_ref)
    tensor = input_f.create_input()
    return tensor, input_f


def add_replica_sharded_variable_input(name: str,
                                       data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                                       dtype: Optional[dtypes.dtype] = None,
                                       by_ref: bool = False) -> Tuple[popxl.Tensor, VariableFactory]:
    """Create a VariableFactory and replica sharded graph_input in the current graph."""
    input_f = VariableFactory(data_iter, dtype, name, by_ref, True)
    tensor = input_f.create_input()
    return tensor, input_f
