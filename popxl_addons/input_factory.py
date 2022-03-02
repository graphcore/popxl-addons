# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Callable, Iterable, Optional, Union, Tuple
import numpy as np

from more_itertools import peekable
import popxl
from popxl import dtypes
from popxl.tensor import HostTensor, host_tensor_types
from popxl_addons.dot_tree import DotTree
from popxl_addons.named_tensors import NamedTensors


class InputFactory:
    def __init__(self,
                 data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                 dtype: Optional[dtypes.dtype] = None,
                 name: Optional[str] = None,
                 constant: bool = False,
                 by_ref: bool = False,
                 replica_sharded: bool = False):
        """
        Generates input tensors for a subgraph from a host tensor data iterator.

        Args:
            data_iter:
                Either a function or iterable that generates data for each instance of the input tensor. Each element of
                data should be a HostTensor type (numpy, pytorch, ect.) with the same shape and data type (this is not
                checked at runtime). If you want your data to be the same for all tensor instances wrap it in a
                lambda function e.g. `lambda: data`.
            name (str):
                The name of the input tensor - by default 't'
            constant (bool):
                If false a variable tensor will be generated, otherwise a constant.
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
        self.constant = constant
        self.by_ref = by_ref
        self.dtype = dtype
        self.shape = None
        self.replica_sharded = replica_sharded

        data_peek = self.data_iter.peek()
        if not isinstance(data_peek, tuple(host_tensor_types)):
            raise ValueError(f"`data_iter` must be of a numpy array, torch tensor or iterable. "
                             f"It provided: {data_peek}.")

        if self.replica_sharded:
            self.shape = (int(np.prod(data_peek.shape)) //
                          popxl.gcg().ir._pb_ir.getSessionOptions().replicatedGraphCount, )

    def create_input(self, prefix: Optional[str] = None) -> popxl.Tensor:
        """
        Create a subgraph input for the current graph.
        Peaks at the data iterator's next element to determine data type and shape of the input.
        """
        name = self.name if prefix is None else f"{prefix}.{self.name}"

        data: HostTensor = self.data_iter.peek()
        data = np.array(data)
        shape = self.shape or data.shape
        dtype = self.dtype or dtypes.dtype.as_dtype(data)
        meta_shape = data.shape if shape != data.shape else None
        t = popxl.graph_input(shape=shape, dtype=dtype, name=name, by_ref=self.by_ref, meta_shape=meta_shape)
        return t

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

        if not self.constant:
            return popxl.variable(data, dtype, name=name)
        else:
            return popxl.constant(data, dtype, name=name)


class NamedInputFactories(DotTree[InputFactory]):
    """A `DotTree` collection of InputFactories """

    def init(self, prefix: Optional[str] = None) -> NamedTensors:
        """Construct tensors for each InputFactory.

        Pseudo example:
        .. code-block:: python
            nif = NamedInputFactories(a=InputFactory(lambda: 1), b=InputFactory(lambda: 2))
            nt = nif.init()
            nt.dict() == {'a': popxl.variable(1), 'b': popxl.variable(2)}

        Args:
            prefix (Optional[str], optional): Prefix the tensor name of the created tensors. Defaults to None.

        Returns:
            NamedTensors: A named Tensor collection. The keys are the same with values being Tensors generated from the
                input factories.
        """
        inputs = {}
        for name, value in self.to_dict().items():
            prefixed = f"{prefix}.{name}" if prefix else name
            inputs[name] = value.create_tensor(prefixed)
        return NamedTensors.from_dict(inputs)


def add_input_tensor(name: str,
                     data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                     dtype: Optional[dtypes.dtype] = None,
                     constant: bool = False,
                     by_ref: bool = False) -> Tuple[popxl.Tensor, InputFactory]:
    """Create an InputFactory and graph_input in the current graph."""
    input_f = InputFactory(data_iter, dtype, name, constant, by_ref)
    tensor = input_f.create_input()
    return tensor, input_f


def add_replica_sharded_input_tensor(name: str,
                                     data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                                     dtype: Optional[dtypes.dtype] = None,
                                     constant: bool = False,
                                     by_ref: bool = False) -> Tuple[popxl.Tensor, InputFactory]:
    """Create an InputFactory and replica sharded graph_input in the current graph."""
    input_f = InputFactory(data_iter, dtype, name, constant, by_ref, True)
    tensor = input_f.create_input()
    return tensor, input_f
