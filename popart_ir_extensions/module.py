# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Tuple, Union, List
import numpy as np
import popart.ir as pir
from popart.ir import dtypes
from popart_ir_extensions.graph_cache import GraphCache
from popart_ir_extensions.named_tensors import NamedTensors
from popart_ir_extensions.input_factory import NamedInputFactories, add_input_tensor, add_replica_sharded_input_tensor, \
    InputFactory
from popart_ir_extensions.graph import GraphWithNamedArgs
from popart.ir.tensor import HostTensor, host_tensor_types


class NameScopeMeta(type):
    """Meta class to wrap `build` methods with a pir.name_scope"""

    def __new__(cls, name, bases, dct):
        build_fn = dct.get("build", None)
        if build_fn is not None and callable(build_fn):

            @wraps(build_fn)
            def wrapper(self, *args, **kwargs):
                scope = getattr(self, "_name_scope", None)
                if scope is not None:
                    with pir.name_scope(scope):
                        return build_fn(self, *args, **kwargs)
                return build_fn(self, *args, **kwargs)

            dct["build"] = wrapper
        return super().__new__(cls, name, bases, dct)


class Module(pir.Module, metaclass=NameScopeMeta):
    """Module class to allow construction of compute graphs that require state.
        TODO: expand, usage
    """

    def __init__(self, cache: bool = False):
        """
        Args:
            cache (bool, optional): Re-use graphs where possible when calling `create_graph`. Defaults to False.
        """
        self._input_factories = NamedInputFactories()
        self._named_inputs = NamedTensors()
        self._graph_cache = GraphCache() if cache else None
        self._args_cache = {}

    def _reset(self):
        self._input_factories._clear()
        self._named_inputs._clear()

    def create_graph(self, *args, **kwargs) -> Tuple[NamedInputFactories, GraphWithNamedArgs]:
        """Construct a compute graph and input factories for the module."""
        if self._graph_cache is not None:
            graph = self._graph_cache.create_graph(self, *args, **kwargs)
            if graph not in self._args_cache:
                self._args_cache[graph] = self._input_factories.copy(), self._named_inputs.copy()
            input_factories, named_inputs = self._args_cache[graph]
        else:
            graph = pir.gcg().ir.create_graph(self, *args, **kwargs)
            input_factories, named_inputs = self._input_factories, self._named_inputs
        result = (input_factories.copy(), GraphWithNamedArgs(graph, named_inputs.copy()))
        self._reset()
        return result

    def add_input_factory(self, input_f: InputFactory, name: Optional[str] = None) -> pir.Tensor:
        """Add an input factory as an input tensor.

        Args:
            input_f (InputFactory): Input factory to generate an input tensor
            name (str): Name for input. If provided it will override the name provided to the input factory
        """
        tensor = input_f.create_input()
        name = name if name else input_f.name
        self._input_factories.insert(name, input_f)
        self._named_inputs.insert(name, tensor)
        return tensor

    def add_input_tensor(self,
                         name: str,
                         data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                         dtype: Optional[dtypes.dtype] = None,
                         constant: bool = False,
                         by_ref: bool = False) -> pir.Tensor:
        """Add an initialised input tensor.

        Args:
            name (str): named of the input Tensor. Used for both the Named InputFactory/Arg and the Tensor debug name.
            data_iter (Union[Callable[[None], np.ndarray], Iterable[np.ndarray]]): 
                Either a function or iterable that generates data for each instance of the input tensor. Each element of
                data should be a HostTensor type (numpy, pytorch, ect.) with the same shape and data type (this is not
                checked at runtime). If you want your data to be the same for all tensor instances wrap it in a
                lambda function e.g. `lambda: data`.
            dtype (Optional[dtypes.dtype], optional): dtype of input. Defaults to None.
            constant (bool, optional): Construct input as constant when initialised. Defaults to False.
            by_ref (bool, optional): Pass the input by reference. Defaults to False.
        """
        tensor, input_f = add_input_tensor(name, data_iter, dtype, constant, by_ref)
        self._input_factories.insert(name, input_f)
        self._named_inputs.insert(name, tensor)
        return tensor

    def add_replica_sharded_input_tensor(self,
                                         name: str,
                                         data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
                                         dtype: Optional[dtypes.dtype] = None,
                                         constant: bool = False,
                                         by_ref: bool = False) -> pir.Tensor:
        """Add a replica sharded initialised input tensor. The graph input will be sharded, however the initialised Tensor will
            be constructed whole.
            `remote_replica_sharded_variable` or `ops.collectives.replicated_reduce_scatter(..., configure_output_for_replicated_tensor_sharding=True)`
            should be used to shard the Tensor before passing to the graph call.

        Args:
            name (str): named of the input Tensor. Used for both the Named InputFactory/Arg and the Tensor debug name.
            data_iter (Union[Callable[[None], np.ndarray], Iterable[np.ndarray]]): 
                Either a function or iterable that generates data for each instance of the input tensor. Each element of
                data should be a HostTensor type (numpy, pytorch, ect.) with the same shape and data type (this is not
                checked at runtime). If you want your data to be the same for all tensor instances wrap it in a
                lambda function e.g. `lambda: data`.
            dtype (Optional[dtypes.dtype], optional): dtype of input. Defaults to None.
            constant (bool, optional): Construct input as constant when initialised. Defaults to False.
            by_ref (bool, optional): Pass the input by reference. Defaults to False.
        """
        tensor, input_f = add_replica_sharded_input_tensor(name, data_iter, dtype, constant, by_ref)
        self._input_factories.insert(name, input_f)
        self._named_inputs.insert(name, tensor)
        return tensor

    def set_name_scope(self, name: str):
        setattr(self, "_name_scope", name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if hasattr(__value, "set_name_scope"):
            __value.set_name_scope(__name)
        if hasattr(__value, "_input_factories"):
            self._input_factories.insert(__name, __value._input_factories)
        if hasattr(__value, "_named_inputs"):
            self._named_inputs.insert(__name, __value._named_inputs)
        return super().__setattr__(__name, __value)

    def add_inputs(self, name: Union[str, int], inputs: NamedInputFactories) -> NamedTensors:
        """Add NamedInputFactories as inputs tensors. Returns NamedTensors which can be used in the graph.
            This is useful for reusing a Module within another module. Usage:
            ```
                def build(self, x: pir.Tensor):
                    args, graph = self.layer.create_graph(x)
                    layer1 = graph.bind(self.add_inputs("1", args))
                    layer2 = graph.bind(self.add_inputs("2", args))
            ```
            this then provides two BoundGraphs `layer1/layer2` which can be called. With each BoundGraph calling the same
            compute Graph but with unique inputs. These inputs being constructed from the `args` NamedInputFactories.

        Args:
            name (str): name of the input factories used in the current module.
                If a positive int, it will be cast to a string
            inputs (NamedInputFactories): Input factories to be added to the current module.

        Returns:
            NamedTensors: Tensors in the current graph for each input factory in `inputs`.
        """
        if isinstance(name, int):
            if name < 0:
                raise ValueError(f'Name of int type must be positive. Value: {name}')
            name = str(name)
        tensors = {}
        for tensor_name, factory in inputs.to_dict().items():
            tensors[tensor_name] = factory.create_input()
        tensors = NamedTensors.from_dict(tensors)
        self._input_factories.insert(name, inputs.copy())
        self._named_inputs.insert(name, tensors)
        return tensors

    @classmethod
    def from_list(cls, sub_modules: List['Module']):
        """Create a module from a list of modules"""
        self = cls()
        for i, module in enumerate(sub_modules):
            setattr(self, str(i), module)
        return self

    @classmethod
    def from_input_factories(cls, inputs: List[InputFactory]):
        """Create a module with `inputs`"""
        self = cls()
        for i, input_f in enumerate(inputs):
            tensor = self.add_input_factory(input_f, name=str(i))
            setattr(self, str(i), tensor)
        return self

    def get(self, key: Union[str, int]):
        """Get attribute. If an int it will be cast to a string."""
        key = str(key) if isinstance(key, int) else key
        return getattr(self, key)

    def __getitem__(self, item: Union[str, int]):
        return self.get(item)
