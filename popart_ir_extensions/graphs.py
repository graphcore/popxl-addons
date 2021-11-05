# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import wraps
from typing import Any, Optional, Tuple, Union, List, Mapping, Callable, Iterable

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from more_itertools import peekable
from popart.ir import dtypes

from popart_ir_extensions.tuple_map import TupleMap, sanitise

__all__ = ['GenericGraph', 'ConcreteGraph', 'CallableGraph', 'graph']


class InputFactory:
    def __init__(self,
                 data_iter: Union[Callable[[None], np.ndarray], Iterable[np.ndarray]],
                 name: Optional[str] = None,
                 constant: bool = False):
        """
        Generates input tensor for each instance of a callable graph.

        Args:
            data_iter:
                Either a function or iterable that generates data for each instance of the input tensor. Each element of
                data should be a `np.ndarray` with the same shape and data type (this is not checked at runtime).
                If you want your data to be the same for all tensor instances wrap it in a lambda function
                e.g. `lambda: data`.
            name (str):
                The name of the input tensor - by default 't'
            constant (bool):
                If false a variable tensor will be generated, otherwise a constant.
        """

        if callable(data_iter):

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

        self.data_iter: peekable[np.ndarray] = peekable(data_iter_)
        self.name = name
        self.constant = constant

        data_peek = self.data_iter.peek()
        if not isinstance(data_peek, np.ndarray):
            raise ValueError(f"`data_iter` must generate `np.ndarray`s. It provided: {data_peek}.")

    def create_input(self, prefix: Optional[str] = None) -> pir.Tensor:
        """
        Create a subgraph input for the current graph.
        Peaks at the data iterator's next element to determine data type and shape of the input.
        """
        data: np.ndarray = self.data_iter.peek()
        pir_dtype = dtypes.dtype.as_dtype(data)
        name = self.name if prefix is None else f"{prefix}/{self.name}"
        return pir.subgraph_input(shape=data.shape, dtype=pir_dtype, name=name)

    def create_tensor(self, prefix: Optional[str] = None):
        """
        Create a new tensor for the current graph.

        Args:
            prefix: String prefixed to name

        Returns:
        """
        name = self.name if prefix is None else f"{prefix}/{self.name}"
        data = next(self.data_iter)

        if not self.constant:
            return pir.variable(data, name=name)
        else:
            return pir.constant(data, name=name)


class CallableMap(TupleMap[pir.Tensor, pir.Tensor]):
    """
    Use to map an input tensor, from another graph, to a tensor on a subgraph.
    name: (subgraph_tensor, tensor)
    """
    def call_input(self):
        """Returns a mapping from subgraph_tensors to tensors."""
        return self.tuple_map()

    @property
    def subgraph_tensors(self):
        return self.a_map()

    @property
    def tensors(self):
        return self.b_map()

    def name_from_tensor(self, tensor: Any):
        for name, t in self.tensors.items():
            if t == tensor:
                return name
        return None


class InputDefs(TupleMap[InputFactory, pir.Tensor]):
    """
    TupleMap of InputFactorys and Tensors inputs for a graph.
    TupleMap: name -> (InputFactor, Tensor)
    """

    _CallableMapType = CallableMap

    @property
    def tensors(self):
        return self.b_map()

    def name_from_tensor(self, tensor: Any):
        for name, t in self.tensors.items():
            if t == tensor:
                return name
        return None


class CallableGraph(CallableMap):
    def __init__(self, graph: pir.Graph, input_defs: Optional[InputDefs] = None):
        super().__init__()
        self._graph = graph
        # This attribute will be picked up by GenericGraph to inherit InputDefs
        self._input_defs = input_defs if input_defs is not None else InputDefs()

    def call(self,
             *args: pir.Tensor,
             subgraph_in_to_parent_in: Optional[Mapping[pir.Tensor, pir.Tensor]] = None,
             **kwargs: pir.Tensor):
        subgraph_in_to_parent_in = subgraph_in_to_parent_in if subgraph_in_to_parent_in is not None else {}
        subgraph_in_to_parent_in = {**self.call_input(), **subgraph_in_to_parent_in}
        return ops.call(self._graph, *args, subgraph_in_to_parent_in=subgraph_in_to_parent_in, **kwargs)

    def call_with_info(self,
                       *args: pir.Tensor,
                       subgraph_in_to_parent_in: Optional[Mapping[pir.Tensor, pir.Tensor]] = None,
                       **kwargs: pir.Tensor):
        subgraph_in_to_parent_in = subgraph_in_to_parent_in if subgraph_in_to_parent_in is not None else {}
        subgraph_in_to_parent_in = {**self.call_input(), **subgraph_in_to_parent_in}
        return ops.call_with_info(self._graph, *args, subgraph_in_to_parent_in=subgraph_in_to_parent_in, **kwargs)


class ConcreteGraph(pir.Graph):
    @wraps(pir.Graph.__init__)
    def __init__(self):
        """Note: `ConcreteGraph` are generally generated from `GenericGraphs`. This method is for advanced users."""
        super().__init__()
        self._input_defs: InputDefs = InputDefs()
        self._input_tensors: CallableMap = CallableMap()

    @classmethod
    def _from_pb(cls, graph, input_defs: Optional[InputDefs] = None, input_tensors: Optional[CallableMap] = None):
        # This method sets the ir._graph_cache to return a new ConcreteGraph instead.
        self = super()._create_from_pb.__func__(cls, graph)

        self._input_defs = input_defs if input_defs is not None else InputDefs()
        self._input_tensors = input_tensors if input_tensors is not None else CallableMap()
        return self

    @property
    def input_defs(self) -> InputDefs:
        return self._input_defs

    @property
    def input_tensors(self) -> CallableMap:
        return self._input_tensors

    def to_callable(self, create_inputs: bool = False, name_prefix: Optional[str] = None) -> CallableGraph:
        """
        Create a CallableGraph from the ConcreteGraph. `create_inputs` determines if the input tensors should be
        created, otherwise a subgraph_input will be constructed for each input and input_defs for the new inputs
        will be attached to the returned CallableGraph.

        Args:
            create_inputs: Should the inputs be created
            name_prefix: Name prefix for tensor inputs (if created)

        Returns:
            CallableGraph
        """
        return self.to_callable_with_mapping({}, create_inputs, name_prefix)

    def to_callable_with_mapping(self,
                                 mapping: Mapping[pir.Tensor, pir.Tensor],
                                 create_inputs: bool = False,
                                 name_prefix: Optional[str] = None) -> CallableGraph:
        """
        The same as `ConcreteGraph.to_callable`. Argument `mapping` allows for overriding the created tensor for an InputDef in input_defs.

        Args:
            mapping (Mapping[pir.Tensor, pir.Tensor]): Mapping from subgraph tensors to tensors to be default inputs to the CallableGraph.
            name_prefix (Optional[str], optional): Defaults to None.

        Returns:
            CallableGraph
        """
        call_map, in_defs = self._create_call_map_and_defs(self.input_defs, create_inputs, mapping, name_prefix)
        graph = CallableGraph(self, in_defs)
        graph.insert_all(call_map)
        graph.insert_all(self._input_tensors)
        for tensor in set(mapping.keys()) - set(call_map.call_input().keys()):
            graph[sanitise(tensor.name)] = tensor, mapping[tensor]
        return graph

    def _create_call_map_and_defs(self,
                                  input_defs: 'InputDefs',
                                  create_inputs: bool = False,
                                  mapping: Optional[Mapping[pir.Tensor, pir.Tensor]] = None,
                                  prefix: Optional[str] = None) -> Tuple['CallableMap', 'InputDefs']:
        '''Convert InputDefs in a TupleMap to input tensors in the current graph.'''
        mapping = mapping or {}
        call_map = input_defs._CallableMapType()
        parent_defs = InputDefs()
        for name, item in input_defs.items():
            if isinstance(item, tuple):
                var_def, graph_tensor = item
                parent_input = mapping.get(graph_tensor, None)
                if parent_input is None:
                    # Create an input tensor if there isn't one in the mapping
                    if create_inputs:
                        parent_input = var_def.create_tensor(prefix)
                    else:
                        parent_input = var_def.create_input()
                call_map[name] = graph_tensor, parent_input
                if not create_inputs:
                    parent_defs[name] = var_def, parent_input
            else:
                child_prefix = name if prefix is None else f"{prefix}.{name}"
                _map, _defs = self._create_call_map_and_defs(item, create_inputs, mapping, child_prefix)  # type: ignore
                call_map[name] = _map
                parent_defs[name] = _defs
        return call_map, parent_defs

    def __getattr__(self, name: str):
        try:
            return getattr(self._input_defs, name)
        except AttributeError as e:
            pass
        try:
            return getattr(self._input_tensors, name)
        except AttributeError as e:
            pass
        return super().__getattribute__(name)

    def add_input_tensor(
        self,
        data_iter: Union[Callable[[None], np.ndarray], Iterable[np.ndarray]],
        name: Optional[str] = None,
        constant: bool = False,
    ) -> pir.Tensor:
        """
        Add an input tensor (variable or constant) to the `ConcreteGraph`. When the graph is converted to
        `CallableGraph` the user can choose if the input tensor is created or not.

        Args:
            name (str):
                The name of the tensor
            data_iter:
                Either a function or iterable that generates data for each instance of the input tensor. Each element of
                data should be a `np.ndarray` with the same shape and data type (this is not checked at runtime).
                If you want your data to be the same for all tensor instances wrap it in a lambda function
                e.g. `lambda: data`
            constant (bool = False):
                If false a variable tensor will be generated, otherwise a constant.

        Returns:
            pir.Tensor
        """

        with self:
            var_def = InputFactory(data_iter, name, constant)
            tensor = var_def.create_input()
        self._input_defs[name] = (var_def, tensor)
        return tensor


class GenericGraph(pir.Module):
    """Graph function that captures any variable_def created during construction."""
    def __init__(self) -> None:
        super().__init__()
        super().__setattr__("_input_defs", InputDefs())
        super().__setattr__("_input_tensors", CallableMap())
        self._input_defs: InputDefs  # Tensors that can be inputs or exist on this graph (TBD)
        self._input_tensors: CallableMap  # Tensors that are going to be inputs

    def to_concrete(self, *args: Any, ir: Optional[pir.Ir] = None, **kwargs: Any) -> ConcreteGraph:
        """
        Creates a graph (known as subgraph in popart internals) that is outlined.
        """
        ir = ir if ir is not None else pir.gcg().ir()
        graph = ir.create_graph(self, *args, **kwargs)
        return ConcreteGraph._from_pb(graph._pb_graph, self._input_defs, self._input_tensors)

    def __setattr__(self, name: str, value: Any) -> None:
        """Inplace another graph and merge it with the current if GenericGraph or CallableGraph.
        Otherwise set a attribute like normal. """
        _input_defs = self.__getattribute__("_input_defs")
        _input_tensors = self.__getattribute__("_input_tensors")
        # Capture child GenericGraphs
        if (isinstance(value, GenericGraph) or isinstance(value, CallableGraph)):
            _input_defs[name] = value._input_defs
        if isinstance(value, GenericGraph):
            _input_tensors[name] = value._input_tensors

        return super().__setattr__(name, value)

    def __getattr__(self, name: str):
        """Get a Tensor def which is a member of the graph or access any other attribute."""
        try:
            return getattr(self._input_defs, name)
        except AttributeError as e:
            pass
        try:
            return getattr(self._input_tensors, name)
        except AttributeError as e:
            pass
        return super().__getattribute__(name)

    def add_input_tensor(self,
                         name: str,
                         data_iter: Union[Callable[[None], np.ndarray], Iterable[np.ndarray]],
                         constant: bool = False) -> pir.Tensor:
        """
        Add an input tensor (variable or constant) to the `GenericGraph`. When the graph is converted to
        `CallableGraph` the user can choose if the input tensor is created or not.

        Args:
            name (str):
                The name of the tensor
            data_iter:
                Either a function or iterable that generates data for each instance of the input tensor. Each element of
                data should be a `np.ndarray` with the same shape and data type (this is not checked at runtime).
                If you want your data to be the same for all tensor instances wrap it in a lambda function
                e.g. `lambda: data`
            constant (bool = False):
                If false a variable tensor will be generated, otherwise a constant.

        Returns:
            pir.Tensor

        Add a variable to the graph.
        When the graph is converted to a concrete graph the user can choose if the variable is an input or a
        member of this graph.

        Args:
            name: name of variable
            data: numpy data

        Returns:
            pir.Tensor
        """
        var_def = InputFactory(data_iter=data_iter, name=name, constant=constant)
        tensor = var_def.create_input()
        self._input_defs[name] = (var_def, tensor)
        return tensor

    def add_static_input_tensor(self, name: str, tensor: pir.Tensor) -> pir.Tensor:
        """
        Add an input tensor (variable or constant) to the `GenericGraph` where the variable has already been
        initialised.

        Args:
            tensor: pir.Tensor to add as a input from another graph

        Returns:
            pir.Tensor which exists on current graph
        """
        tensor_cg = pir.subgraph_input(tensor.shape, tensor.dtype, name)
        self._input_tensors[name] = (tensor_cg, tensor)
        return tensor_cg


def graph(fn):
    """Decorator. Converts a python callable into a GenericGraph"""
    class FreeFnGraph(GenericGraph):
        def build(self, *args: pir.Tensor, **kwargs: pir.Tensor) -> Union[pir.Tensor, Tuple[pir.Tensor, ...]]:
            return fn(*args, **kwargs)

    return FreeFnGraph()


class _GraphList:
    """Abstract class that implements GraphList methods"""
    def get(self, index):
        try:
            return getattr(self, f'i{index}')
        except AttributeError:
            raise IndexError(f"Index '{index}' is out of range for length {len(self)}")

    def __len__(self):
        return len(self._map)


class CallableMapList(CallableMap, _GraphList):
    pass


class InputDefsList(InputDefs, _GraphList):
    _CallableMapType = CallableMapList


class GenericGraphList(GenericGraph, _GraphList):
    def __init__(self, modules: List['GenericGraph']):
        super().__init__()
        super().__setattr__("_input_defs", InputDefsList())
        self._input_defs: InputDefsList

        for index, m in enumerate(modules):
            self.__setattr__(f'i{index}', m)
