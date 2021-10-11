# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Any, Optional, Tuple, Union
from functools import wraps
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops

from popart_extensions.tuple_map import TupleMap

__all__ = ['GenericGraph', 'ConcreteGraph', 'CallableGraph', 'graph']

"""
Evolution of a graph:
GenericGraph -> ConcreteGraph -> CallableGraph

GenericGraph:
    - A graph of ops and VariableDefs
    - VariableDef do not specify dtype, shape or data
    - A GenericGraph is a factor for ConcreteGraphs with variable tensor dtypes and shape

ConcreteGraph:
    - A graph of ops and VariableDef
    - VariableDef do specify dtype and shape but not data
    - A ConcreteGraph is a factory for CallableGraph with variable data
    - ConcreteGraphs also define outlining in the graph

CallableGraph:
    - A graph of ops and Tensors?
    - VariableDef do specify dtype, shape and data

---

The ir contains multiple graphs including a main graph and subgraphs.
Each graph can call another graph (using the `op.call`).
Subgraphs can't contain variables.

## Example 1:
* Both ops are members of the main graph
* Both ops have different code paths and are inlined
* No subgraphs are created

Vanilla popart.ir:
```
ir = pir.Ir()
main = ir.main_graph()
with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")
        scale = pir.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")

        y = x * scale # Op 1
        z = x * scale # Op 2
```

## Example 2:
* A subgraph is created that performs scale
* The subgraph is called twice to before the operation twice
* The code is outlined as you are reusing the same subgraph

Vanilla popart.ir:
```
def scale_fn(x: pir.Tensor, scale: pir.Tensor):
        return x * scale

ir = pir.Ir()
main = ir.main_graph()
with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale = pir.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
        scale_graph = ir.create_graph(scale_fn, x, scale)

        y = ops.call(scale_graph, x, scale) # Subgraph A. Add subgraph to maingraph. Call site 1
        z = ops.call(scale_graph, y, scale) # Subgraph A. Call site 2
```

Extensions popart.ir:
```
class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = pir_ext.variable_def(np.ones(x.shape, x.dtype.as_numpy()), "scale")
        return x * self.scale

ir = pir.Ir()
main = ir.main_graph()
with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale_graph = Scale().to_concrete(x)
        scale = scale_graph.to_callable(create_variables=True)

        y = scale.call(x, scale) # Subgraph A
        z = scale.call(y, scale) # Subgraph A
```

## Example 3:
* Reuse of the same subgraph but with a different scale variable

Vanilla popart.ir:
```
def scale_fn(x: pir.Tensor, scale: pir.Tensor):
        return x * scale

ir = pir.Ir()
main = ir.main_graph()
with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale1 = pir.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
        scale2 = pir.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
        scale_graph = ir.create_graph(scale_fn, x, scale1) #this is only taking shape and type

        y = ops.call(scale_graph, x, scale1) # Subgraph A with scale 1
        z = ops.call(scale_graph, y, scale2) # Subgraph A with scale 2
```

Extensions popart.ir:
```
class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = pir_ext.variable_def(np.ones(x.shape, x.dtype.as_numpy()), "scale")
        return x * self.scale

ir = pir.Ir()
main = ir.main_graph()
with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale_graph = Scale().to_concrete(x)
        scale1 = scale_graph.to_callable(create_variables=True)
        scale2 = scale_graph.to_callable(create_variables=True)

        y = scale1.call(x) # Subgraph A with scale 1
        z = scale2.call(y) # Subgraph A with scale 2
```
"""


class VariableDef:
    """Description of a Variable."""
    def __init__(self,
                 # TODO: This should be a callable so every variable isn't identical
                 data: np.ndarray,
                 name: Optional[str] = None):
        self.data = data
        # TODO: move auto type detection to `pir.variable`
        self.dtype = pir.dtype.as_dtype(data)
        self.name = "var" if name is None else name

    def create_input(self) -> pir.Tensor:
        return pir.subgraph_input(shape=self.data.shape, dtype=self.dtype, name=self.name)

    def create_variable(self, prefix: Optional[str] = None):
        name = self.name if prefix is None else f"{prefix}.{self.name}"
        return pir.variable(self.data, dtype=self.dtype, name=name)


class VariableDefs(TupleMap[VariableDef, pir.Tensor]):
    """Container for VariableDefs.
        Can return all variable defs by reference to another tensor."""

    @classmethod
    def _create_variable_map(cls, variable_defs: 'VariableDefs', prefix: Optional[str] = None) -> 'CallableMap':
        '''Convert VariableDefs in a VariableDefs object to Variables.'''
        variables = CallableMap()

        for name, item in variable_defs.items():
            if isinstance(item, tuple):
                var_def, graph_tensor = item
                variables[name] = graph_tensor, var_def.create_variable(prefix)

            else:
                child_prefix = name if prefix is None else f"{prefix}.{name}"
                # Assume to be child VaribleDefs
                variables[name] = cls._create_variable_map(item, child_prefix)  # type: ignore

        return variables

    @classmethod
    def _create_subgraph_map_and_defs(cls, variable_defs: 'VariableDefs') -> Tuple['CallableMap', 'VariableDefs']:
        '''Convert VariableDefs in a TupleMap to input tensors in the current graph.'''
        call_map = CallableMap()
        parent_defs = VariableDefs()
        for name, item in variable_defs.items():
            if isinstance(item, tuple):
                var_def, graph_tensor = item
                parent_input = var_def.create_input()
                call_map[name] = graph_tensor, parent_input
                parent_defs[name] = var_def, parent_input
            else:
                # Assume to be child VaribleDefs
                _map, _defs = cls._create_subgraph_map_and_defs(item)  # type: ignore
                call_map[name] = _map
                parent_defs[name] = _defs
        return call_map, parent_defs

    @property
    def tensors(self):
        return self.b_map()

    def name_from_tensor(self, tensor: Any):
        for name, t in self.tensors.items():
            if t == tensor:
                return name
        return None


class CallableMap(TupleMap[pir.Tensor, pir.Tensor]):
    def call_input(self):
        """Returns a mapping from subgraph_tensors to tensors"""
        return self.tuple_map()

    @property
    def subgraph_tensors(self):
        return self.a_map()

    @property
    def tensors(self):
        return self.b_map()


class CallableGraph(CallableMap):
    def __init__(self,
                 graph: pir.Graph,
                 variable_defs: Optional[VariableDefs] = None):
        super().__init__()
        self._graph = graph
        # This attribute will be picked up by GenericGraph to inherit VariableDefs
        self._variable_defs = variable_defs if variable_defs is not None else VariableDefs()

    def call(self, *args: pir.Tensor, **kwargs: pir.Tensor):
        return ops.call(self._graph, *args, subgraph_in_to_parent_in=self.call_input(), **kwargs)

    def call_with_info(self, *args: pir.Tensor, **kwargs: pir.Tensor):
        return ops.call_with_info(self._graph, *args, subgraph_in_to_parent_in=self.call_input(), **kwargs)


class ConcreteGraph(pir.Graph):
    @wraps(pir.Graph.__init__)
    def __init__(self):
        super().__init__()
        self._variable_defs: VariableDefs

    @classmethod
    def _from_pb(cls, graph, variable_defs: Optional[VariableDefs] = None):
        # Get the unbound version of _from_pb
        self = super()._from_pb.__func__(cls, graph)
        self._variable_defs = variable_defs if variable_defs is not None else VariableDefs()
        return self

    @property
    def variable_defs(self) -> VariableDefs:
        return self._variable_defs

    def to_callable(self, create_variables: bool = False, debug_prefix: Optional[str] = None) -> CallableGraph:
        if create_variables:
            graph = CallableGraph(self, self.variable_defs)
            graph.insert_all(VariableDefs._create_variable_map(self.variable_defs, debug_prefix))
        else:
            call_map, var_defs = VariableDefs._create_subgraph_map_and_defs(self.variable_defs)
            graph = CallableGraph(self, var_defs)
            graph.insert_all(call_map)
        return graph

    def __getattr__(self, name: str):
        try:
            return getattr(self._variable_defs, name)
        except AttributeError as e:
            pass
        return super().__getattribute__(name)

    def add_var_input(self, name: str, data: np.ndarray) -> pir.Tensor:
        with self:
            var_def = VariableDef(data, name)
            tensor = var_def.create_input()
        self._variable_defs[name] = (var_def, tensor)
        return tensor


class GenericGraph(pir.Module):
    """Graph function that captures any variable_def created during construction."""

    def __init__(self) -> None:
        super().__init__()
        super().__setattr__("_variable_defs", VariableDefs())
        self._variable_defs: VariableDefs

    def to_concrete(self, *args: Any, ir: Optional[pir.Ir] = None, **kwargs: Any) -> ConcreteGraph:
        ir = ir if ir is not None else pir.gcg().ir()
        graph = ir.create_graph(self, *args, **kwargs)
        return ConcreteGraph._from_pb(graph._pb_graph, self._variable_defs)

    def __setattr__(self, name: str, value: Any) -> None:
        _variable_defs = self.__getattribute__("_variable_defs")
        # Capture child GenericGraphs
        if isinstance(value, GenericGraph) or isinstance(value, CallableGraph):
            _variable_defs[name] = value._variable_defs

        return super().__setattr__(name, value)

    def __getattr__(self, name: str):
        try:
            return getattr(self._variable_defs, name)
        except AttributeError as e:
            pass
        return super().__getattribute__(name)

    def add_var_input(self, name: str, data: np.ndarray) -> pir.Tensor:
        var_def = VariableDef(data, name)
        tensor = var_def.create_input()
        self._variable_defs[name] = (var_def, tensor)
        return tensor


def graph(fn):
    """Decorator. Converts a python callable into a GenericGraph"""

    class FreeFnGraph(GenericGraph):
        def build(self, *args: pir.Tensor, **kwargs: pir.Tensor) -> Union[pir.Tensor, Tuple[pir.Tensor, ...]]:
            return fn(*args, **kwargs)

    return FreeFnGraph()
