# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import wraps
import warnings
from typing import Any, Callable, Iterable, Optional, Tuple, Union, List, Mapping, Set, Dict
from typing_extensions import Literal
import inspect

import popxl
from popxl import dtypes, ReplicaGrouping, Tensor, Ir
from popxl.tensor import HostTensor
from popxl.transforms.autodiff import GradGraphInfo
import popart._internal.ir as _ir

from popxl_addons.graph_cache import GraphCache
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.variable_factory import (
    NamedVariableFactories,
    add_variable_input,
    add_replica_sharded_variable_input,
    VariableFactory,
)
from popxl_addons.graph import GraphWithNamedArgs, BoundGraph
from popxl_addons.utils import OrderedDict, overrides, duplicate_items
from contextlib import contextmanager

GraphLike = Union[popxl.Graph, GraphWithNamedArgs, BoundGraph]
GradGraphInfoLike = Union[GradGraphInfo, GraphWithNamedArgs]

_BUILD_CONTEXT: List["Module"] = []
_BUILD_GRAD_CONTEXT: List["Module"] = []


@contextmanager
def _build_context(module: "Module"):
    global _BUILD_CONTEXT
    _BUILD_CONTEXT += [module]
    yield
    _BUILD_CONTEXT.pop()


@contextmanager
def _build_grad_context(module: "Module"):
    global _BUILD_GRAD_CONTEXT
    _BUILD_GRAD_CONTEXT += [module]
    yield
    _BUILD_GRAD_CONTEXT.pop()


def _get_current_build_context() -> Optional["Module"]:
    if len(_BUILD_CONTEXT):
        return _BUILD_CONTEXT[-1]
    return None


def _get_current_build_grad_context() -> Optional["Module"]:
    if len(_BUILD_GRAD_CONTEXT):
        return _BUILD_GRAD_CONTEXT[-1]
    return None


class NameScopeMeta(type):
    """Meta class to wrap `build` methods with a popxl.name_scope"""

    def __new__(cls, name, bases, dct):
        build_fn = dct.get("build", None)
        if build_fn is not None and callable(build_fn):

            @wraps(build_fn)
            def wrapper(self, *args, **kwargs):
                scope = getattr(self, "_name_scope", None)
                if scope is not None:
                    with popxl.name_scope(scope):
                        return build_fn(self, *args, **kwargs)
                return build_fn(self, *args, **kwargs)

            dct["build"] = wrapper
        return super().__new__(cls, name, bases, dct)


class _Aux:
    def __init__(self):
        """A dict like object that holds tensors that are needed for the gradient graph"""
        self._dict = OrderedDict()
        # Maps from grad graph input to fwd graph output
        self._grad_to_fwd = OrderedDict()
        # Maps from key to grad graph input
        self._key_to_tensor = OrderedDict()

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key) -> popxl.Tensor:
        module = _get_current_build_grad_context()
        if module is None or module.aux != self:
            raise Exception(
                "You can only get items within the `build_graph` method which must be called via `create_grad_graph` or `autodiff`."
            )
        aux = self._dict[key]
        if not isinstance(aux, popxl.Tensor):
            return aux
        if key in self._key_to_tensor:
            return self._key_to_tensor[key]
        spec = aux.spec
        graph_input = popxl.graph_input(spec.shape, spec.dtype, key, False, spec.meta_shape)
        self._key_to_tensor[key] = graph_input
        self._grad_to_fwd[graph_input] = aux
        return graph_input

    def __contains__(self, key) -> bool:
        return key in self._dict

    def __repr__(self) -> str:
        return self._dict.__repr__()

    def copy(self) -> "_Aux":
        """Shallow copy of self"""
        new = _Aux()
        new._dict = self._dict.copy()
        new._grad_to_fwd = self._grad_to_fwd.copy()
        new._key_to_tensor = self._key_to_tensor.copy()
        return new


class _VarGrads:
    def __init__(self, grads_required_vars_with_names: Optional[Dict[str, popxl.Tensor]] = None) -> None:
        """A dict like object that specifies which tensors are the gradients of the forward graph's variables"""
        self._dict = OrderedDict()
        self._vars = grads_required_vars_with_names

    def __getitem__(self, key):
        module = _get_current_build_grad_context()
        if module is None or module.var_grad != self:
            raise Exception(
                "You can only use` var_grad` within the `build_graph` method which must be called via `create_grad_graph` or `autodiff`."
            )
        return self._dict[key]

    def __setitem__(self, key, tensor):
        module = _get_current_build_grad_context()
        if module is None or module.var_grad != self or self._vars is None:
            raise Exception(
                "You can only use` var_grad` within the `build_graph` method which must be called via `create_grad_graph` or `autodiff`."
            )
        if not isinstance(tensor, popxl.Tensor):
            raise TypeError(f"Not a tensor: {tensor}")
        if key not in self._vars:
            raise KeyError(f"Not a variable: {key}. Possible inputs: {list(self._vars.keys())}")
        var = self._vars[key]
        if var.shape != tensor.shape or var.meta_shape != tensor.meta_shape:
            raise ValueError(
                "Shape or meta-shape of Tensor does not match variable. "
                f"Tensor shape: {tensor.shape} meta-shape {tensor.meta_shape}. "
                f"Variable shape {var.shape} meta-shape {var.meta_shape}."
            )
        self._dict[key] = tensor

    def __contains__(self, key) -> bool:
        return key in self._dict

    def __repr__(self) -> str:
        return self._dict.__repr__()


class _CalledGraphsGradInfoRefs:
    def __init__(self, map: Optional[Mapping[GraphLike, GradGraphInfoLike]] = None):
        """Container to hold a list of dictionaries that map called graphs to grad graph info objects.
        The container saves the dicts by reference so that sub-modules can continue to modify the dicts inplace"""
        self.map: Dict[GraphLike, GradGraphInfoLike] = (
            _normalise_called_graphs_grad_info_dict(map) if map is not None else {}
        )
        self.submaps: List[_CalledGraphsGradInfoRefs] = []

    def consolidate(self) -> Dict[popxl.Graph, GradGraphInfo]:
        """Consolidate and normalise to a single mapping"""
        called_graphs_grad_info = {}
        for submap in self.submaps:
            called_graphs_grad_info.update(submap.consolidate())
        # Top level map gets priority
        called_graphs_grad_info.update(self.map)
        return called_graphs_grad_info

    def add(self, fwd_graph: GraphLike, grad_graph: GradGraphInfoLike):
        """Add an association between a forward and grad graph"""
        fwd_graph, grad_graph = _normalise_called_graphs_grad_info(fwd_graph, grad_graph)
        self.map[fwd_graph] = grad_graph

    def add_submap(
        self, called_graphs_grad_info: Union["_CalledGraphsGradInfoRefs", Mapping[GraphLike, GradGraphInfoLike]]
    ):
        """Add a mapping"""
        if not isinstance(called_graphs_grad_info, _CalledGraphsGradInfoRefs):
            called_graphs_grad_info = _CalledGraphsGradInfoRefs(called_graphs_grad_info)
        self.submaps.append(called_graphs_grad_info)

    def clear(self):
        """Clear self's and submodule's refs"""
        for map in self.submaps:
            map.clear()
        self.maps = {}
        self.submaps = []


class Module(popxl.Module, metaclass=NameScopeMeta):
    def __init__(
        self,
        cache: bool = False,
        grad_inlining_error: Literal["error", "warning", "none"] = "error",
    ):
        """
        Module class to allow construction of compute graphs that require state.

        Args:
            cache (bool, optional): Re-use graphs where possible when calling `create_graph`. Defaults to False.
            grad_inlining_error (str): Option on weather to raise an error, print a warning or do nothing if the module implements `build_grad` and is attempted to be inlined into another module
        """
        # Module should not hold state. All parameters get cleared after a call or create_graph (with the exception of graph_cache)
        self._variable_factories = NamedVariableFactories()
        self._named_inputs = NamedTensors()
        self._graph_cache = GraphCache() if cache else None
        self._args_cache = {}
        self.aux = _Aux()
        self.var_grad = _VarGrads()
        self.grad_inlining_error = grad_inlining_error
        self.called_graphs_grad_info = _CalledGraphsGradInfoRefs()

        if cache and self.implements_grad_graph:
            raise ValueError("You cannot use graph caching if you have implemented `build_grad`.")

        if self.grad_inlining_error not in ("error", "warning", "none"):
            raise ValueError(
                f"grad_inlining_error should be one of 'error', 'warning', 'none'. Not: {self.grad_inlining_error}"
            )

    def __call__(self, *args, **kwargs) -> Union[None, "Tensor", Iterable["Tensor"]]:
        if self.implements_grad_graph and _get_current_build_context() != self:
            msg = (
                "You are calling this module directly but `build_grad` has been implemented. "
                "If you call this module directly `build_grad` will not be called when using autodiff. "
                "You need to create a graph of this module first and call that graph e.g. use `self.call_module`. "
                "To disable this guardrail set `grad_inlining_error` to `none`"
            )
            if self.grad_inlining_error == "error":
                raise Exception(msg)
            elif self.grad_inlining_error == "warning":
                warnings.warn(msg)
        return super().__call__(*args, **kwargs)

    def build_grad(self, *args, **kwargs) -> Union[None, popxl.Tensor, Iterable[popxl.Tensor]]:
        """Define a gradient graph. The order of the inputs should match the outputs of the forward graph.
        Similarly the order of the outputs should match the outputs of the forward graph. See `Module.create_grad_graph` for more details.
        """
        raise NotImplementedError(
            "Your popxl.Module must implement the `build_grad` method to create a gradient graph."
        )

    @property
    def implements_grad_graph(self) -> bool:
        """Does this module implement the grad_graph method"""
        return overrides(self.__class__, Module, "build_grad")

    def _reset(self):
        """Reset all state of the module (excluding caching). This is done after `create_graph` and `create_grad_graph`"""
        # Use `clear` on _variable_factories, _named_inputs and called_graphs_grad_info so the sub-module containers are also cleared
        self._variable_factories._clear()
        self._named_inputs._clear()
        self.called_graphs_grad_info.clear()
        self.aux = _Aux()
        self.var_grad = _VarGrads()

    def create_graph(self, *args, **kwargs) -> Tuple[NamedVariableFactories, GraphWithNamedArgs]:
        """Construct a compute graph and variable factories for the module."""
        if self._graph_cache is not None:
            # Note: graph caching is not supported with build_grad
            graph = self._graph_cache.create_graph(self, *args, **kwargs)
            if graph not in self._args_cache:
                self._args_cache[graph] = (
                    self._variable_factories.copy(),
                    self._named_inputs.copy(),
                )
            variable_factories, named_inputs = self._args_cache[graph]
        else:
            self.aux = _Aux()
            with _build_context(self):
                graph = popxl.gir().create_graph(self, *args, **kwargs)
            if self.implements_grad_graph:
                graph._pb_graph.setCanBeRecursivelyAutodiffed(False)
            variable_factories, named_inputs = self._variable_factories, self._named_inputs
        # Copy to state to graph - module should not hold state
        graph_wna = GraphWithNamedArgs(
            graph,
            named_inputs.copy(),
            from_module=self,
            aux=self.aux.copy(),
            called_graphs_grad_info=self.called_graphs_grad_info.consolidate(),  # type: ignore
        )
        variable_facts = variable_factories.copy()
        self._reset()  # Module should not hold state
        return variable_facts, graph_wna

    def _create_grad_graph_pre_validation(self, fwd_graph, grads_provided, grads_required):
        """Validation steps needed in create_grad_graph method before creating the graph"""
        # Validate `self.build_grad` function signature
        signature = inspect.signature(self.build_grad, follow_wrapped=True)
        for name, param in signature.parameters.items():
            if param.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                raise TypeError(
                    "The `build_grad` method signature must only have a fixed number of positional parameters as inputs. "
                    f"Not a positional parameter: {name}"
                )

        # Validate grads_provided and grads_required
        if grads_provided is not None:
            grads_provided = list(grads_provided)
            for t in grads_provided:
                if not isinstance(t, popxl.Tensor) or t not in fwd_graph.graph.outputs:
                    raise ValueError(
                        f"A Tensor specified in `grads_provided` is not a output of `fwd_graph` or not a Tensor: {t}"
                    )
            duplicates = duplicate_items(grads_provided)
            if len(duplicates):
                raise ValueError(
                    "A Tensor has been specified more than once in `grads_provided`. Duplicates: {duplicates}"
                )
        else:
            grads_provided = fwd_graph.graph.outputs

        if grads_required is not None:
            grads_required = list(grads_required)
            for t in grads_required:
                if not isinstance(t, popxl.Tensor) or t not in fwd_graph.graph.inputs:
                    raise ValueError(
                        f"A Tensor specified in `grads_required` is not a input of `fwd_graph` or not a Tensor: {t}"
                    )
            duplicates = duplicate_items(grads_required)
            if len(duplicates):
                raise ValueError(
                    "A Tensor has been specified more than once in `grads_required`. Duplicates: {duplicates}"
                )
        else:
            grads_required = fwd_graph.graph.inputs

        # Separate variable and non-variable input tensors
        name_to_variable = fwd_graph.args.to_dict()
        variable_to_name = dict(zip(name_to_variable.values(), name_to_variable.keys()))
        grads_required_vars_with_names = OrderedDict()
        for t in grads_required:
            if t in variable_to_name:
                grads_required_vars_with_names[variable_to_name[t]] = t
        grads_required_not_vars = [t for t in grads_required if t not in variable_to_name]

        return grads_provided, grads_required, grads_required_not_vars, grads_required_vars_with_names, name_to_variable

    def _create_grad_graph_post_validation(self, grad_graph, grads_required_not_vars, grads_required_vars_with_names):
        """Validation steps needed in create_grad_graph method after creating the graph"""
        # Check number of outputs
        missing_grad_required = set(grads_required_vars_with_names.keys()) - set(self.var_grad._dict.keys())
        if len(missing_grad_required):
            raise Exception(
                "Not all variable gradients have been specified in self.var_grad. "
                "If you dont want to output all gradients specify `grads_required` when calling autodiff. "
                f"Variables missing gradients: {missing_grad_required}"
            )

        if len(grad_graph.outputs) > len(grads_required_not_vars):
            raise Exception(
                f"Too many outputs in the grad graph. `grads_required` has {len(grads_required_not_vars)} non-variable outputs. "
                "If you dont want to output all gradients specify `grads_required` when calling autodiff."
            )
        if len(grad_graph.outputs) < len(grads_required_not_vars):
            missing = grad_graph.outputs[len(grad_graph.outputs) :]
            raise Exception(
                "Not all forward graph inputs have been specified as outputs in the grad graph. "
                "If you dont want to output all gradients specify `grads_required` when calling autodiff. "
                f"Outputs missing: {missing}"
            )

        # Check output shapes
        for i, (grad_output, required) in enumerate(zip(grad_graph.outputs, grads_required_not_vars)):
            if grad_output.shape != required.shape or grad_output.meta_shape != required.meta_shape:
                raise ValueError(
                    f"Shape or meta-shape of output Tensor at position {i} does not match the `grad_required` at position {i}. "
                    f"Output Tensor shape: {grad_output.shape} meta-shape {grad_output.meta_shape}. "
                    f"Grad required shape {required.shape} meta-shape {required.meta_shape}."
                )

    def create_grad_graph(
        self,
        fwd_graph: "GraphWithNamedArgs",
        grads_provided: Optional[Iterable[popxl.Tensor]] = None,
        grads_required: Optional[Iterable[popxl.Tensor]] = None,
    ) -> GraphWithNamedArgs:
        """Create a grad graph using the `build_graph` method implemented on this object. It modifies `fwd_graph` inplace so the required outputs
        that have been stored in the `self.aux` attribute are available to the grad graph as inputs (known as stitching). Specify gradients that correspond with variables in the
        forward graph using the `self.var_grad` attribute.

        Example:
        ```python
        class Linear(Module):
            def __init__(self, features: int):
                super().__init__()
                self.features = features

            def build(self, x: popxl.Tensor):
                self.w = self.add_variable_input("w", partial(np.zeros, (x.shape[-1], self.features)), popxl.float32)
                self.b = self.add_variable_input("b", partial(np.zeros, (self.features,)), popxl.float32)
                self.aux["x"] = x
                self.aux["w"] = self.w
                y = (x @ self.w) + self.b
                return y

            def build_grad(self, dLdy: popxl.Tensor):
                dLdx = dLdy @ self.aux["w"].T
                dLdw = (dLdy.T @ self.aux["x"]).T
                dLdb = ops.sum(dLdy, 0)
                self.var_grad["b"] = dLdb
                self.var_grad["w"] = dLdw
                return dLdx
        ```

        Note that this method modifies `fwd_graph` inplace and therefore cannot be called twice on the same forward graph.

        Args:
            fwd_graph (GraphWithNamedArgs): Forward graph that corresponds with the gradient graph specified in `self.build_graph`
            grads_provided (Optional[Iterable[popxl.Tensor]], optional): Specifies the inputs of the gradient graph `self.build_graph` which should be forward graph output tensors. Defaults to all forward graph outputs.
            grads_required (Optional[Iterable[popxl.Tensor]], optional): Specifies the outputs of the gradient graph `self.build_graph` which should be forward graph output tensors. A subset of variable output gradients is valid and this will restrict which outputs are included. Defaults to all forward graph outputs.

        Returns:
            GraphWithNamedArgs: Gradient graph which also holds the `grad_graph_info` object
        """
        ir = popxl.gir()

        if not self.implements_grad_graph:
            raise Exception(
                "The default implementation of `build_graph` needs to be overridden to use `create_grad_graph`."
            )

        # Validate inputs
        (
            grads_provided,
            grads_required,
            grads_required_not_vars,
            grads_required_vars_with_names,
            name_to_variable,
        ) = self._create_grad_graph_pre_validation(fwd_graph, grads_provided, grads_required)

        # Create grad graph. If an input is missing `ir.create_graph` will raise an error
        self.var_grad = _VarGrads(name_to_variable)
        self.aux = fwd_graph._aux.copy()
        with _build_grad_context(self):
            grad_graph = ir.create_graph(self.build_grad, *grads_provided)

        # Validate grad_graph
        self._create_grad_graph_post_validation(grad_graph, grads_required_not_vars, grads_required_vars_with_names)

        # Add var_grad outputs to grad graph in order of grads_required
        with grad_graph:
            for name, _ in grads_required_vars_with_names.items():
                t = self.var_grad._dict[name]
                popxl.graph_output(t)

        # Add aux tensor outputs to forward graph
        with fwd_graph.graph:
            for name, tensor in self.aux._dict.items():
                if isinstance(tensor, popxl.Tensor):
                    popxl.graph_output(tensor)

        grad_output_to_fwd_input = OrderedDict(zip(grad_graph.outputs, grads_required))
        grad_input_to_fwd_output = OrderedDict(zip(grad_graph.inputs, grads_provided))
        grad_input_to_fwd_output.update(self.aux._grad_to_fwd)
        fwd_tensors = set(self.aux._grad_to_fwd.values())

        grad_info = create_grad_graph_info(
            ir, fwd_graph.graph, grad_graph, grad_input_to_fwd_output, grad_output_to_fwd_input, fwd_tensors
        )
        self._reset()  # Module should not hold state
        return GraphWithNamedArgs.from_grad_graph(grad_info)

    def add_variable_factory(self, variable_f: VariableFactory, name: Optional[str] = None) -> popxl.Tensor:
        """Add a variable factory as an input tensor.

        Args:
            input_f (VariableFactory): variable factory to generate an input tensor
            name (str): Name for input. If provided it will override the name provided to the variable factory
        """
        tensor = variable_f.create_input()
        name = name if name else variable_f.name
        self._variable_factories.insert(name, variable_f)
        self._named_inputs.insert(name, tensor)
        return tensor

    def add_variable_input(
        self,
        name: str,
        data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
        dtype: Optional[dtypes.dtype] = None,
        by_ref: bool = False,
        replica_grouping: Optional[ReplicaGrouping] = None,
        overwrite: bool = False,
    ) -> popxl.Tensor:
        """Add an initialised input tensor.

        Args:
            name (str): named of the input Tensor. Used for both the Named VariableFactory/Arg and the Tensor debug name.
            data_iter (Union[Callable[[None], np.ndarray], Iterable[np.ndarray]]):
                Either a function or iterable that generates data for each instance of the input tensor. Each element of
                data should be a HostTensor type (numpy, pytorch, ect.) with the same shape and data type (this is not
                checked at runtime). If you want your data to be the same for all tensor instances wrap it in a
                lambda function e.g. `lambda: data`.
            dtype (Optional[dtypes.dtype], optional): dtype of input. Defaults to None.
            constant (bool, optional): Construct input as constant when initialised. Defaults to False.
            by_ref (bool, optional): Pass the input by reference. Defaults to False.
            replica_grouping (Optional[ReplicaGrouping]): The replica group of the variable. Determines which replicas
                of the variable will have identical data or not when written to
            overwrite (bool): If True nested names will be merged and values will be overwritten
        """
        tensor, variable_f = add_variable_input(name, data_iter, dtype, by_ref, replica_grouping)
        self._variable_factories.insert(name, variable_f, overwrite=overwrite)
        self._named_inputs.insert(name, tensor, overwrite=overwrite)
        return tensor

    def add_replica_sharded_variable_input(
        self,
        name: str,
        data_iter: Union[Callable[[None], HostTensor], Iterable[HostTensor]],
        dtype: Optional[dtypes.dtype] = None,
        by_ref: bool = False,
        replica_grouping: Optional[ReplicaGrouping] = None,
        shard_over: Optional[int] = None,
    ) -> popxl.Tensor:
        """Add a replica sharded initialised input tensor. The graph input will be sharded, however the initialised Tensor will
            be constructed whole.
            `remote_replica_sharded_variable` or `ops.collectives.replicated_reduce_scatter(..., configure_output_for_replicated_tensor_sharding=True)`
            should be used to shard the Tensor before passing to the graph call.

        Args:
            name (str): named of the input Tensor. Used for both the Named VariableFactory/Arg and the Tensor debug name.
            data_iter (Union[Callable[[None], np.ndarray], Iterable[np.ndarray]]):
                Either a function or iterable that generates data for each instance of the input tensor. Each element of
                data should be a HostTensor type (numpy, pytorch, ect.) with the same shape and data type (this is not
                checked at runtime). If you want your data to be the same for all tensor instances wrap it in a
                lambda function e.g. `lambda: data`.
            dtype (Optional[dtypes.dtype], optional): dtype of input. Defaults to None.
            by_ref (bool, optional): Pass the input by reference. Defaults to False.
            replica_grouping (popxl.ReplicaGrouping, optional): variable replica grouping
            shard_over (int, optional): number of replicas in the variable replica group to be used for sharding. If not provided, the variable will be sharded using all replicas.
        """
        tensor, variable_f = add_replica_sharded_variable_input(
            name, data_iter, dtype, by_ref, replica_grouping, shard_over
        )
        self._variable_factories.insert(name, variable_f)
        self._named_inputs.insert(name, tensor)
        return tensor

    def set_name_scope(self, name: str):
        setattr(self, "_name_scope", name)

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, Module):
            # If creating a submodule - inherit it's attributes
            value.set_name_scope(key)
            self._variable_factories.insert(key, value._variable_factories)
            self._named_inputs.insert(key, value._named_inputs)
            self.called_graphs_grad_info.add_submap(value.called_graphs_grad_info)
        return super().__setattr__(key, value)

    def add_variable_inputs(
        self,
        name: Union[str, int],
        variable_f: NamedVariableFactories,
        overwrite: bool = False,
    ) -> NamedTensors:
        """Add NamedVariableFactories as inputs tensors. Returns NamedTensors which can be used in the graph.
            This is useful for reusing a Module within another module. Usage:
            ```
                def build(self, x: popxl.Tensor):
                    args, graph = self.layer.create_graph(x)
                    layer1 = graph.bind(self.add_variable_inputs("1", args))
                    layer2 = graph.bind(self.add_variable_inputs("2", args))
            ```
            this then provides two BoundGraphs `layer1/layer2` which can be called. With each BoundGraph calling the same
            compute Graph but with unique inputs. These inputs being constructed from the `args` NamedVariableFactories.

        Args:
            name (str): name of the variable factories used in the current module.
                If a positive int, it will be cast to a string
            variable_f (NamedVariableFactories): variable factories to be added to the current module.
            overwrite (bool): If True nested variable inputs will be merged and values will be overwritten.

        Returns:
            NamedTensors: Tensors in the current graph for each variable factory in `inputs`.
        """
        if isinstance(name, int):
            if name < 0:
                raise ValueError(f"Name of int type must be positive. Value: {name}")
            name = str(name)
        tensors = {}
        for tensor_name, factory in variable_f.to_dict().items():
            tensors[tensor_name] = factory.create_input()
        tensors = NamedTensors.from_dict(tensors)
        self._variable_factories.insert(name, variable_f.copy(), overwrite=overwrite)
        self._named_inputs.insert(name, tensors, overwrite=overwrite)
        return tensors

    @classmethod
    def from_list(cls, sub_modules: List["Module"]):
        """Create a module from a list of modules"""
        self = cls()
        for i, module in enumerate(sub_modules):
            setattr(self, str(i), module)
        self._sub_modules = sub_modules
        return self

    @classmethod
    def from_variable_factories(cls, inputs: List[VariableFactory]):
        """Create a module with `inputs`"""
        self = cls()
        for i, variable_f in enumerate(inputs):
            tensor = self.add_variable_factory(variable_f, name=str(i))
            setattr(self, str(i), tensor)
        return self

    def get(self, key: Union[str, int]):
        """Get attribute. If an int it will be cast to a string."""
        key = str(key) if isinstance(key, int) else key
        return getattr(self, key)

    def __getitem__(self, item: Union[str, int]):
        return self.get(item)

    def __iter__(self):
        try:
            sub_modules = self._sub_modules
        except AttributeError as e:
            raise AttributeError(
                e,
                "Iterating over `popxl_addons.Module` not created using `popxl_addons.Module.from_list` is not supported.",
            )
        return iter(sub_modules)

    def call_module(
        self,
        module: "Module",
        name: Optional[str] = None,
        autodiff: Literal["auto", "build_grad", "autodiff", "none"] = "auto",
    ):
        """Outline a module and call it: create a graph, create a gradient graph if needed and call the forward graph.

        Example usage:
        ```
        class Example(Module):
            def build(self, x: popxl.Tensor):
                (y,) = self.call_module(Submodule(x.shape[-1]))(x)
                return y
        ```

        Args:
            module (Module): Module to call
            name (Optional[str], optional): Name to use for the module. Defaults to name of the Module's class.
            autodiff: auto: autodiff if grad_graph is implemented otherwise dont. build_grad or autodiff: force autodiff and use this mode. none: don't autodiff Use. Defaults to "auto".
        """
        return _CallModule(self, module, name, autodiff)


class _CallModule:
    def __init__(
        self,
        caller: "Module",
        callee: "Module",
        name: Optional[str],
        autodiff: Literal["auto", "build_grad", "autodiff", "none"],
        grads_provided: Optional[Iterable[popxl.Tensor]] = None,
        grads_required: Optional[Iterable[popxl.Tensor]] = None,
    ):
        self.caller = caller
        self.callee = callee
        self.name = name if name else callee.__class__.__name__
        self.autodiff = autodiff
        self.grads_provided = grads_provided
        self.grads_required = grads_required

    def __call__(self, *args, **kwargs) -> Tuple[popxl.Tensor, ...]:
        var_factories, graph = self.callee.create_graph(*args, **kwargs)
        graph_vars = self.caller.add_variable_inputs(self.name, var_factories)
        self.caller.called_graphs_grad_info.add_submap(graph.called_graphs_grad_info)  # Inherit called_graphs_grad_info
        n_outputs_pre_autodiff = len(graph.graph.outputs)

        # If does not implement grad_graph and autodiff==auto no need to autodiff here as autodiff will do that automatically
        if not (self.autodiff == "none" or (self.autodiff == "auto" and not self.callee.implements_grad_graph)):
            grad_graph = graph.autodiff(
                method=self.autodiff, grads_provided=self.grads_provided, grads_required=self.grads_required
            )
            self.caller.called_graphs_grad_info.add(graph, grad_graph)

        # Autodiff must be applied before call site as it changes the number of outputs of the forward graph
        tensor_outputs = graph.bind(graph_vars).call(*args, **kwargs)
        tensor_outputs = tensor_outputs[:n_outputs_pre_autodiff]

        # Save state for possible reuse
        self.var_factories = var_factories
        self.graph = graph
        self.n_outputs_pre_autodiff = n_outputs_pre_autodiff

        return tensor_outputs


def create_grad_graph_info(
    ir: Ir,
    fwd_graph: popxl.Graph,
    grad_graph: popxl.Graph,
    grad_input_to_fwd_output: OrderedDict[Tensor, Tensor],
    grad_output_to_fwd_input: OrderedDict[Tensor, Tensor],
    fwd_tensors: Set[Tensor],
) -> GradGraphInfo:
    """Create a GradGraphInfo object

    Args:
        ir (Ir): ir
        fwd_graph (popxl.Graph): forward graph
        grad_graph (popxl.Graph): gradient graph
        grad_input_to_fwd_output (OrderedDict[Tensor, Tensor]): An ordered mapping between corresponding gradient graph inputs to forward graph outputs
        grad_output_to_fwd_input (OrderedDict[Tensor, Tensor]): An ordered mapping between corresponding gradient graph outputs to forward graph inputs
        fwd_tensors (Set[Tensor]): A set of Tensors that are outputs of the forward graph and dont require corresponding output gradients. These are usually "aux" tensors

    Returns:
        GradGraphInfo: grad graph info
    """
    # Create grad graph info
    # grad_graph.inputs with the corresponding fwd_graph.outputs
    expected_inputs = [
        _ir.ExpectedConnection(
            fwd_output.id,
            _ir.ExpectedConnectionType.Fwd if fwd_output in fwd_tensors else _ir.ExpectedConnectionType.FwdGrad,
        )
        for grad_input, fwd_output in grad_input_to_fwd_output.items()
    ]

    # grad_graph.outputs should correspond in the same order to fwd_graph.inputs
    expected_outputs = [
        _ir.ExpectedConnection(fwd_input.id, _ir.ExpectedConnectionType.FwdGrad)
        for grad_output, fwd_input in grad_output_to_fwd_input.items()
    ]

    bwd_info = _ir.BwdGraphInfo(grad_graph._pb_graph.id, expected_inputs, expected_outputs)
    grad_info = GradGraphInfo._from_pb(ir._pb_ir, fwd_graph._pb_graph, bwd_info)

    return grad_info


def _normalise_called_graphs_grad_info(
    fwd_graph: GraphLike, grad_graph: GradGraphInfoLike
) -> Tuple[popxl.Graph, GradGraphInfo]:
    """Normalise a (GraphLike, GradGraphInfoLike) association to (popxl.Graph, GradGraphInfo)"""
    fwd_graph = fwd_graph if isinstance(fwd_graph, popxl.Graph) else fwd_graph.graph
    grad_graph = grad_graph if isinstance(grad_graph, GradGraphInfo) else grad_graph.grad_graph_info
    return fwd_graph, grad_graph


def _normalise_called_graphs_grad_info_dict(
    called_graphs_grad_info: Optional[Mapping[GraphLike, GradGraphInfoLike]] = None
) -> Dict[popxl.Graph, GradGraphInfo]:
    """Normalise a (GraphLike, GradGraphInfoLike) dictionary to (popxl.Graph, GradGraphInfo)"""
    called_graphs_grad_info = called_graphs_grad_info if called_graphs_grad_info is not None else {}
    return dict(
        [
            _normalise_called_graphs_grad_info(fwd_graph, grad_graph)
            for fwd_graph, grad_graph in called_graphs_grad_info.items()
        ]
    )
