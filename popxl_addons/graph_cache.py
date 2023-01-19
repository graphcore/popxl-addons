# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import weakref
import inspect
from typing import Callable
from functools import wraps
from popxl_addons.utils import OrderedDict

import popxl
from popxl import ops
from popxl.context import get_current_context
from popxl_addons.graph import BoundGraph, GraphWithNamedArgs


def bound_arguments(fn, *args, **kwargs):
    """Return complete arguments from calling `fn` with `*args, **kwargs`.
    Including any bound and default arguments."""
    if hasattr(fn, "__call__"):
        fn = fn.__call__
    if inspect.ismethod(fn):
        args = (fn.__self__, *args)
        fn = fn.__func__
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def tensor_hash(t: popxl.Tensor):
    return hash((t.shape, t.dtype, t.meta_shape))


def argument_hash(arg):
    """Hash an argument to a function. With special handling for Tensors.
    Recursively handles tuple/list/dict arguments."""
    if isinstance(arg, popxl.Tensor):
        return tensor_hash(arg)
    elif isinstance(arg, BoundGraph):
        return hash((arg.graph, argument_hash(arg.args)))
    elif isinstance(arg, GraphWithNamedArgs):
        return hash(arg.graph)
    elif isinstance(arg, (tuple, list, dict)):
        if isinstance(arg, dict):
            items = arg.items()
        else:
            items = zip(range(len(arg)), arg)
        args = OrderedDict()
        for subarg_name, subarg in items:
            args[subarg_name] = argument_hash(subarg)
        return hash(tuple(args.items()))
    return hash(arg)


def get_function_hash(fn, *args, **kwargs):
    """Get the hash for constructing a graph from `fn` passing `*args, **kwargs`."""
    ctx = get_current_context()
    arguments = bound_arguments(fn, *args, **kwargs)
    sig = [fn, ctx.ipu_id, ctx.io_tile_set]
    for arg in arguments.values():
        sig.append(argument_hash(arg))
    return hash(tuple(sig))


class GraphCache:
    """Cache to reuse graphs when the function and arguments to `create_graph` are compatible.
    Usage, replace `ir.create_graph(...)` with:
    ```
        cache = GraphCache()
        with main:
            graph = cache.create_graph(...)
    ```
    """

    def __init__(self):
        self._cache = weakref.WeakKeyDictionary()

    def _get_graph_cache(self, ir: popxl.Ir):
        if ir not in self._cache:
            self._cache[ir] = weakref.WeakValueDictionary()
        return self._cache[ir]

    def create_graph(self, fn, *args, **kwargs) -> popxl.Graph:
        """Returns a popxl.Graph for `fn` in the current IR.
        If a graph has already been constructed for this function and it's associated arguments
        a cached version is returned.

        Warning: Graphs are not immutable and can be changed after construction.
        """
        ir = popxl.gcg().ir
        cache = self._get_graph_cache(ir)
        graph_hash = get_function_hash(fn, *args, **kwargs)
        if graph_hash not in cache:
            cache[graph_hash] = ir.create_graph(fn, *args, **kwargs)
        return cache[graph_hash]


def function(fn: Callable):
    """Outline the execution of a function.
        When it is called, a `popxl.Graph` will be constructed for the function and `ops.call`.
        A GraphCache will be used to ensure graphs are reused where possible.
        Intended for use as a decorator:
        ```
        @function
        def matmul(x, y):
            return x @ y
        ```

    Args:
        fn (Callable): Function to outline.
    """
    cache = GraphCache()

    @wraps(fn)
    def cached_function(*args, **kwargs):
        graph = cache.create_graph(fn, *args, **kwargs)
        # TODO: Filter out non-Tensor arguments
        return ops.call(graph, *args, **kwargs)

    return cached_function
