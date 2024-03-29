# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from collections import defaultdict
from typing import Iterable, Optional, Union, List, Tuple, TYPE_CHECKING, Mapping
from typing_extensions import Literal
import popxl
from popxl.context import debug_context_frame_offset
from popxl import ops
from popxl.ops.call import CallSiteInfo
from popxl_addons.dot_tree import to_mapping
from popxl_addons.named_tensors import NamedTensors, TensorMap
from popxl.transforms.autodiff import GradGraphInfo
from popxl_addons.graph_common import GraphLike, GradGraphInfoLike

if TYPE_CHECKING:
    from popxl_addons.module import Module, _Aux


class BoundGraph:
    """Container of a popxl.Graph and a TensorMap of bound Tensor inputs"""

    def __init__(self, graph: popxl.Graph, args: Optional[TensorMap] = None):
        self.graph = graph
        self.args = dict(args or {})

    @debug_context_frame_offset(1)
    def call(
        self, *targs: Union[popxl.Tensor, List[popxl.Tensor], int, float], args: Optional[TensorMap] = None
    ) -> Tuple[popxl.Tensor, ...]:
        """Call the bound graph.

        Args:
            *targs (Union[popxl.Tensor, List[popxl.Tensor]]): Positional Tensor arguments
            args (Optional[TensorMap], optional): Optional Tensor arguments. Will overwrite any bound arguments for the
                same graph input Tensor. Defaults to None.

        Returns:
            Output of ops.call
        """
        call_info = self.call_with_info(*targs, args=args)
        return call_info.outputs

    @debug_context_frame_offset(1)
    def call_with_info(
        self, *targs: Union[popxl.Tensor, List[popxl.Tensor], int, float], args: Optional[TensorMap] = None
    ) -> CallSiteInfo:
        """Call the bound graph and return call info object.

        Args:
            *targs (Union[popxl.Tensor, List[popxl.Tensor]]): Positional Tensor arguments
            args (Optional[TensorMap], optional): Optional Tensor arguments. Will overwrite any bound arguments for the
                same graph input Tensor. Defaults to None.

        Returns:
            Output of ops.call_with_info
        """
        inputs_dict = {**self.args, **(args or {})}
        return ops.call_with_info(self.graph, *targs, inputs_dict=inputs_dict)

    def bind(self, args: TensorMap) -> "BoundGraph":
        """Rebind the graph. The new bound Tensors will be the current bound tensors and args

        Args:
            args (TensorMap): Additional Tensors to bind. Any graph input Tensors already bound will be overwritten.

        Returns:
            BoundGraph: A new bound graph.
        """
        return BoundGraph(self.graph, {**self.args, **args})

    def subgraph_io_map(
        self, *targs: popxl.Tensor, args: Optional[TensorMap] = None, outs: Iterable[popxl.Tensor] = None
    ):
        """Return a map of:
            * graph input Tensors to call inputs (*targs, args, self.args)
            * graph output Tensors to call outputs (outs)

        Args:
            *targs (popxl.Tensor): Positional call inputs
            args (Optional[TensorMap], optional): Optional call inputs TensorMap. Defaults to None.
            outs (Iterable[popxl.Tensor], optional): Call outputs. Defaults to None.

        Returns:
            TensorMap
        """
        inputs_dict = {**self.args, **(args or {})}
        for idx, targ in enumerate(targs):
            inputs_dict[popxl.Tensor._from_pb_tensor(self.graph._pb_graph.getInputTensor(idx))] = targ

        subgraph_out_to_parent_out = {}
        for idx, tout in enumerate(outs):
            subgraph_out_to_parent_out[popxl.Tensor._from_pb_tensor(self.graph._pb_graph.getOutputTensor(idx))] = tout

        return {**subgraph_out_to_parent_out, **inputs_dict}


class GraphWithNamedArgs:
    def __init__(
        self,
        graph: popxl.Graph,
        args: Optional[NamedTensors] = None,
        grad_graph_info: Optional[GradGraphInfo] = None,
        from_module: "Optional[Module]" = None,
        aux: "Optional[_Aux]" = None,
        called_graphs_grad_info: Optional[Mapping["GraphLike", "GradGraphInfoLike"]] = None,
    ):
        """
        Container of `popxl.Graph` and `NamedTensors`. The named tensors are members of the graph.
        Method 'bind' can be used to create BoundGraphs.

        If the graph is a grad graph, `GradGraphInfo` object can also be included.

        Args:
            graph (popxl.Graph): Graph
            args (Optional[NamedTensors]): Optional named tensors used for graph inputs.
            grad_graph_info (Optional[popxl.GradGraphInfo]): Optional `popxl.GradGraphInfo` object if a grad graph
            from_module (Optional[Module]): Module that created this graph
            aux (Optional[_Aux]): Auxiliary tensor outputs that can be used when creating a gradient graph
            called_graphs_grad_info (Optional[Mapping["GraphLike", "GradGraphInfoLike"]]): Mapping between forward and gradient graphs that can be used when autodiffing this graph
        """
        from .module import _Aux

        self.graph = graph
        self.args = args or NamedTensors()
        self._grad_graph_info = grad_graph_info
        self._from_module = from_module
        self._aux = aux if aux is not None else _Aux()
        self.called_graphs_grad_info = called_graphs_grad_info if called_graphs_grad_info is not None else {}
        self.autodiff_applied = False  # Has the graph already had autodiff applied to it

    @classmethod
    def from_grad_graph(cls, grad_graph_info: GradGraphInfo, args: Optional[NamedTensors] = None):
        """Container of `GradGraphInfo` and `NamedTensors`"""
        return cls(grad_graph_info.graph, args, grad_graph_info)

    def bind(self, args: Optional[NamedTensors] = None) -> BoundGraph:
        """Create a BoundGraph

        Args:
            args (Optional[NamedTensors], optional): Named arguments to bind. Names should match those of self.args. Defaults to None.

        Returns:
            BoundGraph
        """
        tensor_map = to_mapping(self.args, args) if args else {}
        return BoundGraph(self.graph, tensor_map)

    def call(
        self, *targs: Union[popxl.Tensor, List[popxl.Tensor], int, float], args: Optional[TensorMap] = None
    ) -> Tuple[popxl.Tensor, ...]:
        """Call self.graph with no bound arguments.

        Args:
            *targs (Union[popxl.Tensor, List[popxl.Tensor]]): Positional arguments
            args (Optional[TensorMap], optional): Optional Tensor arguments. Defaults to None.

        Returns:
            Output of ops.call
        """
        return self.bind().call(*targs, args=args)

    def call_with_info(
        self, *targs: Union[popxl.Tensor, List[popxl.Tensor], int, float], args: Optional[TensorMap] = None
    ) -> CallSiteInfo:
        """Call self.graph with no bound arguments. Output call with info object.

        Args:
            *targs (Union[popxl.Tensor, List[popxl.Tensor]]): Positional arguments
            args (Optional[TensorMap], optional): Optional Tensor arguments. Defaults to None.

        Returns:
            Output of ops.call_with_info
        """
        return self.bind().call_with_info(*targs, args=args)

    def autodiff(
        self,
        grads_provided: Optional[Iterable[popxl.Tensor]] = None,
        grads_required: Optional[Iterable[popxl.Tensor]] = None,
        called_graphs_grad_info: Optional[Mapping["GraphLike", "GradGraphInfoLike"]] = None,
        method: Literal["auto", "build_grad", "autodiff"] = "auto",
        force: bool = False,
    ) -> "GraphWithNamedArgs":
        """Autodiff this graph to create a corresponding gradient graph. If the module this graph was created from implements the `build_grad` method then it will use this to create a gradient graph (see `Module.create_grad_graph` for more details.). Otherwise the normal autodiff transform is used.

        Args:
            grads_provided (Optional[Iterable[popxl.Tensor]], optional): Specifies the inputs of the gradient graph `self.build_graph` which should be forward graph output tensors. Defaults to all forward graph outputs.
            grads_required (Optional[Iterable[popxl.Tensor]], optional): Specifies the outputs of the gradient graph `self.build_graph` which should be forward graph output tensors. Defaults to all forward graph outputs.
            called_graphs_grad_info (Optional[Mapping[GraphLike, GradGraphInfoLike]], optional): A mapping between called graphs and the corresponding backward graphs. This object is combined with `self.called_graphs_grad_info` before passing to autodiff or create_gradient_graph. Defaults to None.
            method: Mode to use. Defaults to "auto".
            force (bool): If False it will prevent you from autodiffing the same forward graph twice which will usually not work. Defaults to False.

        Returns:
            GraphWithNamedArgs: Gradient graph which also holds the `grad_graph_info` object
        """
        from popxl_addons.transforms.autodiff import autodiff

        if method not in ("auto", "build_grad", "autodiff"):
            raise ValueError(f"`method` should be one of 'auto', 'build_grad' or 'autodiff'. Not: {method}")

        if self.autodiff_applied and not force:
            raise Exception(
                "Autodiff has already been applied to the graph, as autodiff modifies the forward graph in-place, "
                "applying autodiff multiple times does not give the same effect. "
                "Use `force=True` if you want to ignore this guardrail."
            )

        if method in ("auto", "build_grad"):
            if self._from_module and self._from_module.implements_grad_graph:
                grad_graph = self._from_module.create_grad_graph(
                    fwd_graph=self, grads_provided=grads_provided, grads_required=grads_required
                )
                self.autodiff_applied = True
                return grad_graph
            if method == "build_grad":
                raise Exception(
                    "The default implementation of `build_graph` should be overridden and `from_module` is specified when this object was initialised."
                )

        grad_graph = autodiff(
            graph=self,
            grads_provided=grads_provided,
            grads_required=grads_required,
            called_graphs_grad_info=called_graphs_grad_info,
        )
        self.autodiff_applied = True
        return grad_graph

    def copy(self):
        """Shallow Copy

        Returns:
            GraphWithNamedArgs
        """
        return GraphWithNamedArgs(self.graph, self.args.copy(), self._grad_graph_info)

    @property
    def grad_graph_info(self):
        """GradGraphInfo is available if the graph is a gradient graph (created using autodiff)"""
        if self._grad_graph_info is not None:
            return self._grad_graph_info
        raise AttributeError(f"`grad_graph_info` attribute does not exist. Not a gradient graph.")

    @grad_graph_info.setter
    def grad_graph_info(self, grad_graph_info: GradGraphInfo):
        self._grad_graph_info = grad_graph_info

    def print_schedule(self):
        self.graph = self.graph

        _id_counter = 0

        def next_ids():
            nonlocal _id_counter
            _id_counter += 1
            return _id_counter

        ids = defaultdict(next_ids)

        def tensor_str(t):
            # TODO: Give each id a random color
            t = popxl.Tensor._from_pb_tensor(t)
            return f"%{ids[t.id]} [{t.shape} {t.dtype.name}]"

        ss_graph_name = f"Graph : {self.graph.name}"

        inputs = []
        names, tensors = self.args.unpack()
        for t in self.graph.inputs:
            try:
                idx = tensors.index(t)
                inputs.append(f"{names[idx]}=%{ids[t.id]}")
            except ValueError:
                inputs.append(f"%{ids[t.id]}")
        ss_inputs = ", ".join(inputs)

        ops = []
        for op in self.graph._pb_graph.getOpSchedule(True):
            inputs = "(" + (", ".join(tensor_str(t) for t in op.getInputTensors())) + ")"
            outputs = "(" + (", ".join(tensor_str(t) for t in op.getOutputTensors())) + ")"
            id = f"{op.opType()}.{op.id}"
            if hasattr(op, "getCalledGraph"):
                id += f"({op.getCalledGraph().id})"
            ops.append(" ".join((id, inputs, "->", outputs)))
        ss_ops = "\n".join(ops)

        outputs = []
        for t in self.graph.outputs:
            outputs.append(f"%{ids[t.id]}")
        ss_outputs = ", ".join(outputs)

        ss = ss_graph_name + "\n"
        ss += f"  ({ss_inputs}) -> ({ss_outputs}) " + "{\n    "
        ss += ss_ops.replace("\n", "\n    ") + "\n  }"
        return ss
