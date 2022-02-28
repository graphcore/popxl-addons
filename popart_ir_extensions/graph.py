# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from collections import defaultdict
from typing import Iterable, Optional, Union, List, Tuple
import popart.ir as pir
from popart.ir.context import debug_context_frame_offset
import popart.ir.ops as ops
from popart.ir.ops.call import CallSiteInfo
from popart_ir_extensions.dot_tree import to_mapping
from popart_ir_extensions.named_tensors import NamedTensors, TensorMap
from popart.ir.transforms.autodiff import GradGraphInfo


class BoundGraph:
    """Container of a pir.Graph and a TensorMap of bound Tensor inputs"""

    def __init__(self, graph: pir.Graph, args: Optional[TensorMap] = None):
        self.graph = graph
        self.args = dict(args or {})

    def call(self, *targs: Union[pir.Tensor, List[pir.Tensor]],
             args: Optional[TensorMap] = None) -> Tuple[pir.Tensor, ...]:
        """Call the bound graph.

        Args:
            *targs (Union[pir.Tensor, List[pir.Tensor]]): Positional Tensor arguments
            args (Optional[TensorMap], optional): Optional Tensor arguments. Will overwrite any bound arguments for the
                same graph input Tensor. Defaults to None.

        Returns:
            Output of ops.call
        """
        call_info = self.call_with_info(*targs, args=args)
        return call_info.outputs

    @debug_context_frame_offset(1)
    def call_with_info(self, *targs: Union[pir.Tensor, List[pir.Tensor]],
                       args: Optional[TensorMap] = None) -> CallSiteInfo:
        """Call the bound graph and return call info object.

        Args:
            *targs (Union[pir.Tensor, List[pir.Tensor]]): Positional Tensor arguments
            args (Optional[TensorMap], optional): Optional Tensor arguments. Will overwrite any bound arguments for the
                same graph input Tensor. Defaults to None.

        Returns:
            Output of ops.call_with_info
        """
        inputs_dict = {**self.args, **(args or {})}
        return ops.call_with_info(self.graph, *targs, inputs_dict=inputs_dict)

    def bind(self, args: TensorMap) -> 'BoundGraph':
        """Rebind the graph. The new bound Tensors will be the current bound tensors and args

        Args:
            args (TensorMap): Additional Tensors to bind. Any graph input Tensors already bound will be overwritten.

        Returns:
            BoundGraph: A new bound graph.
        """
        return BoundGraph(self.graph, {**self.args, **args})

    def subgraph_io_map(self, *targs: pir.Tensor, args: Optional[TensorMap] = None, outs: Iterable[pir.Tensor] = None):
        """Return a map of:
            * graph input Tensors to call inputs (*targs, args, self.args)
            * graph output Tensors to call outputs (outs)

        Args:
            *targs (pir.Tensor): Positional call inputs
            args (Optional[TensorMap], optional): Optional call inputs TensorMap. Defaults to None.
            outs (Iterable[pir.Tensor], optional): Call outputs. Defaults to None.

        Returns:
            TensorMap
        """
        inputs_dict = {**self.args, **(args or {})}
        for idx, targ in enumerate(targs):
            inputs_dict[pir.Tensor._from_pb_tensor(self.graph._pb_graph.getInputTensor(idx))] = targ

        subgraph_out_to_parent_out = {}
        for idx, tout in enumerate(outs):
            subgraph_out_to_parent_out[pir.Tensor._from_pb_tensor(self.graph._pb_graph.getOutputTensor(idx))] = tout

        return {**subgraph_out_to_parent_out, **inputs_dict}


class GraphWithNamedArgs:
    def __init__(self,
                 graph: pir.Graph,
                 args: Optional[NamedTensors] = None,
                 grad_graph_info: Optional[GradGraphInfo] = None):
        """
        Container of `pir.Graph` and `NamedTensors`. The named tensors are members of the graph.
        Method 'bind' can be used to create BoundGraphs.

        If the graph is a grad graph, `GradGraphInfo` object can also be included.

        Args:
            graph (pir.Graph): Graph
            args (Optional[NamedTensors]): Optional named tensors used for graph inputs.
            grad_graph_info (Optional[pir.GradGraphInfo]): Optional `pir.GradGraphInfo` object if a grad graph
        """
        self.graph = graph
        self.args = args or NamedTensors()
        self._grad_graph_info = grad_graph_info

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

    def call(self, *targs: Union[pir.Tensor, List[pir.Tensor]],
             args: Optional[TensorMap] = None) -> Tuple[pir.Tensor, ...]:
        """Call self.graph with no bound arguments.

        Args:
            *targs (Union[pir.Tensor, List[pir.Tensor]]): Positional arguments
            args (Optional[TensorMap], optional): Optional Tensor arguments. Defaults to None.

        Returns:
            Output of ops.call
        """
        return self.bind().call(*targs, args=args)

    def call_with_info(self, *targs: Union[pir.Tensor, List[pir.Tensor]],
                       args: Optional[TensorMap] = None) -> CallSiteInfo:
        """Call self.graph with no bound arguments. Output call with info object.

        Args:
            *targs (Union[pir.Tensor, List[pir.Tensor]]): Positional arguments
            args (Optional[TensorMap], optional): Optional Tensor arguments. Defaults to None.

        Returns:
            Output of ops.call_with_info
        """
        return self.bind().call_with_info(*targs, args=args)

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
        raise AttributeError(f'`grad_graph_info` attribute does not exist. Not a gradient graph.')

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
            t = pir.Tensor._from_pb_tensor(t)
            return f"%{ids[t.id]} [{t.shape} {t.dtype._name}]"

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
            ops.append(" ".join((f"{op.opType()}.{op.id}", inputs, "->", outputs)))
        ss_ops = "\n".join(ops)

        outputs = []
        for t in self.graph.outputs:
            outputs.append(f"%{ids[t.id]}")
        ss_outputs = ", ".join(outputs)

        ss = ss_graph_name + "\n"
        ss += f"  ({ss_inputs}) -> ({ss_outputs}) " + "{\n    "
        ss += ss_ops.replace("\n", "\n    ") + "\n  }"
        return ss
