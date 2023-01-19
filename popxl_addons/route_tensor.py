# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import popart._internal.ir as _ir
from popxl.context import get_current_graph
from popxl.tensor import Tensor, graph_input
from popxl.graph import Graph

__all__ = ["route_tensor_into_graph", "is_subgraph"]


def is_subgraph(parent: "Graph", subgraph: "Graph"):
    """Is `parent` a subgraph of `subgraph`?"""
    graphs = [parent._pb_graph]
    while graphs:
        g = graphs.pop()
        if g.id == subgraph._pb_graph.id:
            return True
        graphs += g.getCalledGraphs()
    return False


def called_graph_paths(parent_graph: Graph, subgraph: Graph) -> Set[Tuple[_ir.GraphId, ...]]:
    """Finds all paths from `parent_graph` to `subgraph` by traversing the graph of `popxl.Graph`s."""
    graph_paths: Set[Tuple[_ir.GraphId, ...]] = set()
    search_graphs: List[Tuple[_ir.Graph, Tuple[_ir.GraphId, ...]]] = [(parent_graph._pb_graph, ())]

    while len(search_graphs):
        g, path = search_graphs.pop()
        path = path + (g.id,)
        if g.id == subgraph._pb_graph.id:
            graph_paths.add(path)
        for sg in g.getCalledGraphs():
            search_graphs.append((sg, path))
    return graph_paths


def connect_call_op(
    op: _ir.op.CallOp, tensor: Tensor, input_tensors: Dict[_ir.GraphId, Tensor], modified: Iterable[_ir.view.Region]
):
    """Given a CallOp, add a new input to the callsite and called graph if not already present.
        Additionally set if the input is modified.

    Args:
        op (_ir.op.CallOp): Instance of a popart CallOp.
        tensor (Tensor): Tensor in the same graph as `op`
        input_tensors (Dict[_ir.GraphId, Tensor]): Mapping of previously connected Tensors.
                                                   This will be updated if an input is connected.
        modified (Iterable[_ir.view.Region]): Regions of `tensor` that should be marked as modified.
    """

    sg = Graph._from_pb(op.getCalledGraphs()[0])
    op_idx = None
    sg_tensor = input_tensors.get(sg._pb_graph.id, None)
    if sg_tensor is None:
        if op.hasInputTensor(tensor._pb_tensor):
            sg_tensor = Tensor._from_pb_tensor(
                sg._pb_graph.getInputTensor(op.opInToSubgraphInIndex(op.inIndex(tensor._pb_tensor)))
            )
        else:
            with sg:
                sg_tensor = graph_input(tensor.shape, tensor.dtype, tensor.name)

    if op_idx is None:
        op_idx = op.subgraphInToOpInIndex(sg._pb_graph.getInputIndex(sg_tensor.id))

    if not op.hasInput(op_idx):
        op.connectInTensor(op_idx, tensor.id)

    op.addModified(op_idx, modified)

    input_tensors[sg._pb_graph.id] = sg_tensor


def connect_loop_op(
    op: _ir.op.LoopOp, tensor: Tensor, input_tensors: Dict[_ir.GraphId, Tensor], modified: Iterable[_ir.view.Region]
):
    """Given a LoopOp, add a new input to the callsite and called graph if not already present.
        Additionally set if the input is modified.

    Args:
        op (_ir.op.LoopOp): Instance of a popart LoopOp.
        tensor (Tensor): Tensor in the same graph as `op`
        input_tensors (Dict[_ir.GraphId, Tensor]): Mapping of previously connected Tensors.
                                                   This will be updated if an input is connected.
        modified (Iterable[_ir.view.Region]): Regions of `tensor` that should be marked as modified.
    """
    sg = Graph._from_pb(op.getCalledGraphs()[0])
    op_idx = None
    sg_tensor = input_tensors.get(sg._pb_graph.id, None)
    if sg_tensor is None:
        if op.hasInputTensor(tensor._pb_tensor):
            sg_tensor = Tensor._from_pb_tensor(
                sg._pb_graph.getInputTensor(op.opInToSubgraphInIndex(op.inIndex(tensor._pb_tensor)))
            )
        else:
            op_idx = op.getNumExplicitInputs()
            sg_tensor_id = sg._create_tensor_id(tensor.name)
            op.addLoopInput(op_idx, tensor.id, sg_tensor_id, False)
            sg_tensor = sg.get_tensor(sg_tensor_id)

    if op_idx is None:
        op_idx = op.subgraphInToOpInIndex(sg._pb_graph.getInputIndex(sg_tensor.id))

    if not op.hasInput(op_idx):
        op.connectInTensor(op_idx, tensor.id)

    op.addModified(op_idx, modified)

    input_tensors[sg._pb_graph.id] = sg_tensor


def connect_tensor_to_ops_on_path(
    path: Tuple[_ir.GraphId, ...],
    from_tensor: Tensor,
    input_tensors: Dict[_ir.GraphId, Tensor],
    modified: Iterable[_ir.view.Region],
):
    """For a path of graphs on a call tree, add an input tensor to all callsites and called graphs on the path.
        Example, we have the following call tree:
        ```
        with main:
            a = popxl.variable(1)
            ops.call(sg)
            with sg:
                ops.call(sg1)
                with sg1:
                    a_ = route_tensor_into_graph(a)
        ```
        There is a `path` of `main` -> `sg` -> `sg1`. To be able to route `a` into `sg1` we have to:
        * handle `main` -> `sg`
          * add a new input to Graph `sg`: `sg/a`
          * add Tensor `a` as an input to the callsite `ops.call(sg)` which corresponds to the new input `sg/a`
        * handle `sg` -> `sg1`
          * add a new input to Graph `sg1`: `sg1/a`
          * add Tensor `sg/a` as an input to the callsite `ops.call(sg1)` which corresponds to the new input `sg1/a`
        * return `sg1/a` to be assigned to `a_`

    Args:
        path (Tuple[_ir.GraphId, ...]): Path of graphs from a call tree
        from_tensor (Tensor): Tensor to start from.
        input_tensors (Dict[_ir.GraphId, Tensor]): Mapping of previously connected Tensors.
        modified (Iterable[_ir.view.Region]): Regions of `tensor` that should be marked as modified.
    """
    parent_tensor = from_tensor
    for gid in path:
        for op in parent_tensor._pb_tensor.getGraph().getOps():
            if gid in map(lambda g: g.id, op.getCalledGraphs()):
                if isinstance(op, _ir.op.CallOp):
                    connect_call_op(op, parent_tensor, input_tensors, modified)
                elif isinstance(op, _ir.op.LoopOp):
                    connect_loop_op(op, parent_tensor, input_tensors, modified)

        parent_tensor = input_tensors.get(gid, None)
        if parent_tensor is None:
            raise RuntimeError(f"Could not move {from_tensor} into graph {gid.str()}")


def _route_into_graph(to_graph: Graph, from_tensor: Tensor, modified: Iterable[_ir.view.Region]) -> Tensor:
    """Internal implementation of `route_tensor_into_graph`"""
    from_graph = Graph._from_pb(from_tensor._pb_tensor.getGraph())

    graph_paths = called_graph_paths(from_graph, to_graph)

    if not graph_paths:
        raise ValueError("to_graph is not a subgraph of from_tensor's graph.")

    # Dictionary to track tensor representations of from_tensor
    # in subgraphs
    subgraph_tensors: Dict[_ir.GraphId, Tensor] = {}

    for path in graph_paths:
        # First value in path will always be `to_graph` so we can skip it.
        connect_tensor_to_ops_on_path(path[1:], from_tensor, subgraph_tensors, modified)

    return subgraph_tensors[to_graph._pb_graph.id]


def route_tensor_into_graph(
    tensor: Tensor, graph: Optional[Graph] = None, modified: Union[bool, Iterable[_ir.view.Region]] = False
):
    """Add graph inputs to access a Tensor from a parent graph to a graph if possible.
        This is achieved by recusively connecting `tensor` to callsites within `tensor`'s Graph.
        Example use case:
            Access a Variable from the main graph within a sub-subgraph
            ```
            with main:
                a = popxl.variable(1)
                ops.repeat(sg)
                with sg:
                    ops.call(sg1)
                    with sg1:
                        a_ = route_tensor_into_graph(a, modified=True)
                        ops.var_updates.accumulate_(a_, popxl.constant(1))
            ```


    Args:
        tensor (Tensor): Tensor to be accessed
        graph: (Optional[Graph]): Graph to access the tensor from. By default, the current graph scope will be used.
        modifed (bool): Any modifications on the tensor in the self graph will be applyed to the input tensor

    Raises:
        ValueError: If the tensor's graph is not a parent graph of self then an error will be raised

    Returns:
        Tensor: tensor in `graph`
    """
    if graph is None:
        graph = get_current_graph()
    if tensor not in graph:
        t_graph = Graph._from_pb(tensor._pb_tensor.getGraph())
        if not is_subgraph(t_graph, graph):
            raise ValueError(
                f"{tensor} is not in the graph {graph.name}. "
                "Additionally, the tensor is not in a known parent graph so cannot "
                "be moved into the graph."
            )
        if isinstance(modified, bool):
            modified = [_ir.view.Region.getFull(tensor.shape) if modified else _ir.view.Region.getEmpty(tensor.rank)]
        tensor = _route_into_graph(graph, tensor, modified)
    return tensor
