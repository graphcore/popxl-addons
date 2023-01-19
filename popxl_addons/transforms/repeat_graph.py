# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Dict, List

import popxl
from popart._internal.ir import Tensor as Tensor_, Op, TopoCons
from popxl import Tensor, Graph, ops
from popxl.context import get_current_context


def repeat_graph(graph: popxl.Graph, repeat_count: int) -> Graph:
    """
    Wrap `graph` in a loop. The `graph` is operated on in-place. `graph` can be the main graph.

    Move all ops in `graph` into a new graph and repeatedly call the graph `repeat_count` times.
    Variables are left in `graph` and added as inputs (by ref) to the new graph.
    The new graph is returned.

    If `repeat_count` <= 1 then this transform does nothing and returns `graph`.
    """
    if repeat_count <= 1:
        return graph

    old_to_new_t: Dict[Tensor_, Tensor_] = {}
    new_to_old_inputs: Dict[Tensor, Tensor] = {}
    old_to_new_ops: Dict[Op, List[Op]] = {}

    repeat_graph = graph.ir.create_empty_graph("repeat_graph")

    with repeat_graph:
        # Obtain the ops in topological order
        graph_ops = graph._pb_graph.getOpSchedule()
        for op in graph_ops:
            op: Op
            cloned_op = op.cloneIntoGraph(repeat_graph._pb_graph)

            # Create inputs
            for idx, t_ in op.getInputIndexMap().items():
                t_: Tensor_
                if t_ not in old_to_new_t:
                    t = Tensor._from_pb_tensor(t_)
                    new_t = popxl.graph_input(**t.spec, name=t.name, by_ref=True)
                    old_to_new_t[t_] = new_t._pb_tensor
                    new_to_old_inputs[new_t] = t
                else:
                    new_t = old_to_new_t[t_]
                cloned_op.connectInTensor(idx, new_t.id)

            # Create outputs
            for idx, t_ in op.getOutputIndexMap().items():
                t_: Tensor_
                t = Tensor._from_pb_tensor(t_)
                cloned_op.createAndConnectOutTensor(idx, t.name)
                old_to_new_t[t_] = cloned_op.outTensor(idx)

            cloned_op.setup()

            old_to_new_ops[op] = [cloned_op]

    with graph, popxl.in_sequence("pass"):
        repeat_info = ops.repeat_with_info(repeat_graph, repeat_count, inputs_dict=new_to_old_inputs)
        repeat_Op = repeat_info._op

    # Copy topological constraints.
    topcons: TopoCons = graph._pb_graph.topoCons()
    topcons.transferToSubgraph(repeat_Op, old_to_new_ops, True)

    # Add repeat op (LoopOp) topological constraint if needed
    get_current_context()._add_in_sequence_topocons(repeat_Op)

    # Delete ops in main_graph
    for op in graph_ops:
        op.disconnectAllInputs()
        op.disconnectAllOutputs()
        graph._pb_graph.eraseOp(op.id)

    # TODO: prevent it from removing IO tensors
    # Remove tensors that are now dangling with no connected ops in the old graph
    # graph._pb_graph.removeIsolatedTensors(True, True, True, True)

    return repeat_graph
