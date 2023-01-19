# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook

# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import replicated_all_reduce_strided_binding

from typing import Optional, Tuple, List
import popxl
from popxl import Tensor, ReplicaGrouping, ops
from popxl.context import op_debug_context, get_current_context
from popxl.ops.collectives.collectives import CollectiveOps, to_collective_op
from popxl.ops.utils import check_in_graph
from popxl_addons import NamedTensors, GraphWithNamedArgs
from popxl_addons.utils import null_context

__all__ = [
    "replicated_all_reduce_strided",
    "replicated_all_reduce_strided_identical_inputs",
    "replicated_all_reduce_strided_identical_grad_inputs",
    "replicated_all_reduce_strided_graph",
]


@op_debug_context
def replicated_all_reduce_strided(
    t: Tensor, op: CollectiveOps = "add", group: Optional[ReplicaGrouping] = None
) -> Tensor:
    """
    Replicated all reduce.

    Args:
        t (Tensor): Tensor to be reduced
        op (str, optional): Operation to reduce with. 'add' is currently only supported.
        group (ReplicaGrouping, optional): Stride and group size used in the partition of the replicas. Default all replicas.

    Returns:

    """
    return _replicated_all_reduce_strided(
        t,
        op,
        group,
        identical_inputs=False,
        identical_grad_inputs=False,
    )


@op_debug_context
def replicated_all_reduce_strided_identical_inputs(
    t: Tensor, op: CollectiveOps = "add", group: Optional[ReplicaGrouping] = None
) -> Tensor:
    """
    Replicated all reduce.

    You must run the `OpToIdentityPattern` pattern (which is part of `PreAliasPatterns`) after applying autodiff
    for this op to work correctly.

    Args:
        t (Tensor): Tensor to be reduced
        op (str, optional): Operation to reduce with. 'add' is currently only supported.
        group (ReplicaGrouping, optional): Stride and group size used in the partition of the replicas. Default all replicas.

    Returns:

    """
    return _replicated_all_reduce_strided(
        t,
        op,
        group,
        identical_inputs=True,
        identical_grad_inputs=False,
    )


@op_debug_context
def replicated_all_reduce_strided_identical_grad_inputs(
    t: Tensor, op: CollectiveOps = "add", group: Optional[ReplicaGrouping] = None
) -> Tensor:
    """
    Replicated all reduce.

    You must run the `OpToIdentityPattern` pattern (which is part of `PreAliasPatterns`) after applying autodiff
    for this op to work correctly.

    Args:
        t (Tensor): Tensor to be reduced
        op (str, optional): Operation to reduce with. 'add' is currently only supported.
        group (ReplicaGrouping, optional): Stride and group size used in the partition of the replicas. Default all replicas.

    Returns:

    """
    return _replicated_all_reduce_strided(
        t,
        op,
        group,
        identical_inputs=False,
        identical_grad_inputs=True,
    )


def _replicated_all_reduce_strided(
    t: Tensor,
    op: CollectiveOps,
    group: Optional[ReplicaGrouping],
    identical_inputs: bool,
    identical_grad_inputs: bool,
) -> Tensor:
    op_ = to_collective_op(op)  # Only add is currently supported

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if group is not None:
        stride = group.stride
        size = group.group_size
    else:
        stride = 1
        size = g.ir.replication_factor

    settings = ctx._get_op_settings("ReplicatedAllReduceStrided")
    op = replicated_all_reduce_strided_binding.ReplicatedAllReduceStridedOp.createOpInGraph(
        pb_g,
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("replicated_all_reduce_strided_out"),
        },
        op_,
        stride,
        size,
        identical_inputs,
        identical_grad_inputs,
        settings,
    )
    ctx._op_created(op)

    return Tensor._from_pb_tensor(op.outTensor(0))


def replicated_all_reduce_strided_graph(
    tensors: NamedTensors,
    op: CollectiveOps = "add",
    group: Optional[ReplicaGrouping] = None,
    use_io_tiles: bool = False,
) -> Tuple[GraphWithNamedArgs, List[str]]:
    """Create a GraphWithNamedArgs that reduces each Tensor in `tensors` using op.
    The Graph with have a NamedArg for each the tensors in `tensors`.

    Usage:
    ```
        g, names = replicated_all_reduce_strided_graph(tensors)
        ...
        reduced_ts = NamedTensors.pack(names, g.bind(ts).call())
    ```

    Args:
        tensors (NamedTensors): Input Tensors to replica reduced.
        op (CollectiveOps): Operation to use for reduction.
        group (ReplicaGrouping, optional): Stride and group size used in the partition of the replicas. Default all replicas.
        use_io_tiles (bool, optional): If True, tensors will be copied to IO tiles before reducing. Defaults to False.

    Returns:
        Tuple[GraphWithNamedArgs, List[str]]: Created Graph, names of outputs from calling the graph.
    """
    ir = popxl.gcg().ir
    graph = ir.create_empty_graph("replica_reduce")

    names = []
    args = {}

    tile_context = popxl.io_tiles() if use_io_tiles else null_context()

    with graph, popxl.in_sequence(False), tile_context:
        for name, tensor in zip(*tensors.unpack()):
            sg_t = popxl.graph_input(tensor.shape, tensor.dtype, tensor.name, meta_shape=tensor.meta_shape)

            args[name] = sg_t
            names.append(name)

            if use_io_tiles:
                sg_t = ops.io_tile_copy(sg_t)

            sg_t = replicated_all_reduce_strided(sg_t, group, op)

            popxl.graph_output(sg_t)

    return GraphWithNamedArgs(graph, NamedTensors.from_dict(args)), names
