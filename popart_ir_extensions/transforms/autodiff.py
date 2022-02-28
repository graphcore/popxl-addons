# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Dict, Iterable, List, Optional, Mapping, Tuple

import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.transforms.autodiff import (autodiff as _autodiff, GradGraphInfo)
from popart_ir_extensions.dot_tree import sanitise

from popart_ir_extensions.input_factory import InputFactory, NamedInputFactories, add_input_tensor
from popart_ir_extensions.named_tensors import NamedTensors
from popart_ir_extensions.graph import GraphWithNamedArgs

__all__ = ["autodiff", "autodiff_with_accumulation"]


def _autodiff_with_patterns(
        graph: pir.Graph,
        grads_provided: Optional[Iterable[pir.Tensor]] = None,
        grads_required: Optional[Iterable[pir.Tensor]] = None,
        called_graphs_grad_info: Optional[Mapping[pir.Graph, GradGraphInfo]] = None) -> GradGraphInfo:
    """
    Extension Autodiff.
    This method calls `pir.transforms.autodiff` and then some required patterns after to ensure the returned
    grad graph is lowerable.

    Args:
        graph (pir.Graph)
        grads_provided (Optional[Iterable[pir.Tensor]], optional). Defaults to all outputs of the provided graph.
        grads_required (Optional[Iterable[pir.Tensor]], optional). Defaults to all inputs of the provided graph.
        called_graphs_grad_info (Optional[Mapping[pir.Graph, GradGraphInfo]], optional). Defaults to None.

    Returns:
        GradGraphInfo: grad graph of `graph`
    """
    grad_info_all: Dict[pir.Graph, GradGraphInfo] = _autodiff(graph,
                                                              grads_provided=grads_provided,
                                                              grads_required=grads_required,
                                                              called_graphs_grad_info=called_graphs_grad_info,
                                                              return_all_grad_graphs=True)  # type: ignore
    grad_info = grad_info_all[graph]

    ir = grad_info.graph.ir._pb_ir
    # TODO: Only run required patterns
    ir.setPatterns(_ir.patterns.Patterns(_ir.patterns.PatternsLevel.Default))

    for grad_info_i in grad_info_all.values():
        ir.applyPreAliasPatterns(grad_info_i.graph._pb_graph)
        # TODO: Should inplacing be run?
        #       If not, we can end up with lots of outplace identities
        ir.applyInplacePattern(grad_info_i.graph._pb_graph)

    return grad_info


def autodiff(
        graph: GraphWithNamedArgs,
        grads_provided: Optional[Iterable[pir.Tensor]] = None,
        grads_required: Optional[Iterable[pir.Tensor]] = None,
) -> GraphWithNamedArgs:
    """
    Extension Autodiff.
    This method calls `pir.transforms.autodiff` and then some required patterns after to ensure the returned
    grad graph is lowerable.

    Args:
        graph (pir.Graph)
        grads_provided (Optional[Iterable[pir.Tensor]], optional). Defaults to all outputs of the provided graph.
        grads_required (Optional[Iterable[pir.Tensor]], optional). Defaults to all inputs of the provided graph.

    Returns:
        GraphWithNamedArgs: grad graph of `graph` wrapped in a GraphWithNamedArgs with grad graph info
    """
    grad_info = _autodiff_with_patterns(graph.graph, grads_provided=grads_provided, grads_required=grads_required)
    return GraphWithNamedArgs.from_grad_graph(grad_info)


def autodiff_with_accumulation(
        graph: GraphWithNamedArgs,
        tensors_to_accumulate_grads: Iterable[pir.Tensor],
        grads_required: Optional[Iterable[pir.Tensor]] = None) -> Tuple[NamedInputFactories, GraphWithNamedArgs]:
    """
    Calls autodiff and then for each tensor in `tensors_to_accumulate_grads` adds an operation to the output gradient
    graph which takes a running mean of the tensor and the result stored in an accumator tensor. The accumulators are
    added as NamedArgs TensorByRef inputs to the grad graph and the corresponding output of the original tensor removed.

    This is known as a Gradient Accumulation Step (GAS) for pipeline or batch serialisation execution.

    The NamedArg `mean_accum_counter` counts the number of accumulated tensors, which is incremented with each call of
    the gradient graph. To reset the running mean it's sufficient to just reset this counter
    i.e. `ops.var_updates.accumulator_zero_(grad_args.mean_accum_counter)`.

    Args:
        graph (pir.Graph)
        tensors_to_accumulate_grads (Iterable[pir.Tensor]). Tensors to accumulate and calculate a running mean.
        grads_required (Optional[Iterable[pir.Tensor]], optional). Defaults to all inputs of the provided graph.

    Returns:
        NamedInputFactories: input factories for the accumulation tensors (initialised as zeros) needed for graph inputs
        GraphWithNamedArgs: grad graph of `graph` with NamedArgs
        GradGraphInfo: grad graph of `graph`
    """

    grads_required = list(grads_required or [])
    grads_required += tensors_to_accumulate_grads

    # Autodiff the graph.
    grad_info = _autodiff_with_patterns(graph.graph, grads_required=grads_required)

    expected_outputs = grad_info.outputs

    indices_to_remove: List[int] = []

    named_inputs: Dict[str, pir.Tensor] = {}
    input_factories: Dict[str, InputFactory] = {}

    # Flatten named tensors
    names = {t: name for name, t in graph.args.named_tensors.items()}

    def add_input(name, *args, **kwargs):
        t, f = add_input_tensor(name, *args, **kwargs)
        named_inputs[name] = t
        input_factories[name] = f
        return t

    with grad_info.graph, pir.in_sequence(True):
        counter = add_input("mean_accum_counter", partial(np.zeros, shape=()), pir.float32, by_ref=True)

        with pir.in_sequence(False):
            for tensor in tensors_to_accumulate_grads:
                idx = expected_outputs.index(tensor)
                indices_to_remove.append(idx)
                subgraph_tensor = pir.Tensor._from_pb_tensor(grad_info.graph._pb_graph.getOutputTensor(idx))

                name = names.get(tensor, sanitise(tensor.name))

                accum = add_input(name, partial(np.zeros, shape=tensor.shape), tensor.dtype, by_ref=True)
                ops.var_updates.accumulate_mean_(accum, subgraph_tensor, counter)

        ops.var_updates.accumulate_(counter, pir.constant(1, pir.float32))

    for idx in indices_to_remove:
        grad_info.graph._pb_graph.removeOutput(idx)

    return (NamedInputFactories.from_dict(input_factories),
            GraphWithNamedArgs.from_grad_graph(grad_info, NamedTensors.from_dict(named_inputs)))


def remap_grad_info(grad_info: GradGraphInfo, forward_graph: pir.Graph, backward_graph: pir.Graph) -> GradGraphInfo:
    """Remaps GradGraphInfo to expected connections from a different forward graph.
        The input/output index of the original connection will be used for the connections from the new graph.
    """
    ir = forward_graph.ir._pb_ir
    old_fwd = grad_info.forward_graph
    old_bwd = grad_info.graph
    old_inputs = old_fwd._pb_graph.getInputIds()

    expected_inputs = []
    for ec in grad_info.expected_inputs:
        if ec.fwd_tensor.id in old_inputs:
            idx = old_fwd._pb_graph.getInputIndex(ec.fwd_tensor.id)
            new_id = forward_graph._pb_graph.getInputId(idx)
        else:
            idx = old_fwd._pb_graph.getOutputIndex(ec.fwd_tensor.id)
            new_id = forward_graph._pb_graph.getOutputId(idx)
        expected_inputs.append(_ir.ExpectedConnection(new_id, ec._pb_ec.type))
    expected_outputs = []
    for ec in grad_info.expected_outputs:
        idx = old_fwd._pb_graph.getInputIndex(ec.fwd_tensor.id)
        new_id = forward_graph._pb_graph.getInputId(idx)
        expected_outputs.append(_ir.ExpectedConnection(new_id, ec._pb_ec.type))

    new_info = _ir.BwdGraphInfo(backward_graph._pb_graph.id, expected_inputs, expected_outputs)

    return GradGraphInfo._from_pb(ir, forward_graph._pb_graph, new_info)
