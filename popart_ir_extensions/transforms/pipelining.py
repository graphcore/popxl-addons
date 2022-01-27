# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.context import get_current_context
from popart.ir.ops.call import SubgraphOpInfo
from popart.ir.transforms.autodiff import GradGraphInfo

from popart_ir_extensions.graph import BoundGraph
from popart_ir_extensions.graph_cache import GraphCache
from popart_ir_extensions.module import Module
from popart_ir_extensions.route_tensor import route_tensor_into_graph

__all__ = ["pipelined_execution", "stash_and_restore_activations"]

OpId = int


class PipelineBoundGraph(BoundGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_input_to_arg: Dict[str, pir.Tensor] = {}
        self._produces_original_tensor: List[str] = []

    def _moved_args(self, subgraph_in_to_parent_in: Optional[Mapping[pir.Tensor, pir.Tensor]] = None):
        # Move inputs into the current graph.
        moved_inputs = {
            sg_t: route_tensor_into_graph(v,
                                          modified=sg_t._pb_tensor.modifiedRegionsByOps(self.graph._pb_graph.getOps()))
            for sg_t, v in self.args.items()
        }
        if subgraph_in_to_parent_in:
            moved_inputs.update({**subgraph_in_to_parent_in})
        return moved_inputs

    def call(self, *args: pir.Tensor,
             subgraph_in_to_parent_in: Optional[Mapping[pir.Tensor, pir.Tensor]] = None) -> Tuple[pir.Tensor, ...]:
        return super().call(*args, args=self._moved_args(subgraph_in_to_parent_in))

    @property
    def original_output_ids(self):
        return self._produces_original_tensor

    def add_input_tensor(self, tensor: pir.Tensor):
        with self.graph:
            input_tensor = pir.subgraph_input(tensor.shape, tensor.dtype, tensor.name, by_ref=True)
            self._original_input_to_arg[tensor.id] = input_tensor
            self.args[input_tensor] = tensor
        return input_tensor

    def add_output_tensor(self, tensor: pir.Tensor, original_id: str):
        with self.graph:
            pir.subgraph_output(tensor)
        self._produces_original_tensor.append(original_id)

    def reconnect_input(self, original_id: str, new: pir.Tensor):
        if original_id in self._original_input_to_arg.keys():
            sg = self._original_input_to_arg[original_id]
            self.args[sg] = new

    def __contains__(self, value: Any):
        return value in self.graph


class Pipelining:
    def __init__(self, graph: pir.Graph, steps: int):
        self.ir = graph.ir()
        self.graph = self.ir.create_empty_graph("pipeline_graph")
        self.original_graph = graph
        self.steps = steps

        self.original_tid_to_cloned_tensor: Dict[str, pir.Tensor] = {}
        self.original_tid_to_graph: Dict[str, PipelineBoundGraph] = {}
        self.external_inputs: Dict[PipelineBoundGraph, Dict[str, pir.Tensor]] = defaultdict(dict)

        self.stages: Dict[int, PipelineBoundGraph] = {}
        self.loads: Dict[int, PipelineBoundGraph] = {}
        self.stores: Dict[int, PipelineBoundGraph] = {}
        self.ipu_copies = PipelineBoundGraph(self.ir.create_empty_graph("ipu_copies"))
        self.ipu_copy_dummy_inputs: Dict[str, pir.Tensor] = {}

    def apply(self, include_ops: Optional[List[OpId]] = None):
        self.construct_graphs(include_ops)
        self.num_stages = len(self.stages.keys())
        if self.steps < 2 * self.num_stages:
            raise ValueError("Pipelining requires the steps at least 2x the number of stages in the pipeline. "
                             f"steps={self.steps} stages={self.num_stages}")
        self.build()

    def get_op_pipeline_stage(self, op: _ir.Op):
        if not op.hasPipelineStage():
            raise ValueError("All Ops in pipelined execution should have a pipelineStage")
        return op.getPipelineStage()

    def required_for_host_load(self, op: _ir.Op):
        if isinstance(op, _ir.op.exchange.HostLoadOp):
            return True

        if isinstance(op, _ir.op.InitOp):
            output: _ir.Tensor = op.outTensor(0)
            consumers = output.consumers.getOps()
            if len(consumers) == 1 and isinstance(consumers[0], _ir.op.exchange.HostLoadOp):
                return True
        return False

    def construct_graphs(self, include_ops: Optional[List[OpId]]):
        load_ops = defaultdict(list)
        compute_ops = defaultdict(list)
        store_ops = defaultdict(list)
        copy_ops = defaultdict(list)

        for op in self.original_graph._pb_graph.getOpSchedule(False):
            if include_ops and op.id not in include_ops:
                continue

            stage = self.get_op_pipeline_stage(op)

            if self.required_for_host_load(op):
                load_ops[stage].append(op)
            elif isinstance(op, _ir.op.exchange.HostStoreOp):
                store_ops[stage].append(op)
            elif isinstance(op, _ir.op.IpuCopyOp):
                copy_ops[stage].append(op)
            else:
                compute_ops[stage].append(op)

        for stage in sorted(compute_ops.keys()):
            for op in load_ops[stage]:
                if stage not in self.loads:
                    self.loads[stage] = PipelineBoundGraph(self.ir.create_empty_graph(f"loads_stage_{stage}"))
                self.move_op_into_graph(self.loads[stage], op)
            for op in compute_ops[stage]:
                if stage not in self.stages:
                    self.stages[stage] = PipelineBoundGraph(self.ir.create_empty_graph(f"stage_{stage}"))
                self.move_op_into_graph(self.stages[stage], op)
            for op in store_ops[stage]:
                if stage not in self.stores:
                    self.stores[stage] = PipelineBoundGraph(self.ir.create_empty_graph(f"stores_stage_{stage}"))
                self.move_op_into_graph(self.stores[stage], op)
            for op in copy_ops[stage]:
                self.move_op_into_graph(self.ipu_copies, op)

    def move_op_into_graph(self, graph: PipelineBoundGraph, op: _ir.Op):
        cloned_op = op.cloneIntoGraph(graph.graph._pb_graph)
        inputs = op.getInputIndexMap()
        outputs = op.getOutputIndexMap()

        for idx, tensor in inputs.items():
            tensor = pir.Tensor._from_pb_tensor(tensor)

            if tensor.id not in self.original_tid_to_cloned_tensor.keys():
                external_inputs = self.external_inputs[graph]
                if tensor.id in external_inputs.keys():
                    sg_tensor = external_inputs[tensor.id]
                else:
                    sg_tensor = graph.add_input_tensor(tensor)
                    external_inputs[tensor.id] = sg_tensor
            else:
                sg_tensor = self.original_tid_to_cloned_tensor[tensor.id]

            if sg_tensor not in graph:
                # If sg_tensor is not in the current graph, Mark it
                # as an output of it's graph and create a new input for the current graph.
                last_graph = self.original_tid_to_graph[tensor.id]
                if tensor.id not in last_graph.original_output_ids:
                    last_graph.add_output_tensor(sg_tensor, tensor.id)
                sg_tensor = graph.add_input_tensor(tensor)
                self.original_tid_to_cloned_tensor[tensor.id] = sg_tensor
                self.original_tid_to_graph[tensor.id] = graph

            if isinstance(op, _ir.op.IpuCopyOp):
                source_ipu = op.getSourceIpu(tensor.id)
                cloned_op.connectInTensor(idx, sg_tensor.id, source_ipu)
                with self.graph, pir.ipu(source_ipu):
                    dummy_input = ops.init(tensor.shape, tensor.dtype, "dummy__" + tensor.name)
                    graph.reconnect_input(tensor.id, dummy_input)
                    self.ipu_copy_dummy_inputs[tensor.id] = dummy_input
            else:
                cloned_op.connectInTensor(idx, sg_tensor.id)

        for idx, tensor in outputs.items():
            tensor = pir.Tensor._from_pb_tensor(tensor)
            cloned_op.createAndConnectOutTensor(idx, graph.graph._create_tensor_id(tensor.name))
            out_tensor = pir.Tensor._from_pb_tensor(cloned_op.outTensor(idx))
            self.original_tid_to_cloned_tensor[tensor.id] = out_tensor
            self.original_tid_to_graph[tensor.id] = graph

        cloned_op.setup()

        op.disconnectAllInputs()
        op.disconnectAllOutputs()
        self.original_graph._pb_graph.eraseOp(op.id)

    @pir.in_sequence()
    def build(self):
        ops.call(self.graph)
        with self.graph:
            # RampUp
            for i in range(1, self.num_stages):
                self.cycle(0, i)
            self.main_cycle()
            # RampDown
            for i in range(1, self.num_stages):
                self.cycle(i, self.num_stages)

    def cycle(self, start: int, end: int):
        # HostLoad
        self.load(start, end)
        # Compute
        self.compute(start, end)
        # HostStore
        self.store(start, end)
        # Skip the final copy
        if start == self.num_stages - 1:
            return
        # IPUCopy
        self.copy()

    def remap_tensors(self, original_ids: Iterable[str], new_outputs: Iterable[pir.Tensor]):
        """Changes the connected tensors on each PipelineBoundGraph to a new tensor"""
        for original, new in zip(original_ids, new_outputs):
            for stage in range(self.num_stages):
                if stage in self.loads.keys():
                    self.loads[stage].reconnect_input(original, new)
                if stage in self.stages.keys():
                    self.stages[stage].reconnect_input(original, new)
                if stage in self.stores.keys():
                    self.stores[stage].reconnect_input(original, new)
            self.ipu_copies.reconnect_input(original, new)
            self.original_tid_to_cloned_tensor[original] = new

    def move_copied_tensor_out_of_nested_repeat_graph(self, op_in_index: int, original: str, repeat_op: _ir.op.LoopOp):
        op_out_index = op_in_index - 2

        sg_tensor = self.original_tid_to_cloned_tensor[original]
        sg_graph = sg_tensor._pb_tensor.getGraph()
        sg_graph.markAsOutput(sg_tensor.id)

        repeat_tensor = None
        repeat_graph = pir.Graph._from_pb(repeat_op.getCalledGraphs()[0])
        for op in repeat_graph._pb_graph.getOps():
            if isinstance(op, _ir.op.CallOp) and op.getCalledGraphs()[0].id == sg_graph.id:
                repeat_tensor = repeat_graph._create_tensor_id(sg_tensor.name)
                op.createAndConnectOutTensor(sg_graph.getOutputIndex(sg_tensor.id), repeat_tensor)
                op.setup()
                break
        if repeat_tensor is None:
            raise RuntimeError("There should have been a CallOp in the repeat_op's graph")

        repeat_op.addLoopOutput(op_out_index, repeat_op.outId(op_out_index), repeat_tensor, True)
        out_tensor = pir.Tensor._from_pb_tensor(repeat_op.outTensor(op_out_index))

        return out_tensor

    def main_cycle(self):
        tensors_before_repeat = self.original_tid_to_cloned_tensor.copy()

        cycle_graph = pir.gcg().ir().create_empty_graph("main_cycle")
        main_cycles = self.steps - (self.num_stages - 1)
        # LoopOp can have operations that require a graph to execute on.
        with pir.ipu(0):
            ops.repeat(cycle_graph, main_cycles)
        with cycle_graph:
            self.cycle(0, self.num_stages)

        # TODO: This is a terrible way to get the repeat op.
        repeat_op = None
        for op in self.graph._pb_graph.getOps():
            if isinstance(op, _ir.op.LoopOp):
                repeat_op = op
        if repeat_op is None:
            raise RuntimeError("There should have been a LoopOp")

        inputs = repeat_op.getInputIndexMap()
        remap_original = []
        remap_new = []
        for original, tensor in tensors_before_repeat.items():
            # Remap tensors to the input tensor to the loopOp. Inputs will
            # be passed with modified=True so it is correct to use the inputs.
            # Tensors that are output by ipu_copies should instead be remapped
            # to the output tensor. The output in the nested graph should also
            # be connected up through the nested graphs and setup as a loop carried output.
            for idx, in_tensor in inputs.items():
                if in_tensor.id == tensor.id:
                    if original in self.ipu_copies.original_output_ids:
                        remap_tensor = self.move_copied_tensor_out_of_nested_repeat_graph(idx, original, repeat_op)
                    else:
                        remap_tensor = pir.Tensor._from_pb_tensor(in_tensor)

                    remap_original.append(original)
                    remap_new.append(remap_tensor)
                    break

        # Stage0 outputs will not be available after the main cycle. So reconnect dummy inputs
        for original, dummy in self.ipu_copy_dummy_inputs.items():
            self.ipu_copies.reconnect_input(original, dummy)

        self.remap_tensors(remap_original, remap_new)

    def load(self, start: int, end: int):
        original = []
        outputs = []
        for stage in range(start, end):
            if stage in self.loads.keys():
                original += self.loads[stage].original_output_ids
                outputs += self.loads[stage].call()
        self.remap_tensors(original, outputs)

    def compute(self, start: int, end: int):
        original = []
        outputs = []
        for stage in range(start, end):
            original += self.stages[stage].original_output_ids
            outputs += self.stages[stage].call()
        self.remap_tensors(original, outputs)

    def store(self, start: int, end: int):
        original = []
        outputs = []
        for stage in range(start, end):
            if stage in self.stores.keys():
                original += self.stores[stage].original_output_ids
                outputs += self.stores[stage].call()
        self.remap_tensors(original, outputs)

    def copy(self):
        outputs = self.ipu_copies.call()
        self.remap_tensors(self.ipu_copies.original_output_ids, outputs)


@contextmanager
def pipelined_execution(steps: int):
    """Pipeline Transformation Context.
        Ops created in this context will be executed in a pipeline where each stage is executed `steps` times.
        Pipeline stages should be annotated using `pir.pipeline_stage(..)`.
        All operations should have a pipeline stage.
        
        See `docs/pipelining.md` for more details

    Args:
        steps (int): Number of times the scoped computation should execute.
    """
    graph = pir.gcg()
    transform = Pipelining(graph, steps)
    ops: List[int] = []

    def hook(op: _ir.Op):
        ops.append(op.id)

    handle = graph.register_op_created_hook(hook)
    yield
    graph.remove_op_created_hook(handle)

    transform.apply(ops)


class Stash(Module):
    @pir.in_sequence()
    def build(self, t: pir.Tensor, stash_size: int):
        counter = self.add_input_tensor("counter", partial(np.zeros, (1, )), pir.uint32, by_ref=True)

        # Make `t` have the correct 0th dimension
        stashed_shape = t.shape
        if stashed_shape == ():
            stashed_shape = (1, )
        t = t.reshape((1, *stashed_shape))

        if stash_size <= 1:
            raise TypeError(f"Stash must be larger than 1. size={stash_size}, t={t}")

        stash = self.add_input_tensor(f"stash", partial(np.zeros, (stash_size, *stashed_shape)), t.dtype, by_ref=True)
        ops.dynamic_update_(stash, counter, t, axes=[0], sizes=[1], no_overlap=True)
        ops.increment_mod_(counter, 1, stash_size)


class Restore(Module):
    @pir.in_sequence()
    def build(self, stash: pir.TensorByRef):
        counter = self.add_input_tensor("counter", partial(np.zeros, (1, )), pir.uint32, by_ref=True)
        t = ops.dynamic_slice(stash, counter, axes=[0], sizes=[1], no_overlap=True)
        stash_size = stash.shape[0]
        ops.increment_mod_(counter, 1, stash_size)
        return t.reshape(t.shape[1:])


def _is_external_input(t: pir.Tensor):
    if t._pb_tensor.hasProducer():
        prod = t._pb_tensor.getProducer()
        return not prod.hasPipelineStage()
    return True


def stash_and_restore_tensor(t: pir.Tensor, cache: Optional[GraphCache] = None,
                             from_stage: Optional[int] = None) -> pir.Tensor:
    """Create stash and counter variables to tensors t to. Stash size will be calculated from the


    Args:
        t (pir.Tensor): tensor to stash
        from_stage (Optional[int], optional): stage to locate the stashOp. Defaults to None.

    Raises:
        RuntimeError: if `from_stage` is not specified and t does not have a producer.
        RuntimeError: if `stash_and_restore_tensor` is not called in a `pir.pipeline_stage` context.

    Returns:
        pir.Tensor: The restored tensor
    """
    if from_stage is None:
        if not t._pb_tensor.hasProducer():
            raise RuntimeError(
                "If the tensor to be stash does not have a producer `from_stage` must be specified to `stash_and_restore_tensor`"
            )
        from_stage = t._pb_tensor.getProducer().getPipelineStage()

    to_stage = get_current_context().pipeline_stage

    if to_stage is None:
        raise RuntimeError("`stash_and_restore_tensor` must be called in a pir.pipeline_stage context.")

    stash_size = (to_stage - from_stage) + 1

    with pir.pipeline_stage(from_stage):
        args, graph = Stash(cache).create_graph(t, stash_size)
        stash_args = args.init()
        graph.bind(stash_args).call(t)

    with pir.pipeline_stage(to_stage):
        args, graph = Restore(cache).create_graph(stash_args.stash)
        restored, = graph.bind(args.init()).call(stash_args.stash)

    return restored


def stash_and_restore_activations(call_info: SubgraphOpInfo,
                                  grad_info: GradGraphInfo,
                                  cache: Optional[GraphCache] = None) -> Dict[pir.Tensor, pir.Tensor]:
    activations = grad_info.get_inputs_from_forward_call_info(call_info)

    # If the activation is produced on the current pipeline stage then don't create a stash.
    from_stage = call_info._op.getPipelineStage()
    to_stage = get_current_context().pipeline_stage

    if to_stage != from_stage:
        for sg_tensor, act in activations.items():
            if not _is_external_input(act):
                act = stash_and_restore_tensor(act, cache, from_stage=from_stage)
            activations[sg_tensor] = act
    return activations
