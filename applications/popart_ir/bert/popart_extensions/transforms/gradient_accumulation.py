# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Dict, Optional
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.transforms.autodiff import GradGraphInfo


__all__ = ["accumulate_gradients_in_graph"]


def accumulate_gradients_in_graph(grad_info: GradGraphInfo,
                                  tensors_to_accumulate_grads: Iterable[pir.Tensor],
                                  accum_type: Optional[pir.dtype] = None) -> Dict[pir.Tensor, pir.Tensor]:
    """Replace the outputs in grad graph that represent a gradient of a tensor in 'tensors_to_accumulate_grads'
        Adds a new input to the grad graph and an accumulate op in the grad graph.

       Returns a mapping from tensors in 'tensors_to_accumulate_grads' to the new subgraph_input"""
    graph = grad_info.graph
    ir = graph.ir()

    expected_outputs = grad_info.get_output_tensors()

    variables: Dict[pir.Tensor, pir.Tensor] = {}

    for tensor in tensors_to_accumulate_grads:
        idx = expected_outputs.index(tensor)
        subgraph_tensor = pir.Tensor._from_pb_tensor(graph._pb_graph.getOutputTensor(idx))
        graph._pb_graph.removeOutput(idx)

        accum_type = tensor.dtype if accum_type is None else accum_type

        with graph:
            accum = pir.subgraph_input(tensor.shape, accum_type, "Accum__"+tensor.name)
            ops.accumulate(accum, subgraph_tensor)

        variables[tensor] = accum

    return variables
