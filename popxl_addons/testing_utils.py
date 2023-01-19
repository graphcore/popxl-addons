# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Iterable, Type

from popart._internal import ir as _ir
import popxl
from popxl_addons.module import Module
from popxl.tensor import Variable

from popxl_addons import NamedTensors
from popxl_addons.ops import host_store
import numpy as np
from typing import Union, Callable, Dict, List, Optional


def ops_of_type(ops: Iterable[_ir.Op], op_type: Type[_ir.Op]) -> int:
    return len(list(filter(lambda op: isinstance(op, op_type), ops)))


class TensorInput:
    """
    Wrapper for a numpy array that is meant to be a tensor input of a graph.
    If a dytpe is specified, the popxl tensor will be created with such dtype,
    otherwise the dtype will be inferred from the numpy array.
    Please note that poplar does not support DOUBLE dtype, which is the default numpy
    dtype.
    """

    def __init__(self, data: np.ndarray, dtype: Optional[popxl.dtype] = None) -> None:
        self.data = data
        self.dtype = dtype


def run_module(
    module: Module, *args, weights: Optional[Callable[[NamedTensors], Dict[Variable, np.ndarray]]] = None, **kwargs
) -> List[np.ndarray]:
    """
    Run a module with provided args and return graph output data (numpy arrays)

    Args:
        - module (Module): the `addons.Module` instance that defines the graph
        - *args: arguments needed to build and call the module. For tensors you can use numpy data wrapped in a TensorInput class.
                This way they will automatically be converted into popxl variables before feeding them to the model.
        - weights (optional, Callable[[NamedTensors], Dict[Variable, np.ndarray]]): a callable that takes the graph variables and produces
                                                                                    a dictionary of their values. Must be specified as a kwarg.
        - **kwargs: additional kwargs to pass to the model

    A ir is created, inputs marked as TensorInput are turned into popxl variables and the args provided are used to
    create graph and factories from the module.
    If weights are provided for the variables, they are copied to IPU before the session is run.
    The session is run and graph outputs are returned as numpy arrays.

    Usage:
        input_np = np.random.rand(4)
        input_torch = torch.Tensor(input_np)

        torch_linear = torch.nn.Linear(4,2)
        output_torch = torch_linear(input_torch)
        output_torch = output_torch.detach().numpy()

        input_t = TensorInput(input_np, popxl.float32)
        popxl_linear = Linear(2)

        def weight_mapping(variables: NamedTensors):
            return {
                variables.weight : to_numpy(torch_linear.weight.data).T,
                variables.bias: to_numpy(torch_linear.bias)
            }

        output_popxl, = run_module(popxl_linear, input_t, weights=weight_mapping)

        np.testing.assert_allclose(output_popxl, output_torch.reshape(output_popxl.shape),rtol=10e-3)

    Return:
        List[np.ndarray] : outputs of the module
    """

    def typed_arg(a):
        if isinstance(a, TensorInput):
            dtype = a.dtype
            if dtype is None:
                # try infer from numpy
                dtype = popxl.dtype.as_dtype(a.data)
            return popxl.variable(a.data, dtype=dtype)
        else:
            return a

    ir = popxl.Ir()
    with ir.main_graph:
        new_args = [typed_arg(a) for a in args]
        new_kwargs = {k: typed_arg(v) for k, v in kwargs.items()}
        facts, graph = module.create_graph(*new_args, **new_kwargs)
        vars = facts.init()
        tensor_args = [a for a in new_args if isinstance(a, popxl.Tensor)]
        tensor_kwargs = {k: a for k, a in new_kwargs.items() if isinstance(a, popxl.Tensor)}
        outputs = graph.bind(vars).call(*tensor_args, **tensor_kwargs)
        outs_d2h = []
        for o in outputs:
            outs_d2h.append(host_store(o))

    with popxl.Session(ir=ir, device_desc="ipu_hw") as session:
        if weights:
            if isinstance(weights, Callable):
                weights = weights(vars)
            session.write_variables_data(weights)
        outs = session.run()
    np_outs = [outs[d2h] for d2h in outs_d2h]
    return np_outs
