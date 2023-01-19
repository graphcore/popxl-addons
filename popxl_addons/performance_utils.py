# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import re
from math import log10, floor
from collections import defaultdict
from typing import Dict, Tuple
from typing_extensions import Literal
import numpy as np

import popxl
from popxl.utils import table_to_string

__all__ = ["total_FLOPs", "evaluate_ir_FLOPs", "evaluate_FLOPs", "print_FLOPs", "format_flops_scientific"]


def total_FLOPs(ir: popxl.Ir) -> int:
    """
    Give a estimation of the total number of Floating Point Operations in a PopXL IR

    Supported operations: MatMul, GroupNormalisation, Add, Subtract, Div and Mul

    Args:
        ir (popxl.Ir): _description_

    Returns:
        int: _description_
    """
    return evaluate_ir_FLOPs(ir)["TOT"]


def evaluate_ir_FLOPs(ir: popxl.Graph) -> Dict[str, int]:
    """
    Gives a estimation of the total number of Floating Point Operations for a PopXL IR.

    Supported operations: MatMul, GroupNormalisation, Add, Subtract, Div and Mul

    Args:
        graph (popxl.Graph): the graph for which you want to estimate FLOPs.
    Returns:
        Dict[str,int]: A dictionary representing the FLOPs breakdown of the IR.
                       You can then print the breakdown in different ways using print_FLOPs(breakdown).
                       Key "TOT" contains the total FLOPs of the IR
    """
    return _multiply_dict(evaluate_FLOPs(ir.main_graph), ir.replication_factor)


def evaluate_FLOPs(graph: popxl.Graph) -> Dict[str, int]:
    """
    Gives a estimation of the total number of Floating Point Operations for a given graph.

    Supported operations: MatMul, GroupNormalisation, Add, Subtract, Div and Mul

    Args:
        graph (popxl.Graph): the graph for which you want to estimate FLOPs.
    Returns:
        Dict[str,int]: A dictionary representing the FLOPs breakdown of the graph.
                       You can then print the breakdown in different ways using print_FLOPs(breakdown).
                       Key "TOT" contains the total FLOPs of the Graph
    """
    visited_graphs = {}
    flops_breakdown = _evaluate_FLOPs(graph._pb_graph, visited_graphs)
    return flops_breakdown


def print_FLOPs(
    flops_breakdown: Dict[str, int],
    mode: Literal["type", "flat"] = "type",
    numerical_format: Literal["K", "M", "G", "T", "nearest3"] = "nearest3",
):
    """
    Print the FLOPs breakdown of a graph.
    If `mode` is `type`, the breakdown is printed per operation type, summing all FLOPs of ops of the same type.
        Op Type            | FLOPs
        -------------------------------
        TOT                | 9.072E+03
        MatMul             | 9.072E+03
        Add                | 0.000E+00
        Sub                | 0.000E+00
        GroupNormalization | 0.000E+00

    If `mode` is `flat`, the breakdown keys are flattened and you get a flat view of each op FLOPs.
        Op Type | FLOPs      | Op Name
        ----------------------------------
        TOT     | 9.072E+03  |
        MatMul  | 8.100E+03  | MatMul.104
        MatMul  | 0.900E+03  | MatMul.103
        MatMul  | 64.000E+00 | MatMul.100
        MatMul  | 8.000E+00  | MatMul.102

    Breakdown is sorted in descending order.
    """

    def _flatten(dict, new_dict, key=""):
        for k, v in dict.items():
            if isinstance(v, Dict):
                new_key = key
                if key != "":
                    new_key += "."
                _flatten(v, new_dict, new_key + str(k))
            else:
                new_key = key
                if key != "":
                    new_key += "."
                new_dict.update({new_key + str(k): v})

    rows = []
    if mode == "type":
        rows = [["Op Type", "FLOPs", "%"]]
        grouped = defaultdict(lambda: 0)
        for k, v in flops_breakdown.items():
            grouped[k] += _sum_dict(v)
        for k, v in sorted(grouped.items(), key=lambda item: item[1], reverse=True):
            pcent = 100 * v / grouped["TOT"]
            rows.append([k, format_flops_scientific(v, numerical_format), f"{pcent:.1f}"])

    elif mode == "flat":
        rows = [["Op Type", "FLOPs", "Op Name"]]
        flattened_breakdown = {}
        _flatten(flops_breakdown, flattened_breakdown)
        for k, v in sorted(flattened_breakdown.items(), key=lambda item: item[1], reverse=True):
            tokens = k.split(".")
            op_type = k
            if len(tokens) > 0:
                op_type = tokens[0]
            rows.append([op_type, format_flops_scientific(v, numerical_format), k])
        rows.append(["TOT", format_flops_scientific(flattened_breakdown["TOT"], numerical_format), ""])

    else:
        raise ValueError("Unknown mode specified. Please use `type` or `flat`.")

    print(table_to_string(rows))


def matmul_flops(op) -> int:
    # reduces to
    #   2m for dot product of vectors (m,)
    #   2mn for (n, m) (m,)
    #   2mnp for (n, m) (m, p)
    #   2bmnp for (b,n, m) (b, m, p)
    a_info = op.getInputTensors()[0].info
    b_info = op.getInputTensors()[1].info
    c_info = op.getOutputTensors()[0].info

    comm_dim = a_info.shape()[-1]
    assert comm_dim == b_info.shape()[-2] if len(b_info.shape()) > 1 else b_info.shape()[-1]
    other_dims = np.prod(c_info.shape())
    return 2 * comm_dim * other_dims


def elementwise_binary_flops(op) -> int:
    info = op.getOutputTensors()[0].info
    return np.prod(info.shape())


def group_norm_flops(op) -> int:
    # ----- sketch of computation -----
    # x = (x - mu)/sigma
    # x = scale * x + bias
    x_info = op.getInputTensors()[0].info
    out_info = op.getOutputTensors()[0].info
    mean_info = op.getOutputTensors()[1].info

    groups = int(mean_info.shape()[0] / x_info.shape()[0])
    averaged_elements = int(x_info.shape()[-1] * x_info.shape()[-2] / groups)
    mean_calc_flops = averaged_elements  # sum all elements + a single division (ignored)
    # var(x) = <x>^2 - <x^2>.
    # For <x^2>: multiply each element and sum -> 2*averaged_elements
    # For <x>^2 - <x^2>: multiply each element in <x> to get <x>^2, and subtract -> 2 * mean_info.shape()[0]
    std_calc_flops = 2 * averaged_elements + 2 * mean_info.shape()[0]
    out_elements = np.prod(out_info.shape())
    rescale_flops = 2 * out_elements  # subtraction and mul in  x = (x - mu)/sigma
    # extra scale and bias
    scale_flops = out_elements
    bias_flops = out_elements
    flops = mean_calc_flops + std_calc_flops + scale_flops + bias_flops + rescale_flops
    return flops


FLOPS_FNS = {
    "Add": elementwise_binary_flops,
    "Sub": elementwise_binary_flops,
    "Div": elementwise_binary_flops,
    "Mul": elementwise_binary_flops,
    "MatMul": matmul_flops,
    "GroupNormalization": group_norm_flops,
}


def _multiply_dict(v, factor):
    if isinstance(v, Dict):
        return {k: _multiply_dict(val, factor) for k, val in v.items()}
    return v * factor


def _sum_dict(v):
    if isinstance(v, Dict):
        return sum(map(_sum_dict, v.values()))
    return v


def _evaluate_FLOPs(pb_graph, visited_graphs: Dict[popxl.Graph, Tuple[int, Dict[str, int]]]):
    if pb_graph in visited_graphs:
        return visited_graphs[pb_graph]

    total_flops = 0
    flops_breakdown = defaultdict(dict)

    for op in pb_graph.getOps():
        if hasattr(op, "getCalledGraph"):
            called = op.getCalledGraph()
            called_flops_breakdown = _evaluate_FLOPs(called, visited_graphs)

            if hasattr(op, "getTripCountValue"):
                called_flops_breakdown = _multiply_dict(called_flops_breakdown, op.getTripCountValue())

            total_flops += called_flops_breakdown["TOT"]

            graph_name = re.sub(r"_subgraph\(\d+\)", "", str(called.id))
            key = op.opType() + "_" + str(op.id) + "(" + graph_name + ")"
            called_flops_breakdown = {k: {key: v} for k, v in called_flops_breakdown.items()}
            for k, v in called_flops_breakdown.items():
                flops_breakdown[k].update(v)
        else:
            op_type = str(op.opType())
            if op_type in FLOPS_FNS.keys():
                flops = FLOPS_FNS[op_type](op)
                total_flops += flops
                flops_breakdown[op_type].update({op.id: flops})

    flops_breakdown.update({"TOT": total_flops})
    visited_graphs[pb_graph] = flops_breakdown
    return flops_breakdown


def format_flops_scientific(val, format: Literal["K", "M", "G", "T", "nearest3"] = "nearest3") -> str:
    if format not in ["K", "M", "G", "T", "nearest3"]:
        raise ValueError("unknown format specified")

    decimals = 3
    exponent_template = "{:0>%d}" % 2  # 03, 09
    mantissa_template = "{:.%df}" % decimals

    if format == "nearest3":
        val_power = floor(log10(abs(val))) if val > 0.0 else 0
        lowest_three_power = val_power // 3
        # the remainder of the division by three can be 1 or 2. If it's 2, we are closer to the next power.
        extra_power = int(val_power % 3 == 2)
        nearest_third = 3 * (lowest_three_power + extra_power)
        adjusted_mantissa = val * 10 ** (-nearest_third)
        adjusted_mantissa_string = mantissa_template.format(adjusted_mantissa)
        adjusted_exponent_string = "+-"[nearest_third < 0] + exponent_template.format(abs(nearest_third))
        return adjusted_mantissa_string + "E" + adjusted_exponent_string
    else:
        label_to_exp = {"K": 3, "M": 6, "G": 9, "T": 12}
        adjusted_mantissa = val * 10 ** (-label_to_exp[format])
        adjusted_mantissa_string = mantissa_template.format(adjusted_mantissa)
        adjusted_exponent_string = "+" + exponent_template.format(abs(label_to_exp[format]))
        return adjusted_mantissa_string + "E" + adjusted_exponent_string
