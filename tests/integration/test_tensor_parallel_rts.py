# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from popxl_addons.ops.replicated_strided_collectives.replicated_reduce_scatter_strided import replicated_reduce_scatter_strided
import pytest
import numpy as np
import popxl
from popxl import ops
from popxl_addons.ops.replicated_strided_collectives import replicated_all_gather_strided


def shard(x: np.ndarray, n_shards: int, axis: int) -> np.ndarray:
    """Shard array along a given axis"""
    if axis < 0:
        axis = len(x.shape) + axis

    return np.ascontiguousarray(np.concatenate(np.split(x[np.newaxis, ...], n_shards, axis=axis + 1)))


def unshard(x: np.ndarray, axis: int) -> np.ndarray:
    x = np.concatenate(np.split(x, x.shape[0], axis=0), axis=axis + 1)
    return np.ascontiguousarray(x.reshape(x.shape[1:]))


def repeat(x: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """Repeat array along new axis inserted at position `axis`"""
    return np.repeat(np.expand_dims(x, axis), n, axis=axis)


def test_tensor_parallel_rts():
    ir = popxl.Ir()
    ir.replication_factor = 8
    tensor_parallel_grouping = ir.replica_grouping(group_size=2)
    data_parallel_grouping = ir.replica_grouping(stride=2)
    rts_grouping = ir.replica_grouping(stride=2, group_size=2)
    np.random.seed(1984)
    input_x = np.random.normal(0, 1, (4, 4)).astype(np.float32)
    input_ff1 = np.random.normal(0, 1, (4, 4)).astype(np.float32)
    input_ff2 = np.random.normal(0, 1, (4, 4)).astype(np.float32)

    with ir.main_graph, popxl.in_sequence():
        h2d = popxl.h2d_stream((4, 4), popxl.float32)
        x = ops.host_load(h2d)

        ff1_v, ff1_shard = popxl.replica_sharded_variable(shard(input_ff1, tensor_parallel_grouping.group_size, axis=1),
                                                          popxl.float32,
                                                          "ff1",
                                                          replica_grouping=data_parallel_grouping,
                                                          shard_over=rts_grouping.group_size)
        ff2_v, ff2_shard = popxl.replica_sharded_variable(shard(input_ff2, tensor_parallel_grouping.group_size, axis=0),
                                                          popxl.float32,
                                                          "ff2",
                                                          replica_grouping=data_parallel_grouping,
                                                          shard_over=rts_grouping.group_size)

        ff1 = replicated_all_gather_strided(ff1_shard, group=rts_grouping)

        y = x @ ff1

        ff2 = replicated_all_gather_strided(ff2_shard, group=rts_grouping)

        z = y @ ff2

        z_reduced = ops.collectives.replicated_all_reduce(z, group=tensor_parallel_grouping)
        d2h = popxl.d2h_stream(z_reduced.shape, z_reduced.dtype)
        ops.host_store(d2h, z_reduced)

    with popxl.Session(ir, "ipu_hw") as session:
        input_full = repeat(input_x, ir.replication_factor, axis=0)
        out = session.run({h2d: input_full})[d2h]

        np.testing.assert_almost_equal(out[0], (input_x @ input_ff1) @ input_ff2, 5)
        for t in out[1:]:
            np.testing.assert_almost_equal(out[0], t)

    to_check = [ff1_v, ff2_v]
    vs = session.get_tensors_data(to_check)
    for axis, t, np_t in zip([1, 0], to_check, [input_ff1, input_ff2]):
        np.testing.assert_almost_equal(unshard(vs[t], axis), np_t)


if __name__ == "__main__":
    test_tensor_parallel_rts()
