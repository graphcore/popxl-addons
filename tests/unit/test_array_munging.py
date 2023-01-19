# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, Callable, List, Sequence, Union
from typing_extensions import Literal

import numpy as np
import pytest

from popxl_addons.array_munging import (
    shard,
    unshard_arrays,
    unshard,
    repeat,
    repeat_axis,
    split2D,
    shard2D,
    unshard2D,
    repeat_shard,
    tensor_parallel_input,
    pad_axis,
    squeeze_safe,
    handle_negative_axis,
)


@pytest.mark.parametrize(
    "n,axis,target",
    [
        (2, 0, [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),  # Shard axis 0 into 2
        (2, -2, [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),  # Same with negative axis
        (2, 1, [[[0], [2], [4], [6]], [[1], [3], [5], [7]]]),  # Shard axis 1 into 2
    ],
)
def test_shard(n, axis, target):
    data = np.arange(4 * 2).reshape(4, 2)
    output = shard(data, n, axis)
    target = np.array(target)
    np.testing.assert_equal(output, target)


@pytest.mark.parametrize("axis", (2, -1))
def test_unshard(axis):
    data = np.random.random((10, 3, 4))
    s2 = shard(data, 2, 2)
    output = unshard(s2, axis)
    np.testing.assert_equal(output, data)


@pytest.mark.parametrize(
    "n,axis,target",
    [
        (2, 0, [[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]]),  # Repeat twice along axis 0
        (2, -3, [[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]]),  # Same with negative axis
        (2, 1, [[[0, 1, 2], [0, 1, 2]], [[3, 4, 5], [3, 4, 5]]]),  # Repeat twice along axis 1
    ],
)
def test_repeat(n, axis, target):
    data = np.arange(2 * 3).reshape(2, 3)
    output = repeat(data, n, axis)
    target = np.array(target)
    np.testing.assert_equal(output, target)


@pytest.mark.parametrize(
    "n,axis,target",
    [
        (2, 0, [[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]]),  # Repeat twice along axis 0
        (2, -2, [[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]]),  # Same with negative axis
        (2, 1, [[0, 1, 2, 0, 1, 2], [3, 4, 5, 3, 4, 5]]),  # Repeat twice along axis 1
    ],
)
def test_repeat_axis(n, axis, target):
    data = np.arange(2 * 3).reshape(2, 3)
    output = repeat_axis(data, n, axis)
    target = np.array(target)
    np.testing.assert_equal(output, target)


@pytest.mark.parametrize(
    "n_1, n_2, axis_1, axis_2, target",
    [
        (
            2,  # Split data into 2*2 shards. Axis 0 is outermost axis and, 1 is innermost
            2,
            0,
            1,
            [[[0], [2]], [[1], [3]], [[4], [6]], [[5], [7]]],
        ),
        (2, 2, -2, -1, [[[0], [2]], [[1], [3]], [[4], [6]], [[5], [7]]]),  # Same with negative axis
        (
            2,  # Split data into 2*2 shards. Axis 1 is outermost axis and, 0 is innermost
            2,
            1,
            0,
            [[[0], [2]], [[4], [6]], [[1], [3]], [[5], [7]]],
        ),
    ],
)
def test_shard2D(n_1, n_2, axis_1, axis_2, target):
    data = np.arange(4 * 2).reshape(4, 2)
    output = shard2D(data, n_1, n_2, axis_1, axis_2)
    target = np.array(target)
    np.testing.assert_equal(output, target)


@pytest.mark.parametrize("axis_1, axis_2", [(0, 2), (-3, -1)])
def test_unshard2D(axis_1, axis_2):
    data = np.random.random((10, 3, 4))
    s2 = shard2D(data, 2, 4, 0, 2)
    output = unshard2D(s2, 2, 4, axis_1, axis_2)
    np.testing.assert_equal(output, data)


@pytest.mark.parametrize(
    "n_repeats, n_shards, shard_axis, sharded_tensor, target",
    [(2, 2, 0, "contiguous", [[0, 1], [2, 3], [0, 1], [2, 3]]), (2, 2, 0, "strided", [[0, 1], [0, 1], [2, 3], [2, 3]])],
)
def test_repeat_shard(n_repeats, n_shards, shard_axis, sharded_tensor, target):
    data = np.arange(4)
    output = repeat_shard(data, n_repeats, n_shards, shard_axis, sharded_tensor)
    target = np.array(target)
    np.testing.assert_equal(output, target)


@pytest.mark.parametrize(
    "tp, rf, target",
    [(1, 2, [[0, 1], [2, 3]]), (2, 2, [[0, 0], [1, 1], [2, 2], [3, 3]]), (2, 4, [[0, 0, 1, 1], [2, 2, 3, 3]])],
)
def test_tensor_parallel_input(tp, rf, target):
    data = np.arange(4)
    output = tensor_parallel_input(data, tp, rf)
    target = np.array(target)
    np.testing.assert_equal(output, target)


def test_pad_axis():
    data = np.arange(3 * 2).reshape(3, 2)
    output = pad_axis(data, 5, 1)
    target = np.array([[0, 1, 0, 0, 0], [2, 3, 0, 0, 0], [4, 5, 0, 0, 0]])
    np.testing.assert_equal(output, target)


@pytest.mark.parametrize(
    "in_shape, out_shape, axis",
    [
        [(4, 1, 3), (4, 3), None],
        [(4, 3), (4, 3), None],
        [(4, 1, 3), (4, 3), 1],
        [(4, 1, 3), (4, 3), -2],
        [(4, 1, 3), (4, 1, 3), 0],
        [(4, 1, 3), (4, 1, 3), -3],
        [(4, 1, 3, 1), (4, 3), [1, 3]],
        [(4, 1, 3, 1), (4, 3, 1), [1, 0]],
        [(4, 1, 3, 1), (4, 1, 3), [-1, 0]],
    ],
)
def test_squeeze_safe(in_shape, out_shape, axis):
    data = np.arange(np.prod(in_shape)).reshape(in_shape)
    output = squeeze_safe(data, axis)
    assert output.shape == out_shape


def test_handle_negative_axis():
    # Rank 4
    data = np.arange(1).reshape([1] * 4)

    out = handle_negative_axis(data, 3)
    assert out == 3

    out = handle_negative_axis(data, -1)
    assert out == 3

    with pytest.raises(IndexError):
        handle_negative_axis(data, 4)

    with pytest.raises(IndexError):
        handle_negative_axis(data, -5)

    # Rank 0
    data = np.array(0)

    with pytest.raises(IndexError):
        handle_negative_axis(data, 0)
