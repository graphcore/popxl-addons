# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, Callable, List
from typing_extensions import Literal

import numpy as np
from more_itertools import flatten, sliced


def shard(x: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Shard array along a given axis. Outputs one array concatenated on a newly created first dimension.

    Example:
        .. code-block:: python
        data = np.arange(4*2).reshape(4, 2)
        # Shape: (4, 2)
        # [[0, 1],
        #  [2, 3],
        #  [4, 5],
        #  [6, 7]]

        # Shard axis 0 into 2
        shard(data, 2, 0)
        # Shape (2, 2, 2)
        # [[[0, 1],
        #   [2, 3]],
        #  [[4, 5],
        #   [6, 7]]]

        # Shard axis 1 into 2
        shard(data, 2, 1)
        # Shape (2, 4, 1)
        # [[[0],
        #   [2],
        #   [4],
        #   [6]],
        #  [[1],
        #   [3],
        #   [5],
        #   [7]]]
    """
    if axis < 0:
        axis = len(x.shape) + axis

    return np.concatenate(np.split(x[np.newaxis, ...], n, axis=axis + 1))


def unshard_arrays(xs: List[np.ndarray], axis: int) -> np.ndarray:
    """Concat a list of arrays along a given axis+2 and squeeze the first axis."""
    x = np.concatenate(xs, axis=axis + 1)
    return x.squeeze(0)


def unshard(x: np.ndarray, axis: int) -> np.ndarray:
    """Opposite to `shard`. Split along first dimension and concat on given axis.

    Example:
        .. code-block:: python
        data = np.random.random((10, 3, 4))  # Shape: (10, 3, 4)

        # creates 2 shards from axis 2
        s2 = shard(data, 2, 2)               # Shape: (2, 10, 3, 2)

        # Undo sharding, first splitting along first dimension to get a list of two
        # shards of shape (1, 10, 3, 2), then concatenating along the specified axis.
        unsharded = unshard(s2, 2)          # Shape: (10, 3, 4)
    """
    return unshard_arrays(np.split(x, x.shape[0], axis=0), axis)


def repeat(x: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """Repeat array along new axis inserted at position `axis`.

    Example:
        .. code-block:: python
        data = np.arange(2*3).reshape(2, 3)
        # Shape: (2, 3)
        # [[0, 1, 2],
        #  [3, 4, 5]]

        # Repeat twice along axis 0
        repeat(data, 2, 0)
        # Shape (2, 2, 3)
        # [[[0, 1, 2],
        #   [3, 4, 5]],
        #  [[0, 1, 2],
        #   [3, 4, 5]]]

        # Repeat twice along axis 1
        repeat(data, 2, 1)
        # Shape (2, 2, 3)
        # [[[0, 1, 2],
        #   [0, 1, 2]],
        #  [[3, 4, 5],
        #   [3, 4, 5]]]
    """
    return np.repeat(np.expand_dims(x, axis), n, axis=axis)


def repeat_axis(x: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """Repeat array along a given axis.

    Example:
        .. code-block:: python
        data = np.arange(2*3).reshape(2, 3)
        # Shape: (2, 3)
        # [[0, 1, 2],
        #  [3, 4, 5]]

        # Repeat twice along axis 0
        repeat_axis(data, 2, 0)
        # Shape (4, 3)
        # [[0, 1, 2],
        #  [3, 4, 5],
        #  [0, 1, 2],
        #  [3, 4, 5]]

        # Repeat twice along axis 1
        repeat_axis(data, 2, 1)
        # Shape (2, 6)
        # [[0, 1, 2, 0, 1, 2],
        #  [3, 4, 5, 3, 4, 5]]
    """
    reps = [1] * x.ndim
    reps[axis] = n
    return np.tile(x, list(reps))


def split2D(x: np.ndarray, n_1: int, n_2: int, axis_1: int, axis_2: int) -> List[List[np.ndarray]]:
    """Split array in 2 dimensions along `axis_1` and `axis_2`. Output a list of lists of arrays. See shard2D for
    similar examples."""
    arrays = np.split(x, n_1, axis_1)
    arrays2d = [np.split(a, n_2, axis_2) for a in arrays]
    return arrays2d


def shard2D(x: np.ndarray, n_1, n_2, axis_1, axis_2) -> np.ndarray:
    """Shard array along 2 given axes. Outputs one array concatenated on a newly created first dimension.
    Axis 1 is the outermost axis and axis 2 the innermost axis.

    Example:
        .. code-block:: python
        data = np.arange(4*2).reshape(4, 2)
        # Shape: (4, 2)
        # [[0, 1],
        #  [2, 3],
        #  [4, 5],
        #  [6, 7]]

        # Split data into 2*2 shards. Axis 0 is outermost axis and, 1 is innermost
        shard2D(data, 2, 2, 0, 1)
        # Shape (4, 2, 1)
        # [[[0],
        #   [2]],
        #  [[1],
        #   [3]],
        #  [[4],
        #   [6]],
        #  [[5],
        #   [7]]]

        # Split data into 2*2 shards. Axis 1 is outermost axis and, 0 is innermost
        shard2D(data, 2, 2, 1, 0)
        # Shape (4, 2, 1)
        # [[[0],
        #   [2]],
        #  [[4],
        #   [6]],
        #  [[1],
        #   [3]],
        #  [[5],
        #   [7]]]
    """
    arrays = split2D(x, n_1, n_2, axis_1, axis_2)
    arrays = [a[np.newaxis, ...] for a in flatten(arrays)]
    return np.concatenate(arrays)


def unshard2D(x: np.ndarray, n_1: int, n_2: int, axis_1: int, axis_2: int):
    """Opposite to shard2D. Split along first dimension and concat on given axes.

    Example:
        .. code-block:: python
        data = np.random.random((10, 3, 4))   # Shape: (10, 3, 4)

        # creates 2 shards from axis 2
        s2 = shard2d(data, 2, 4, 0, 2)        # Shape: (8, 5, 3, 1)

        # Undo 2D sharding
        unsharded = unshard2D(s2, 2, 4, 0, 2) # Shape: (10, 3, 4)
    """
    assert x.shape[0] == n_1 * n_2

    arrays = np.split(x, x.shape[0], axis=0)

    # Unshard innermost dim
    arrays = [unshard_arrays(list(array_group), axis_2) for array_group in sliced(arrays, n_2)]

    # Unshard outermost dim
    x = np.concatenate(arrays, axis=axis_1)
    return x


def repeat_shard(x: np.ndarray,
                 n_repeats,
                 n_shards,
                 shard_axis,
                 sharded_tensor: Literal['contiguous', 'strided'] = 'contiguous') -> np.ndarray:
    """Shard `x` into `n_shards` along `shard_axis`, repeat the sharded tensor `n_repeats`.

    Example:
        .. code-block:: python
        data = np.arange(4)
        # [0, 1, 2, 3]

        repeat_shard(data, 2, 2, 0, sharded_tensor='contiguous')
        # [[0, 1],
        #  [2, 3],
        #  [0, 1],
        #  [2, 3]]

        shard_repeat(data, 2, 2, 0, sharded_tensor='strided')
        # [[0, 1],
        #  [0, 1],
        #  [2, 3],
        #  [2, 3]]
    """
    y = shard(x, n_shards, shard_axis)
    z = repeat_axis(y, n_repeats, 0)

    if sharded_tensor == 'contiguous':
        return z
    elif sharded_tensor == 'strided':
        return z.reshape(n_shards, n_repeats, *y.shape[1:]).swapaxes(0, 1).reshape(n_shards * n_repeats, *y.shape[1:])
    else:
        raise ValueError(f"sharded_tensor should be either 'contiguous' or 'strided'. Not: {sharded_tensor}")


def tensor_parallel_input(input_data: np.ndarray,
                          tp: int,
                          rf: int,
                          repeat_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None):
    """Repeat the data in `input_data` such that consecutive replicas with groupSize tp get the same data
    (optionally modified by repeat_fn)

    Take data of shape (host_loads, *data_shape)
    Output (host_loads, replication_factor, *data_shape)
    Where host_loads = device_iterations * gradient_accumulation_step

    Examples:
        a = np.arange(4)
        [0, 1, 2, 3]
        shape: (4,)

        tensor_parallel_input(a, 1, 2)  # tp = 1, dp = 2
        [[0, 1],
         [2, 3]]
        shape: (2, 2)

        tensor_parallel_input(a, 2, 2)  # tp = 2, dp = 1
        [[0, 0],
         [1, 1],
         [2, 2],
         [3, 3]]
        shape: (4, 2)

        tensor_parallel_input(a, 2, 4)  # tp = 2, dp = 2
        [[0, 0, 1, 1],
         [2, 2, 3, 3]]
        shape: (2, 4)

    Args:
        input_data (np.ndarray): Data to repeat
        tp (int): Tensor parallel replicas
        rf (int): Total Replicas
        repeat_fn (Optional[Callable[[np.ndarray, int], np.ndarray]], optional):
            Optional function to modify each repeat by. Defaults to None.

    Returns:
        data: Data repeated for DP and TP
    """
    assert tp <= rf
    assert (rf / tp).is_integer()

    data = np.expand_dims(input_data, axis=1)
    repeats: List[np.ndarray] = []
    for i in range(tp):
        repeat = data
        if repeat_fn:
            repeat = repeat_fn(repeat.copy(), i)
        repeats.append(repeat)
    data = np.concatenate(repeats, axis=1)
    return data.reshape(-1, rf, *input_data.shape[1:])


def pad_axis(x: np.ndarray, n: int, axis: int):
    """Zero pad an axis until length n.

    Example:
        .. code-block:: python
        data = np.arange(3*2).reshape(3, 2)
        # [[0, 1],
        #  [2, 3],
        #  [4, 5]]

        pad_axis(data, 5, 1)
        # [[0, 1, 0, 0, 0],
        #  [2, 3, 0, 0, 0],
        #  [4, 5, 0, 0, 0]]
    """
    assert x.shape[axis] <= n

    padding = [[0, 0]] * len(x.shape)
    padding[axis] = [0, n - x.shape[axis]]
    return np.pad(x, padding)
