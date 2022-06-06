# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
from math import ceil

import numpy as np
from scipy.stats import truncnorm
import popxl
from popxl import ops, ReplicaGrouping
from typing import Optional

import popxl_addons as addons


class Embedding(addons.Module):
    def __init__(self,
                 dtype: popxl.dtype,
                 vocab_size: int,
                 hidden_size: int,
                 axis: int = 0,
                 replica_grouping: Optional[ReplicaGrouping] = None):
        """
        Args:
            dtype: numerical type
            vocab_size: dimension of the input space. Input indices take value in this space, ranging from 0 ... vocab_size
            hidden_size: dimension of the output space, (dimension of embedded vectors). Each input index corresponds to a distinct vector of size hidden_size.
            axis: vocab axis. Each index selects elements in the embedding matrix along this axis. Deafult to 0: the vocab axis is along axis 0, rows.
            replica_grouping: if specified and the number of groups > 1 the embedding will be sharded across the vocab axis
        """
        super().__init__()
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.axis = axis
        self.replica_grouping = replica_grouping
        if replica_grouping:
            self.n_shards = self.replica_grouping.num_groups
        else:
            self.n_shards = 1

        self.vocab_shard_size = self.get_vocab_shard_size(self.vocab_size, self.n_shards)
        self.offsets = self.get_offsets(self.vocab_size, self.n_shards)

    @staticmethod
    def get_vocab_shard_size(vocab_size: int, n_shards: int) -> int:
        """If using sharding, vocab size per shard. If no sharding it is equivalent to `vocab_size`"""
        return ceil(vocab_size / n_shards)

    @staticmethod
    def get_offsets(vocab_size: int, n_shards: int) -> np.ndarray:
        """If using sharding, indices offset per shard."""
        vocab_shard_size = Embedding.get_vocab_shard_size(vocab_size, n_shards)
        return np.arange(max(vocab_size, n_shards), step=vocab_shard_size)

    def build(self, indices: popxl.Tensor) -> popxl.Tensor:
        """

        Args:
            indices: token indices. Shape (...,sequence_length, vocab_size)
            If using sharding, the indices need to be pre-offsetted. You can obtain the offsets
            from the `get_offsets` method.

        Returns:
            Embedded vectors for each index. Shape (...,sequence_length, hidden_size)
            Embedding corresponds to a table lookup: each index in `indices` selects a row in the embedding weights matrix.
            If using sharding, out of range indices will be automatically set to zero.
        """
        self.weight = self.add_variable_input("weight",
                                              partial(truncnorm.rvs,
                                                      -2,
                                                      2,
                                                      loc=0,
                                                      scale=0.02,
                                                      size=(self.vocab_shard_size, self.hidden_size)),
                                              self.dtype,
                                              replica_grouping=self.replica_grouping)
        return ops.gather(self.weight, indices, axis=self.axis, zero_OOR=self.n_shards > 1)
