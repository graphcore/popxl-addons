# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Callable, List, Mapping, Tuple, Dict, Type, TypeVar, Optional
import popxl
from popxl import ReplicaGrouping, ops
from popxl.context import debug_context_frame_offset
from popxl.tensor import Variable
from popxl_addons.dot_tree import DotTree
if TYPE_CHECKING:
    from popxl_addons import GraphWithNamedArgs

CLS = TypeVar("CLS", bound='NamedReplicaGrouping')


def fill_none_group(rg: ReplicaGrouping, none_value: ReplicaGrouping):
    if rg is None:
        return none_value
    else:
        return rg


class NamedReplicaGrouping(DotTree[popxl.ReplicaGrouping]):
    """A `DotTree` collection of ReplicaGrouping"""

    @classmethod
    def build_groups(cls: Type[CLS], names: List[str], value: Optional[ReplicaGrouping] = None):
        value = value or popxl.gcg().ir.replica_grouping()
        groups = [value for _ in range(len(names))]
        return cls.pack(names, groups)


def get_instance_replica_grouping(replica_grouping: popxl.ReplicaGrouping) -> popxl.ReplicaGrouping:
    ir = popxl.gcg().ir

    if replica_grouping.stride > ir.instance_replication_factor:
        raise ValueError("replica grouping stride must be < ir.instance_replication_factor")

    if is_cross_instance(replica_grouping):
        return ir.replica_grouping(stride=replica_grouping.stride,
                                   group_size=ir.instance_replication_factor // replica_grouping.stride)
    else:
        return replica_grouping


def is_cross_instance(replica_grouping: popxl.ReplicaGrouping):
    ir = popxl.gcg().ir
    return replica_grouping.group_size * replica_grouping.stride > ir.instance_replication_factor
