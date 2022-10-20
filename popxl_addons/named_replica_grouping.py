# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import json
import logging
import os
from typing import List, Type, TypeVar, Optional
import popxl
from popxl import ReplicaGrouping
from popxl_addons.dot_tree import DotTree

CLS = TypeVar("CLS", bound='NamedReplicaGrouping')


def fill_none_group(rg: ReplicaGrouping, none_value: ReplicaGrouping):
    if rg is None:
        return none_value
    else:
        return rg


class NamedReplicaGrouping(DotTree[popxl.ReplicaGrouping]):
    """A `DotTree` collection of ReplicaGrouping"""

    @classmethod
    def build_groups(cls: Type[CLS], names: List[str], value: ReplicaGrouping):
        groups = [value for _ in range(len(names))]
        return cls.pack(names, groups)


def get_instance_replica_grouping(replica_grouping: Optional[popxl.ReplicaGrouping] = None) -> popxl.ReplicaGrouping:
    """
    Given a replica grouping, returns its equivalent group restricted to a single instance, meaning
    that the group will have the same stride but will include only IPUs in the instance replication factor.
    If replica_grouping is None, it defaults to all replicas inside an instance
    """
    ir = popxl.gcg().ir
    replica_grouping = replica_grouping or popxl.gcg().ir.replica_grouping()
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


_SHOULD_WARN = True


def _warn_assumed_ild_size(size):
    """Only warn once per process to avoid spamming the logs"""
    import popdist
    global _SHOULD_WARN
    if _SHOULD_WARN and popdist.isPopdistEnvSet():
        _SHOULD_WARN = False
        logging.warning(f"Could not read IPU link domain size from enviroment. Assuming {size}.")


def get_ild_size_from_popdist() -> int:
    """
    Fetch the IPU Link Domain size from the popdist enviroment.
    """
    opts = json.loads(os.environ.get("POPLAR_TARGET_OPTIONS", "{}"))
    size = opts.get("ipuLinkDomainSize", None)
    if size is None:
        size = min(popxl.gcg().ir.replication_factor, 64)
        _warn_assumed_ild_size(size)
    else:
        size = int(size)
    return size


def get_ild_replica_grouping(replica_grouping: Optional[popxl.ReplicaGrouping] = None) -> popxl.ReplicaGrouping:
    """
    Given a replica grouping, returns its equivalent group restricted to a single IPU Link Domain (ILD), meaning
    that the group will have the same stride but will include only IPUs in the ILD.
    If replica_grouping is None, it defaults to all replicas inside an ILD.
    """
    ir = popxl.gcg().ir
    replica_grouping = replica_grouping or popxl.gcg().ir.replica_grouping()
    ild_size = get_ild_size_from_popdist()
    if replica_grouping.stride > ild_size:
        raise ValueError("replica grouping stride must be < IPU Link Domain size")

    if is_cross_ild(replica_grouping):
        return ir.replica_grouping(stride=replica_grouping.stride, group_size=ild_size // replica_grouping.stride)
    else:
        return replica_grouping


def is_cross_ild(replica_grouping: popxl.ReplicaGrouping):
    return replica_grouping.group_size * replica_grouping.stride > get_ild_size_from_popdist()
