# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import sys
import subprocess
import pytest
import numpy as np
import popxl
from popxl import ops
from popxl_addons.ops.replicated_strided_collectives import replicated_all_gather_strided, replicated_reduce_scatter_strided, replicated_all_reduce_strided


@pytest.mark.serial
@pytest.mark.parametrize("use_rts", [True, False])
def test_distributed_collectives(use_rts: bool):
    cmd = [
        'poprun', '--num-replicas', '8', '--num-instances', '2', 'python',
        os.path.realpath(__file__), "1" if use_rts else "0"
    ]
    completed = subprocess.run(args=cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    combined_output = str(completed.stdout, 'utf-8')
    try:
        completed.check_returncode()
        return_code_ok = True
    except subprocess.CalledProcessError:
        return_code_ok = False

    if not return_code_ok:
        raise RuntimeError(f"The following command failed: {cmd}\nOutput of failed command:\n{combined_output}")


if __name__ == "__main__":
    use_rts = bool(int(sys.argv[1]))

    ir = popxl.Ir(replication='popdist')
    tp = 2
    np.random.seed(42)
    data = np.random.normal(0, 1, (ir.replication_factor, )).reshape((tp, -1))

    dp_group = ir.replica_grouping(stride=tp)
    within_instance = ir.replica_grouping(stride=tp, group_size=ir.instance_replication_factor // tp)
    across_instance = ir.replica_grouping(stride=ir.instance_replication_factor)

    with ir.main_graph:
        v = popxl.variable(data, popxl.float32, replica_grouping=dp_group)

        within = replicated_reduce_scatter_strided(v,
                                                   group=within_instance,
                                                   configure_output_for_replicated_tensor_sharding=use_rts)

        across = replicated_all_reduce_strided(within, group=across_instance)

        reduced = replicated_all_gather_strided(across, group=within_instance)

        d2h = popxl.d2h_stream(reduced.shape, reduced.dtype)
        ops.host_store(d2h, reduced)

    reduced_data = data * dp_group.group_size

    with popxl.Session(ir, 'ipu_hw') as sess:
        output = sess.run()[d2h]

        for idx, out in enumerate(output):
            np.testing.assert_allclose(out, reduced_data[idx % tp])
