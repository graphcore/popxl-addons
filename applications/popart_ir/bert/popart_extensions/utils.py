# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Iterable, Mapping, Tuple
import numpy as np

import popart
import popart.ir as pir
from popart.ir.streams import DeviceToHostStream


def run_ir(ir: pir.Ir, inputs: Mapping[str, np.ndarray], outputs: Iterable[DeviceToHostStream], weights: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
    dataFlow = popart.DataFlow(
        batchesPerStep=1,
        anchorTensors={output.tensor_id(): popart.AnchorReturnType("All") for output in outputs})
    _ir = ir._pb_ir
    _ir.logIr()
    _ir.setDataFlow(dataFlow)

    opts = ir._pb_ir.getSessionOptions()
    opts.constantWeights = False
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    _ir.updateVertices()
    _ir.setIsPrepared()

    device = popart.DeviceManager().createIpuModelDevice({"numIPUs": 1})

    session = popart.InferenceSession.fromIr(
        ir=_ir, deviceInfo=device)

    session.prepareDevice()

    session.writeWeights(popart.PyWeightsIO(weights))
    session.weightsFromHost()

    # Create buffers for anchors
    anchors: Dict[str, np.ndarray] = session.initAnchorArrays()  # type: ignore

    stepio = popart.PyStepIO(
        inputs=inputs,
        outputs=anchors)
    session.run(stepio)

    return tuple(anchors[output.tensor_id()] for output in outputs)
