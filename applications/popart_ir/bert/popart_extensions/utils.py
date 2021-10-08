# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Sequence, Mapping, Tuple, Union
import numpy as np

import popart
import popart.ir as pir
import torch
from popart.ir.streams import HostToDeviceStream, DeviceToHostStream

HostTensor = Union[np.ndarray, torch.Tensor]


def _to_numpy(a: HostTensor) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a.copy()
    if isinstance(a, torch.Tensor):
        return a.detach().numpy().copy()
    else:
        raise ValueError(f"Do not recognise type: {a}")


def run_ir(ir: pir.Ir, inputs: Mapping[HostToDeviceStream, HostTensor],
           outputs: Union[DeviceToHostStream, Sequence[DeviceToHostStream]],
           weights: Mapping[pir.Tensor, HostTensor], device_type="hw") -> Tuple[np.ndarray, ...]:
    inputs: Mapping[str, np.ndarray] = {t.tensor_id(): _to_numpy(v) for t, v in inputs.items()}
    weights: Mapping[str, np.ndarray] = {t.id: _to_numpy(v) for t, v in weights.items()}
    try:
        outputs = list(outputs)
    except:
        outputs = [outputs]

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

    if device_type == "hw":
        device = popart.DeviceManager().createIpuModelDevice({"numIPUs": 1})
    elif device_type == "cpu":
        device = popart.DeviceManager().createCpuDevice()
    else:
        raise ValueError(f"Do not recognise device type: {device_type}")

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

    host_outputs = tuple(anchors[output.tensor_id()] for output in outputs)

    return host_outputs
