# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
from typing import Union, Sequence, Mapping, Tuple, Dict, Optional

import numpy as np
import popart
from popart import ir as pir
from popart.ir.streams import DeviceToHostStream, HostToDeviceStream

from popart_ir_extensions.utils import to_numpy, HostTensor

POPART_CACHE_DIR_ENV = 'POPART_CACHE_DIR'


class Runner:
    def __init__(self,
                 ir: pir.Ir,
                 outputs: Union[None, DeviceToHostStream, Sequence[DeviceToHostStream]] = None,
                 weights: Optional[Mapping[pir.Tensor, HostTensor]] = None,
                 device_type: Union[str, int] = "cpu",
                 engine_caching: bool = True,
                 replicas: Optional[int] = None):
        outputs = outputs if outputs is not None else {}
        weights = weights if weights is not None else {}

        weights: Mapping[str, np.ndarray] = {t.id: to_numpy(v) for t, v in weights.items()}
        try:
            outputs = list(outputs)
        except:
            outputs = [outputs]

        dataFlow = popart.DataFlow(
            batchesPerStep=1, anchorTensors={output.tensor_id(): popart.AnchorReturnType("All")
                                             for output in outputs})
        _ir = ir._pb_ir
        _ir.logIr()
        _ir.setDataFlow(dataFlow)

        opts = ir._pb_ir.getSessionOptions()
        opts.constantWeights = False
        opts.useHostCopyOps = True
        opts.enableExplicitMainLoops = True
        opts.aliasZeroCopy = True
        opts.explicitRecomputation = True

        if isinstance(replicas, int):
            opts.enableReplicatedGraphs = True
            opts.replicatedGraphCount = replicas

        cache_dir = os.environ.get(POPART_CACHE_DIR_ENV)
        cache_dir = engine_caching if isinstance(engine_caching, str) else cache_dir
        if cache_dir is not None and engine_caching is not False:
            opts.enableEngineCaching = True
            opts.cachePath = cache_dir

        _ir.updateVertices()
        _ir.setIsPrepared()

        if device_type == "hw" or isinstance(device_type, int):
            if not isinstance(device_type, int):
                device_type = 1
            dm = popart.DeviceManager()
            dm.setOnDemandAttachTimeout(int(1e4))
            device = dm.acquireAvailableDevice(device_type,
                                               connectionType=popart.DeviceConnectionType.OnDemand,
                                               selectionCriterion=popart.DeviceSelectionCriterion.Random)
        elif device_type == "cpu":
            device = popart.DeviceManager().createIpuModelDevice({"numIPUs": 1})
        else:
            raise ValueError(f"Do not recognise device type: {device_type}")

        session = popart.InferenceSession.fromIr(ir=_ir, deviceInfo=device)

        session.prepareDevice()

        session.writeWeights(popart.PyWeightsIO(weights))
        session.weightsFromHost()

        self.ir = ir
        self.weights = weights
        self.device_type = device_type
        self.session = session
        self.outputs = outputs

    def run(
        self,
        inputs: Optional[Mapping[HostToDeviceStream, HostTensor]] = None
    ) -> Union[None, np.ndarray, Tuple[np.ndarray, ...]]:
        inputs = inputs if inputs is not None else {}
        inputs: Mapping[str, np.ndarray] = {t.tensor_id(): to_numpy(v) for t, v in inputs.items()}

        anchors: Dict[str, np.ndarray] = self.session.initAnchorArrays()

        stepio = popart.PyStepIO(inputs=inputs, outputs=anchors)
        self.session.run(stepio)

        host_outputs = tuple(anchors[output.tensor_id()] for output in self.outputs)

        if len(host_outputs) > 1:
            return host_outputs
        elif len(host_outputs) == 1:
            return host_outputs[0]
