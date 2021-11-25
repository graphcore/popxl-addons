# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
from typing import Iterable, Union, Sequence, Mapping, Tuple, Dict, Optional

import numpy as np
import popart
from popart import ir as pir
from popart.ir.streams import DeviceToHostStream, HostToDeviceStream

from popart_ir_extensions.utils import to_numpy, HostTensor

POPART_CACHE_DIR_ENV = 'POPART_CACHE_DIR'

__all__ = ["Runner"]


class Runner:
    def __init__(self,
                 ir: pir.Ir,
                 outputs: Union[None, DeviceToHostStream, Iterable[DeviceToHostStream]] = None,
                 weights: Optional[Mapping[pir.Tensor, HostTensor]] = None,
                 device_type: Union[str, int] = "cpu",
                 engine_caching: bool = True,
                 replicas: int = 1,
                 device_iterations: int = 1):

        outputs = outputs or []
        if isinstance(outputs, DeviceToHostStream):
            outputs = [outputs]

        dataFlow = popart.DataFlow(
            batchesPerStep=device_iterations,
            anchorTensors={output.tensor_id(): popart.AnchorReturnType("All")
                           for output in outputs})
        _ir = ir._pb_ir
        _ir.setDataFlow(dataFlow)

        opts = ir._pb_ir.getSessionOptions()
        opts.constantWeights = False
        opts.useHostCopyOps = True
        opts.enableExplicitMainLoops = True
        opts.aliasZeroCopy = True
        opts.explicitRecomputation = True

        if replicas > 1:
            opts.enableReplicatedGraphs = True
            opts.replicatedGraphCount = replicas

        cache_dir = os.environ.get(POPART_CACHE_DIR_ENV)
        cache_dir = engine_caching if isinstance(engine_caching, str) else cache_dir
        if cache_dir is not None and engine_caching is not False:
            opts.enableEngineCaching = True
            opts.cachePath = cache_dir

        _ir.removeIsolatedGraphs()
        _ir.removeIsolatedTensors(True)
        _ir.updateVertices()
        _ir.setIsPrepared()
        _ir.logIr()

        if device_type == "hw" or isinstance(device_type, int):
            if not isinstance(device_type, int):
                device_type = 1
            dm = popart.DeviceManager()
            dm.setOnDemandAttachTimeout(int(1e4))
            self.device = dm.acquireAvailableDevice(device_type,
                                                    connectionType=popart.DeviceConnectionType.OnDemand,
                                                    selectionCriterion=popart.DeviceSelectionCriterion.Random)
        elif device_type == "cpu":
            self.device = popart.DeviceManager().createIpuModelDevice({"numIPUs": 1})
        else:
            raise ValueError(f"Do not recognise device type: {device_type}")

        self.session = popart.InferenceSession.fromIr(ir=_ir, deviceInfo=self.device)
        self.session.prepareDevice()

        self.write_weights(weights or {})

        self.ir = ir
        self.outputs = outputs

    def write_weights(self, weights: Mapping[pir.Tensor, HostTensor]):
        if weights:
            for t, ht in weights.items():
                if t.shape != ht.shape:
                    raise ValueError(f"{t} has an incompatible host tensor with shape {ht.shape}")
            self.session.writeWeights(
                popart.PyWeightsIO({t.id: to_numpy(v).astype(t.dtype.as_numpy())
                                    for t, v in weights.items()}))
        self.session.weightsFromHost()

    def read_weights(self, weights: Optional[Iterable[pir.Tensor]] = None) -> Mapping[pir.Tensor, HostTensor]:
        if weights is None:
            weights = self.ir.main_graph().get_variables()
        self.session.weightsToHost()
        result: Dict[str, np.ndarray] = {t.id: np.zeros(t.shape, t.dtype.as_numpy())
                                         for t in weights}  # type: ignore np.zeros returns Any
        self.session.readWeights(popart.PyWeightsIO(result))
        return {t: result[t.id] for t in weights}

    def detach(self):
        self.device.detach()

    @property
    def expected_inputs(self):
        stream_tensors = (pir.Tensor._from_pb_tensor(t) for t in self.ir._pb_ir.dataStreamTensors())
        return set(HostToDeviceStream._from_tensor(t) for t in stream_tensors)

    def run(
        self,
        inputs: Optional[Mapping[HostToDeviceStream, HostTensor]] = None
    ) -> Union[None, np.ndarray, Tuple[np.ndarray, ...]]:
        inputs = inputs or {}

        if set(inputs.keys()) != set(self.expected_inputs):
            unexpected = {str(s) for s in set(inputs.keys()) - self.expected_inputs}
            missing = {str(s) for s in self.expected_inputs - set(inputs.keys())}
            raise ValueError(f"Unexpected/Missing inputs.\n  Unexpected: {unexpected}\n  Missing: {missing}")

        np_inputs: Mapping[str, np.ndarray] = {t.tensor_id(): to_numpy(v) for t, v in inputs.items()}

        anchors: Dict[str, np.ndarray] = self.session.initAnchorArrays()

        stepio = popart.PyStepIO(inputs=np_inputs, outputs=anchors)
        self.session.run(stepio)

        host_outputs = tuple(anchors[output.tensor_id()] for output in self.outputs)

        if len(host_outputs) == 1:
            return host_outputs[0]
        elif len(host_outputs) > 1:
            return host_outputs
        return
