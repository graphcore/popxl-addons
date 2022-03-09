# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import time
from typing import Iterable, Union, Mapping, Tuple, Dict, Optional
from typing_extensions import Literal

import numpy as np
import popart
import popxl
from popxl.streams import DeviceToHostStream, HostToDeviceStream
from popxl.tensor import HostTensor
from popxl_addons.utils import to_numpy

from math import ceil, log

__all__ = ["Runner"]


class Runner:
    def __init__(self,
                 ir: popxl.Ir,
                 outputs: Union[None, DeviceToHostStream, Iterable[DeviceToHostStream]] = None,
                 weights: Optional[Mapping[popxl.Tensor, HostTensor]] = None,
                 device_type: Literal['cpu', 'hw'] = 'hw',
                 device_num: int = 1,
                 device_iterations: int = 1):

        outputs = outputs or []
        if isinstance(outputs, DeviceToHostStream):
            outputs = [outputs]

        dataFlow = popart.DataFlow(
            batchesPerStep=device_iterations,
            anchorTensors={output.tensor_id: popart.AnchorReturnType("All")
                           for output in outputs})
        _ir = ir._pb_ir
        _ir.setDataFlow(dataFlow)

        opts = ir._pb_ir.getSessionOptions()
        opts.constantWeights = False
        opts.useHostCopyOps = True
        opts.enableExplicitMainLoops = True
        opts.aliasZeroCopy = True
        opts.explicitRecomputation = True

        if device_type == "hw":
            dm = popart.DeviceManager()
            dm.setOnDemandAttachTimeout(int(1e4))
            self.device = dm.acquireAvailableDevice(device_num,
                                                    connectionType=popart.DeviceConnectionType.OnDemand,
                                                    selectionCriterion=popart.DeviceSelectionCriterion.Random)
        elif device_type == "cpu":
            self.device = popart.DeviceManager().createIpuModelDevice({"numIPUs": device_num})
        else:
            raise ValueError(f"Do not recognise device type: {device_type}")

        ir_ipus = set(ipu for g in _ir.getAllGraphs() for ipu in g.getAllVirtualGraphIds(True))
        max_ipus = max(ir_ipus) + 1
        if opts.enableReplicatedGraphs:
            max_ipus *= opts.replicatedGraphCount
        if max_ipus > device_num:
            raise ValueError(f"The IR uses {max_ipus} IPUs but you have only requested to acquire {device_num}. "
                             f"Please request {2**ceil(log(max_ipus))} IPUs (must be power of 2).")

        _ir.removeIsolatedGraphs()
        _ir.removeIsolatedTensors(True)

        for g in _ir.getAllGraphs():
            _ir.applyPreAliasPatterns(g)

        _ir.updateVertices()
        _ir.logIr()

        compile_start = time.perf_counter()
        self.session = popart.InferenceSession.fromIr(ir=_ir, deviceInfo=self.device)

        # TODO move this to _ir instead
        self.session.checkInplacingAmbiguity()

        self.session.prepareDevice()
        print(f"Compiled. Duration {time.perf_counter() - compile_start:.1f} seconds")

        self.write_weights(weights or {})

        self.ir = ir
        self.outputs = outputs

    def write_weights(self, weights: Mapping[popxl.Tensor, HostTensor]):
        if weights:
            for t, ht in weights.items():
                if t.shape != ht.shape:
                    raise ValueError(f"{t} has an incompatible host tensor with shape {ht.shape}")
            self.session.writeWeights(
                popart.PyWeightsIO({t.id: to_numpy(v, dtype=t.dtype.as_numpy())
                                    for t, v in weights.items()}))
        self.session.weightsFromHost()

    def read_weights(self, weights: Optional[Iterable[popxl.Tensor]] = None) -> Dict[popxl.Tensor, np.ndarray]:
        if weights is None:
            weights = self.ir.main_graph.variables
        self.session.weightsToHost()
        result: Dict[str, np.ndarray] = {t.id: np.zeros(t.shape, t.dtype.as_numpy())
                                         for t in weights}  # type: ignore np.zeros returns Any
        self.session.readWeights(popart.PyWeightsIO(result))
        return {t: result[t.id] for t in weights}

    def detach(self):
        self.device.detach()

    def __del__(self):
        self.detach()

    @property
    def expected_inputs(self):
        stream_tensors = (popxl.Tensor._from_pb_tensor(t) for t in self.ir._pb_ir.dataStreamTensors())
        return set(HostToDeviceStream._from_tensor(t) for t in stream_tensors)

    def run(self, inputs: Optional[Mapping[HostToDeviceStream, HostTensor]] = None) -> Tuple[np.ndarray, ...]:
        inputs = inputs or {}

        if set(inputs.keys()) != set(self.expected_inputs):
            unexpected = {str(s) for s in set(inputs.keys()) - self.expected_inputs}
            missing = {str(s) for s in self.expected_inputs - set(inputs.keys())}
            raise ValueError(f"Unexpected/Missing inputs.\n  Unexpected: {unexpected}\n  Missing: {missing}")

        np_inputs: Mapping[str, np.ndarray] = {t.tensor_id: to_numpy(v) for t, v in inputs.items()}

        anchors: Dict[str, np.ndarray] = self.session.initAnchorArrays()

        stepio = popart.PyStepIO(inputs=np_inputs, outputs=anchors)
        self.session.run(stepio)

        host_outputs = tuple(anchors[output.tensor_id] for output in self.outputs)

        if len(host_outputs) == 1:
            return host_outputs[0]
        elif len(host_outputs) > 1:
            return host_outputs
        return
