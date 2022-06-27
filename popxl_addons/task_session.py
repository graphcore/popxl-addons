# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import logging
import tempfile
from time import time
from typing import Dict, Iterable, Mapping, Union
from typing_extensions import Literal
import numpy as np

import popxl
from popxl_addons import InputStreams, OutputStreams
from popxl_addons.named_tensors import NamedTensors

# TODO: remove this method once T61041 has landed and use normal `Session.write_variables_data`
import popart

__all__ = ["TaskSession", "write_variables_pb"]

ReportMissingMethods = Literal['none', 'warn', 'error']


# TODO: remove this method once T61041 has landed and use normal `Session.write_variables_data`
def write_variables_pb(session: popxl.Session, weights: Mapping[popxl.Tensor, np.ndarray]):
    weightsIo = popart.PyWeightsIO({xl_t.id: np_t for xl_t, np_t in weights.items()})
    session._pb_session.writeWeights(weightsIo)
    if session.is_attached:
        session._pb_session.weightsFromHost()


class TaskSession(popxl.Session):
    def __init__(self, inputs: Union[InputStreams, Iterable[popxl.HostToDeviceStream]],
                 outputs: Union[OutputStreams, Iterable[popxl.DeviceToHostStream]], model: NamedTensors, *args,
                 **kwargs):
        t = time()
        super().__init__(*args, **kwargs)
        logging.info(f"popxl compilation duration: {(time() - t) / 60:.2f} mins")
        self.inputs: InputStreams = inputs if isinstance(inputs, InputStreams) else InputStreams.from_streams(inputs)
        self.outputs: OutputStreams = outputs if isinstance(outputs,
                                                            OutputStreams) else OutputStreams.from_streams(outputs)
        self.model = model

    def save_checkpoint(self, file_path: str):
        t_ckpt = self.get_tensors_data(self.model.tensors)
        ckpt = {}
        for name, t in self.model.to_dict().items():
            ckpt[name] = t_ckpt[t]

        if file_path.startswith("wandb://"):
            self.save_checkpoint_to_wandb(file_path.replace("wandb://", ""), ckpt)
        else:
            self.save_checkpoint_to_file(file_path, ckpt)

    def save_checkpoint_to_file(self, file_path: str, ckpt: Mapping[str, np.ndarray]):
        np.savez(file_path, ckpt=ckpt)

    def save_checkpoint_to_wandb(self, name: str, ckpt: Mapping[str, np.ndarray]):
        import wandb
        artifact = wandb.Artifact(name=name, type="ckpt")
        with tempfile.TemporaryDirectory() as td:
            file_path = os.path.join(td, "model.npz")
            self.save_checkpoint_to_file(file_path, ckpt)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)

    def load_checkpoint(self, file_path: str, report_missing: ReportMissingMethods = 'warn'):
        if file_path.startswith("wandb://"):
            self.load_checkpoint_from_wandb(file_path.replace("wandb://", ""), report_missing)
        else:
            self.load_checkpoint_from_file(file_path, report_missing)

    def load_checkpoint_from_file(self, file_path: str, report_missing: ReportMissingMethods = 'warn'):
        loaded = np.load(file_path, allow_pickle=True)
        assert loaded is not None

        ckpt = loaded["ckpt"].item()
        variables = self.model.to_dict()

        ckpt_keys = set(ckpt.keys())
        model_keys = set(variables.keys())
        self._report_missing(ckpt_keys, model_keys, report_missing)

        self.write_variables_data({variables[name]: ckpt[name] for name in model_keys.intersection(ckpt_keys)})

    def load_checkpoint_from_wandb(self, name: str, report_missing: ReportMissingMethods = 'warn'):
        import wandb
        artifact = wandb.use_artifact(name, type='ckpt')
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, "model.npz")
        self.load_checkpoint_from_file(file_path, report_missing)

    def load_from_session(self, src: 'TaskSession', report_missing: ReportMissingMethods = 'none'):
        loaded = src.get_tensors_data(src.model.tensors)
        self.load_from_tensors(src.model, loaded, report_missing)

    def load_from_tensors(self,
                          src: NamedTensors,
                          loaded: Mapping[popxl.Tensor, np.ndarray],
                          report_missing: ReportMissingMethods = 'none'):
        self._report_missing(src.to_dict().keys(), self.model.to_dict().keys(), report_missing)

        src_dst = src.to_mapping(self.model)

        self.write_variables_data({src_dst[src_t]: t for src_t, t in loaded.items() if src_t in src_dst.keys()})

    def load_from_session_pb(self,
                             src: 'TaskSession',
                             loaded: Mapping[popxl.Tensor, np.ndarray],
                             report_missing: ReportMissingMethods = 'none'):
        # TODO: remove this method once T56776 has landed and use normal `load_from_tensors`
        src = src.model
        self._report_missing(src.to_dict().keys(), self.model.to_dict().keys(), report_missing)
        src_dst = src.to_mapping(self.model)
        write_variables_pb(self, {src_dst[src_t]: t for src_t, t in loaded.items() if src_t in src_dst.keys()})

    def _report_missing(self, ckpt_keys: Iterable[str], model_keys: Iterable[str], method: ReportMissingMethods):
        if method is not 'none':
            ckpt_keys = set(ckpt_keys)
            model_keys = set(model_keys)

            message = ""
            if ckpt_keys - model_keys:
                message += f"Checkpoint Tensors not in model: {ckpt_keys - model_keys}"
            if model_keys - ckpt_keys:
                if message:
                    message += "\n"
                message += f"Model Tensors not in checkpoint: {model_keys - ckpt_keys}"
            if message:
                if method == 'error':
                    raise ValueError(message)
                else:
                    logging.warning(message)

    def wandb_variable_histograms(self, step: int = 0):
        """Track all variables as a histogram in Weights & Biases"""
        import wandb
        for t, np_t in self.get_tensors_data(self.ir.main_graph.variables).items():
            np_t = np_t.flatten().astype(np.float32)
            wandb.log({t.name: wandb.Histogram(np_t)}, step=step)
