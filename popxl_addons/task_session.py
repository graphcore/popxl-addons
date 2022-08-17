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
from popxl_addons.named_tensors import NamedTensors, NamedTensorData
from popxl_addons.utils import timer

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
    """
    A `popxl` session customised for a specific task.
    Gathers inputs, outputs and variables for the task and provides utilities to save/load the state (variables).
    Usage:

    """

    def __init__(self, inputs: Union[InputStreams, Iterable[popxl.HostToDeviceStream]],
                 outputs: Union[OutputStreams, Iterable[popxl.DeviceToHostStream]], state: NamedTensors, *args,
                 **kwargs):
        with timer('PopXL compilation'):
            super().__init__(*args, **kwargs)
        self.inputs: InputStreams = inputs if isinstance(inputs, InputStreams) else InputStreams.from_streams(inputs)
        self.outputs: OutputStreams = outputs if isinstance(outputs,
                                                            OutputStreams) else OutputStreams.from_streams(outputs)
        self.state = state

    def get_named_tensors_data(self) -> NamedTensorData:
        """
        Retrieves variables' data from IPU, returning a `DotTree` collection
        which associates the variable name to the numpy array with data.
        """
        state_dict_t = self.get_tensors_data(self.state.tensors)
        named_state_dict = {}
        for name, t in self.state.to_dict().items():
            named_state_dict[name] = state_dict_t[t]
        return NamedTensorData.from_dict(named_state_dict)

    def save_checkpoint(self, file_path: str):
        """
        Save the state to file in .npz format (see `numpy savez <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`__. )
        or to wandb.
        Data for variables in `state` is read from IPU and saved to checkpoint.
        Args:
            file_path (str): file to save the checkpoint.
                             If a wandb address is provided (starts with "wandb://" ),
                             checkpoint is saved to wandb.
        """
        ckpt = self.get_named_tensors_data().to_dict()

        if file_path.startswith("wandb://"):
            self._save_checkpoint_to_wandb(file_path.replace("wandb://", ""), ckpt)
        else:
            self._save_checkpoint_to_file(file_path, ckpt)

    def load_checkpoint(self, file_path: str, report_missing: ReportMissingMethods = 'warn'):
        """
        Load checkpoint from a file in .npz format or from wandb.
        Data is read from the checkpoint and written to the ipu to the corresponding `state` variables.
        Args:
            file_path: path to the .npz checkpoint or to wandb checkpoint (starting with "wandb://")
            report_missing: how to report missing keys in the checkpoint.
        """
        if file_path.startswith("wandb://"):
            self._load_checkpoint_from_wandb(file_path.replace("wandb://", ""), report_missing)
        else:
            self._load_checkpoint_from_file(file_path, report_missing)

    def load_from_session(self, src: 'TaskSession', report_missing: ReportMissingMethods = 'none'):
        """
        Copy variables data from `src` session.
        """
        loaded = src.get_tensors_data(src.state.tensors)
        self._load_from_tensors(src.state, loaded, report_missing)

    def wandb_variable_histograms(self, step: int = 0):
        """Track all variables as a histogram in Weights & Biases"""
        import wandb
        for t, np_t in self.get_tensors_data(self.ir.main_graph.variables).items():
            np_t = np_t.flatten().astype(np.float32)
            finite_mask = np.isfinite(np_t)
            finite_data = np_t[finite_mask]
            n_non_finite = np.sum(~finite_mask)
            wandb.log({
                f'{t.name}:histogram': wandb.Histogram(finite_data),
                f'{t.name}:not_finite': n_non_finite
            },
                      step=step)

    def _save_checkpoint_to_file(self, file_path: str, ckpt: Mapping[str, np.ndarray]):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez(file_path, ckpt=ckpt)

    def _save_checkpoint_to_wandb(self, name: str, ckpt: Mapping[str, np.ndarray]):
        import wandb
        artifact = wandb.Artifact(name=name, type="ckpt")
        with tempfile.TemporaryDirectory() as td:
            file_path = os.path.join(td, "state.npz")
            self._save_checkpoint_to_file(file_path, ckpt)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)

    def _load_checkpoint_from_file(self, file_path: str, report_missing: ReportMissingMethods = 'warn'):
        loaded = np.load(file_path, allow_pickle=True)
        assert loaded is not None

        ckpt = loaded["ckpt"].item()
        variables = self.state.to_dict()

        ckpt_keys = set(ckpt.keys())
        state_keys = set(variables.keys())
        self._report_missing(ckpt_keys, state_keys, report_missing)

        self.write_variables_data({variables[name]: ckpt[name] for name in state_keys.intersection(ckpt_keys)})

    def _load_checkpoint_from_wandb(self, name: str, report_missing: ReportMissingMethods = 'warn'):
        import wandb
        artifact = wandb.use_artifact(name, type='ckpt')
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, "state.npz")
        self.load_checkpoint_from_file(file_path, report_missing)

    def _load_from_tensors(self,
                           src: NamedTensors,
                           loaded: Mapping[popxl.Tensor, np.ndarray],
                           report_missing: ReportMissingMethods = 'none'):
        self._report_missing(src.to_dict().keys(), self.state.to_dict().keys(), report_missing)

        src_dst = src.to_mapping(self.state)

        self.write_variables_data({src_dst[src_t]: t for src_t, t in loaded.items() if src_t in src_dst.keys()})

    def _report_missing(self, ckpt_keys: Iterable[str], state_keys: Iterable[str], method: ReportMissingMethods):
        if method is not 'none':
            ckpt_keys = set(ckpt_keys)
            state_keys = set(state_keys)

            message = ""
            if ckpt_keys - state_keys:
                message += f"Checkpoint Tensors not in state: {ckpt_keys - state_keys}"
            if state_keys - ckpt_keys:
                if message:
                    message += "\n"
                message += f"state Tensors not in checkpoint: {state_keys - ckpt_keys}"
            if message:
                if method == 'error':
                    raise ValueError(message)
                else:
                    logging.warning(message)
