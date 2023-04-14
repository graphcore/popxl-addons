# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import logging
import tempfile
from time import time
from typing import Dict, Iterable, Mapping, Union, Optional
from typing_extensions import Literal
import numpy as np
import shutil
from collections import deque

import popxl
from popxl_addons import InputStreams, OutputStreams
from popxl_addons.named_tensors import NamedTensors, NamedTensorData
from popxl_addons.utils import timer
import json
from popxl.session import Session, d2hStreamBufferMaps, h2dStreamBufferMaps
import popart
import popdist
import glob

__all__ = ["TaskSession", "write_variables_pb"]

ReportMissingMethods = Literal["none", "warn", "error"]


# TODO: remove this method once T61041 has landed and use normal `Session.write_variables_data`
def write_variables_pb(session: Session, weights: Mapping[popxl.Tensor, np.ndarray]):
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

    def __init__(
        self,
        inputs: Union[InputStreams, Iterable[popxl.HostToDeviceStream]],
        outputs: Union[OutputStreams, Iterable[popxl.DeviceToHostStream]],
        state: NamedTensors,
        max_checkpoints: int = 1,
        *args,
        **kwargs,
    ):
        with timer("PopXL compilation"):
            super().__init__(*args, **kwargs)
        self.inputs: InputStreams = inputs if isinstance(inputs, InputStreams) else InputStreams.from_streams(inputs)
        self.outputs: OutputStreams = (
            outputs if isinstance(outputs, OutputStreams) else OutputStreams.from_streams(outputs)
        )
        self.state = state
        self.session_state = {"steps": 0}
        self.__dataloader = None
        self.__session_filename = r"session_info.json"
        self.__dataloader_filename = r"dataloader.bin"
        self.__checkpoints = deque([])
        self.__max_checkpoints = max_checkpoints

    @property
    def dataloader(self):
        if self.__dataloader is None:
            logging.warning(
                "No dataloader set. Checkpoint support is incomplete. You won't be able to resume training."
            )
        return self.__dataloader

    @dataloader.setter
    def dataloader(self, dl):
        if hasattr(dl, "save") and hasattr(dl, "resume"):
            self.__dataloader = dl
        else:
            raise ValueError("Dataloader must implement save and resume methods")

    def add_session_state_info(self, info: Dict):
        """
        Add any extra info required to resume training.
        For example, model configuration, total training steps, or learning rate schedule.
        All the state is dumped to a json file in checkpoints.
        """
        self.session_state.update(info)

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

    def load_from_session(self, src: "TaskSession", report_missing: ReportMissingMethods = "none"):
        """
        Copy variables data from `src` session.
        """
        loaded = src.get_tensors_data(src.state.tensors)
        self.__load_from_tensors(src.state, loaded, report_missing)

    def wandb_histogram(self, name: str, step: int, data: np.ndarray):
        """Histogram data using Weights & Biases"""
        import wandb

        data = data.flatten().astype(np.float32)
        finite_mask = np.isfinite(data)
        finite_data = data[finite_mask]
        n_non_finite = np.sum(~finite_mask)
        wandb.log({f"{name}:histogram": wandb.Histogram(finite_data), f"{name}:not_finite": n_non_finite}, step=step)

    def wandb_variable_histograms(self, step: int = 0):
        """Track all variables as a histogram in Weights & Biases"""
        for t, np_t in self.get_tensors_data(self.ir.main_graph.variables).items():
            self.wandb_histogram(t.name, step, np_t)

    def save_checkpoint(self, checkpoint_dir: str):
        """
        Save a checkpoint for the task, consisting of the model state (weights), the optimiser state, the dataloader state and
        the session state (always containing the step) which can be customised with application-specific requirements.

        Tensors are saved in .npz format (see `numpy savez <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`__. )
        The dataloader state is saved in binary format.
        Session state is saved in json format.

        In case of distributed training, only instance 0 saves on file.
        Args:
            checkpoint_dir (str): directory to save the checkpoint.
        """
        # weightsFromHost calls must be performed by all instances,
        # since they it involves cross-instances collectives
        with timer(f"Get tensor data from IPU ... "):
            state = self.get_named_tensors_data().to_dict()

        if popdist.getInstanceIndex() == 0:
            with timer(f"Saving checkpoint {checkpoint_dir} ... "):
                if checkpoint_dir.startswith("wandb://"):
                    import wandb

                    checkpoint_dir = checkpoint_dir.replace("wandb://", "")
                    artifact = wandb.Artifact(name=checkpoint_dir, type=f"checkpoints")
                    with tempfile.TemporaryDirectory() as td:
                        os.makedirs(os.path.join(td, "model"), exist_ok=True)
                        self.__save_checkpoint_to_file(td, state)
                        artifact.add_dir(td)
                    wandb.log_artifact(artifact)
                    # not able to delete artifacts
                else:
                    checkpoint_dir = os.path.relpath(os.path.expanduser(checkpoint_dir))
                    if checkpoint_dir != "":
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        os.makedirs(os.path.join(checkpoint_dir, "model"), exist_ok=True)
                    if len(self.__checkpoints) >= self.__max_checkpoints:
                        to_delete = self.__checkpoints.popleft()
                        try:
                            logging.info(f"\t Deleting old checkpoint {to_delete} ... ")
                            shutil.rmtree(to_delete)
                        except OSError as e:
                            print(f"\t Failed to remove {to_delete} ", e)

                    self.__save_checkpoint_to_file(checkpoint_dir, state)
                    self.__checkpoints.append(checkpoint_dir)

    def load_checkpoint(
        self, checkpoint_dir: str, report_missing: ReportMissingMethods = "warn", skip_memmap: bool = False
    ):
        """
        Load a checkpoint for the task, consisting of the model state (weights), the optimiser state, the dataloader state and
        the session state (always containing the step) which can be customised with application-specific requirements.

        Tensors are saved in .npz format (see `numpy savez <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`__. )
        The dataloader state is saved in binary format.
        Session state is saved in json format.

        Args:
            checkpoint_dir (str): directory to save the checkpoint.
        """
        with timer(f"Loading checkpoint {checkpoint_dir} ... "):
            if checkpoint_dir.startswith("wandb://"):
                import wandb

                checkpoint_dir = checkpoint_dir.replace("wandb://", "")
                artifact = wandb.use_artifact(checkpoint_dir, type="checkpoints")
                artifact_dir = artifact.download()
                self.__load_checkpoint_from_file(artifact_dir, report_missing, skip_memmap)
            else:
                self.__load_checkpoint_from_file(checkpoint_dir, report_missing, skip_memmap)

    def __save_checkpoint_to_file(self, checkpoint_dir: str, state: Dict):
        # save model state
        logging.info("\t Saving model state...")
        for name, var in state.items():
            if isinstance(var.base, np.memmap):
                mmap_path: str = var.base.filename
                shutil.copyfile(mmap_path, os.path.join(checkpoint_dir, "model", os.path.basename(mmap_path)))
            else:
                np.savez(os.path.join(checkpoint_dir, "model", f"{name}.npz"), **{name: var})

        # save dataloader state
        if self.dataloader:
            logging.info("\t Saving dataloader state...")
            self.dataloader.save(os.path.join(checkpoint_dir, self.__dataloader_filename))

        # session info
        with open(os.path.join(checkpoint_dir, self.__session_filename), "w") as f:
            logging.info(f"\t Saving session state...")
            json.dump(self.session_state, f)

    def __load_checkpoint_from_file(
        self, checkpoint_dir: str, report_missing: ReportMissingMethods = "warn", skip_memmap: bool = False
    ):
        """
        Load checkpoint from a file in .npz format or from wandb.
        Data is read from the checkpoint and written to the ipu to the corresponding `state` variables.
        Args:
            file_path: path to the .npz checkpoint or to wandb checkpoint (starting with "wandb://")
            report_missing: how to report missing keys in the checkpoint.
        """
        variables = self.state.to_dict()

        # check missing keys
        ckpt = {}
        files = glob.glob(os.path.join(checkpoint_dir, "model", "*.npz"))

        def filename_from_var_name(name: str):
            return os.path.join(checkpoint_dir, "model", f"{name}.npz")

        expected = map(filename_from_var_name, variables.keys())
        files = set(files)
        expected = set(expected)
        self.__report_missing(files, expected, report_missing)
        existing_keys = files.intersection(expected)

        # load existing keys
        for filename in existing_keys:
            loaded = np.load(filename, allow_pickle=True)
            assert loaded is not None
            ckpt.update({name: loaded[name] for name in loaded.files})

        # write variables data to IPU for existing keys
        data = {}
        for var_name in variables:
            if var_name in ckpt:
                data[variables[var_name]] = ckpt[var_name]

        self.write_variables_data(data)
        logging.info("\t Loaded model state")

        # dataloader
        if self.dataloader:
            self.dataloader.resume(filename=os.path.join(checkpoint_dir, self.__dataloader_filename))
            logging.info(
                f"\t Loaded dataloader state: {self.dataloader.get_state()}",
            )

        # session info
        with open(os.path.join(checkpoint_dir, self.__session_filename), "r") as f:
            loaded_state = json.load(f)
            self.__report_missing(loaded_state.keys(), self.session_state.keys(), report_missing)
            self.session_state = loaded_state
        logging.info(f"\t Loaded session state: {self.session_state}")

    def __load_from_tensors(
        self,
        src: NamedTensors,
        loaded: Mapping[popxl.Tensor, np.ndarray],
        report_missing: ReportMissingMethods = "none",
    ):
        self.__report_missing(src.to_dict().keys(), self.state.to_dict().keys(), report_missing)

        src_dst = src.to_mapping(self.state)

        self.write_variables_data({src_dst[src_t]: t for src_t, t in loaded.items() if src_t in src_dst.keys()})

    def __report_missing(self, ckpt_keys: Iterable[str], state_keys: Iterable[str], method: ReportMissingMethods):
        if method != "none":
            ckpt_keys = set(ckpt_keys)
            state_keys = set(state_keys)

            message = ""
            if ckpt_keys - state_keys:
                message += f"Checkpoint keys not in state: {ckpt_keys - state_keys}"
            if state_keys - ckpt_keys:
                if message:
                    message += "\n"
                message += f"state keys not in checkpoint: {state_keys - ckpt_keys}"
            if message:
                if method == "error":
                    raise ValueError(message)
                else:
                    logging.warning(message)

    def run(
        self,
        inputs: Optional[h2dStreamBufferMaps] = None,
        downcast_inputs: bool = True,
    ) -> d2hStreamBufferMaps:
        """
        Run the program and keep track of the times `session.run` is called.
        """
        self.session_state["steps"] += 1
        return super().run(inputs, downcast_inputs)


def initialise_memmap_dir_from_checkpoint(checkpoint_dir, memmap_dir):
    """Copy memmap variable data from a checkpoint to a memmap_dir"""
    files = glob.glob(os.path.join(checkpoint_dir, "model", "*.npy"))
    for f in files:
        shutil.copyfile(f, os.path.join(memmap_dir, os.path.basename(f)))
