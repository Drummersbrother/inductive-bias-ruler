import json
import os
from pathlib import Path
from typing import Any, Optional, List, Dict, Union

from neptune.new import init_run
from pytorch_lightning.loggers.neptune import NeptuneLogger

from src import MODELS, NEPTUNE_LOGS


class NeptuneLogging(NeptuneLogger):
    """Simple wrapper around Pytorch-Lightning's NeptuneLogger, with useful additions:
    * Allows usage as context manager (automatically stopping at the end)
    * Has extra properties like run_id
    * Can log/upload to local disk & neptune at the same time
        * These functions are named local_(usual neptune.experiment function name)
        * Also has easier logging functions, i.e. not on neptune.experiment, but on this object
            * Use your IDE's type inference to differentiate NeptuneLogging and NeptuneLogger methods
    * Automatically inserts the project name into init params
    """

    def __init__(self, **kwargs):
        # No args allowed, specify what you mean!
        default_kwargs = dict(
            capture_stderr=False,  # See https://github.com/Lightning-AI/lightning/issues/1261
            capture_stdout=False,
            log_model_checkpoints=False,
            capture_hardware_metrics=False, # Due to our reopenings of runs all the time, logging monitor metrics will
            # produce loads of MetadataInconsistency exceptions when comparing models. Nothing that stops the code from
            # running, but does pollute stderr unnecessarily
        )
        kwargs = dict(default_kwargs, **kwargs)
        if "with_id" in kwargs:
            kwargs["run"] = kwargs["with_id"]
            del kwargs["with_id"]
            assert isinstance(kwargs["run"], str)
            super().__init__(run=init_run(**kwargs))
        else:
            super().__init__(**kwargs)

        self.run_id = str(self.experiment["sys/id"].fetch())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize("Success")
        self.experiment.stop()

    @property
    def save_dir(self) -> Optional[str]:
        """Overrides neptune's default saving behaviour to not take cwd into account

        Returns:
            the root directory where experiment logs get saved
        """
        return str(NEPTUNE_LOGS)

    @property
    def local_filepath(self) -> Path:
        lp = MODELS / self.run_id
        os.makedirs(lp, exist_ok=True)
        return lp

    def local_upload_json(self, obj: Any, to_path: str):
        """
        :param obj: Object to upload
        :param to_path: Path to save to, equivalent to experiment[to_path] and relative to NeptuneLogging.local_filepath
            Automatically removes(!) .json file ending if it exists
            Replaces /s with folder locally
        """
        if to_path.endswith(".json"):
            to_path.removesuffix(".json")
        to_path_pathed = to_path.replace("/", os.sep)
        local_path = str(self.local_filepath / (to_path_pathed+".json"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as file:
            json.dump(obj, file)
        self.experiment[to_path].upload(local_path)

    def local_download_json(self, from_path: str) -> Union[List, Dict]:
        if from_path.endswith(".json"):
            from_path.removesuffix(".json")
        from_path_pathed = from_path.replace("/", os.sep)
        local_path = str(self.local_filepath / (from_path_pathed+".json"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.experiment[from_path].download(local_path)
        with open(local_path, "r") as file:
            obj = json.load(file)
        return obj

    def log(self, value: Any, to: str):
        """
        Like neptune.experiment[to].log(value)
        """
        self.experiment[to].log(value)
