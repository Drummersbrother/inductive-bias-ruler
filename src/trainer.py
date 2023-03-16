from __future__ import annotations
from typing import List
import warnings
from warnings import warn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from src import models, datasets, ml_logging, PL_LOGS, NEPTUNE_LOGS

# Py3.9 doesn't support type aliases from PEP613
RunId = str


def train_model(model: models.BaseModule, logger_kwargs: dict = None,
                trainer_kwargs: dict = None) -> RunId:
    """
    This function trains a model and logs the training to a new Neptune run.

    Arguments:
        model: The model to be trained.
        logger_kwargs, trainer_kwargs: Dictionaries of keyword arguments
            for the respective function calls.

        Returns:
            The ID of the neptune run that was logged to as a string.

        Raises:
            KeyboardInterrupt
                If training is interrupted, the pytorchlightning default interrupt handler is run, and the exception is
                reraised (since we don't then want to continue training the next module
    """
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    data_module = datasets.get_data_module(model.dataset_name, dl_kwargs=model.model_config["dl_kwargs"])

    with ml_logging.NeptuneLogging(
            tags=["training", model.model_architecture],
            **({} if logger_kwargs is None else logger_kwargs)) as neptune_logger:
        neptune_logger.local_upload_json(model.model_config, "model_config")
        neptune_logger.log_hyperparams(model.model_config)
        # This log is exclusively for visualisation of the model's layers on Neptune.
        neptune_logger.log(str(model), "model_layers")

        max_epochs = model.model_config["epochs"]
        callbacks = [
            LearningRateMonitor(),
            EarlyStopping("val_avg_accuracy", patience=2, mode="max")
        ]

        # Create an instance of the pl.Trainer class with keyword arguments.
        # kwargs contains default arguments, which will be extended by keyword
        # arguments in trainer_kwargs. Where one key exists in both dictionaries,
        # trainer_kwargs overwrites kwargs.
        kwargs = dict(max_epochs=max_epochs,
                      enable_progress_bar=True,
                      default_root_dir=str(PL_LOGS),
                      logger=neptune_logger, callbacks=callbacks,
                      devices=1, accelerator="auto")
        kwargs.update(trainer_kwargs) if trainer_kwargs is not None else {}
        trainer = pl.Trainer(**kwargs)

        # Train the model on the training data.
        trainer.fit(model, datamodule=data_module)

        if trainer.state.status == pl.trainer.trainer.TrainerStatus.INTERRUPTED:
            print("Model training was interrupted, raising again...")
            raise KeyboardInterrupt

        # Test the model on the test dataloader. This will upload the test accuracy to Neptune.
        trainer.test(model, datamodule=data_module)

        # This has to be fetched within the logging context manager, otherwise the logger is already closed
        run_id = neptune_logger.run_id

    return run_id


def train_models(models: List[models.BaseModule], **train_model_kwargs) -> List[RunId]:
    """
    This function applies the function train_model to a list of models.

    Arguments:
        models: list of initialised models.
        train_model_kwargs: An arbitrary number of keyword arguments to pass on to train_model().

    Returns:
        A list of the Neptune run IDs of the trained models.
    """

    return list(map(lambda model: train_model(model, **(train_model_kwargs.copy())), iter(models)))
