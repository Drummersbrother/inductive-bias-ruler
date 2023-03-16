from typing import Tuple, List

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn


class BaseModule(pl.LightningModule):
    """
    This class is a LightningModule that configures itself according to model_config.
    Attributes:
        model_config: dict: {
                            "conv_layers": list of dict: [{"out_channels": int, "kernel_size": int, "stride": int, "padding": str or int, other kwargs for LazyConv2d},
                                                           {...}...]
                            "activation_fct": dict: {"name": str, "kwargs": dict}
                            "post_activation_fct": list of dict: [{"name": str, "kwargs": dict}, ... ]
                            "post_conv_fct": list of dict: [{"name": str, "kwargs": dict}, ... ]
                            "linear_layers": list of dict: [{"out_features": int}, {...}, ...]
                            "final_activation_fct": list of dict: [{"name": str, "kwargs": dict}, ...]
                            "datasets": str, "loss_function": str, "learning_rate": float,
                            "learning_rate_scheduler": Tuple[str, dict], "dl_kwargs": dict,
                            "epochs": int, "optimizer": Tuple[str, dict]
                            }
                            The layers will be built up in the following way:
                            1)  "conv_layers"[0]
                                "activation_fct"
                                "post_activation_fct"
                                "conv_layers"[1]
                                "activation_fct"
                                "post_activation_fct"
                                ...
                            2)  "post_conv_layers"
                            3)  nn.Flatten()
                            4)  "linear_layers"[0]
                                "linear_layers"[1]
                                ...
                            5)  "final_activation_fct"
        layers: nn.ModuleList built according to model_config.
        criterion: The loss function according to model_config.
        accuracy: instance of torchmetrics.Accuracy. Called to calculate the ratio of accurate predictions.
    """

    def __init__(self, model_config: dict = None):
        assert model_config is not None, "Model config was None, please pass a valid model config dict!"
        self.model_config = model_config
        super().__init__()
        self.model_architecture = "FC"  # Is automatically changed in _get_layers(), if needed.
        self.dataset_name = self.model_config["dataset_name"]
        self.layers = self._get_layers()  # Sequential
        self.criterion = getattr(nn, self.model_config["loss_function"])()
        self.learning_rate = self.model_config["learning_rate"]
        self.accuracy = torchmetrics.Accuracy(num_classes=self.model_config["number_of_classes"], task="multiclass")

        self.epoch_metrics = {"train": [], "validation": [], "test": []}

    def _get_layers(self) -> nn.Sequential:  # TODO Klaus(?): Add different kinds of layers, as we go along.
        """
        This method uses the information in model_config to build the model's layers.

        Returns:
            nn.ModuleList: Contains all the model's layers.
        """
        layers = nn.ModuleList()

        def _get_act_fn(name: str, **kwargs):
            return getattr(nn, name)(**kwargs)

        if "conv_layers" in self.model_config.keys():
            self.model_architecture = "CNN"
            conv_layers = nn.ModuleList()
            # Example for conv_layer_dict: {'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding': 0}
            for conv_layer_dict in self.model_config["conv_layers"]:
                conv_layers.append(nn.LazyConv2d(**conv_layer_dict))
                for activation_layer in self.model_config[
                    "activation_fct"]:  # Only one activation_fct, but written like this for conformity with other layers.
                    conv_layers.append(_get_act_fn(activation_layer["name"], **activation_layer["kwargs"]))
                for post_activation_layer in self.model_config["post_activation_fct"]:
                    conv_layers.append(_get_act_fn(post_activation_layer["name"], **post_activation_layer["kwargs"]))
            layers.extend(conv_layers)

        if "post_conv_fct" in self.model_config.keys():
            post_conv_layers = nn.ModuleList()
            for post_conv_layer in self.model_config["post_conv_fct"]:
                post_conv_layers.append(_get_act_fn(post_conv_layer["name"], **post_conv_layer["kwargs"]))
            layers.extend(post_conv_layers)

        if "linear_layers" in self.model_config.keys():
            linear_layers = nn.ModuleList([nn.Flatten()])
            for lin_layer_inx, linear_layer_dict in enumerate(self.model_config["linear_layers"]):
                if linear_layer_dict["out_features"] == "num_classes":
                    linear_layer_dict.update(out_features=self.model_config["number_of_classes"])
                linear_layers.append(nn.LazyLinear(**linear_layer_dict))
                if lin_layer_inx + 1 == len(self.model_config["linear_layers"]):
                    break
                for activation_layer in self.model_config[
                    "activation_fct"]:  # Only one activation_fct, but written like this for conformity with other layers.
                    linear_layers.append(_get_act_fn(activation_layer["name"], **activation_layer["kwargs"]))
            layers.extend(linear_layers)

        if "final_activation_fct" in self.model_config.keys():
            final_activation_layers = nn.ModuleList()
            for final_activation_layer in self.model_config["final_activation_fct"]:
                final_activation_layers.append(
                    _get_act_fn(final_activation_layer["name"], **final_activation_layer["kwargs"]))
            layers.extend(final_activation_layers)

        return nn.Sequential(*layers)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        Apply all the layers in self.layers to input_batch.

        Arguments:
            input_batch: torch.tensor of torch.Size([b, c, h, w])

        Returns: torch.tensor of torch.Size([b, number_of_classes])
        """
        return self.layers(input_batch)

    def configure_optimizers(self) -> Tuple[List, List]:
        """
        This method returns the desired optimizer and learning rate scheduler.
        They are set according to the respective parameters in self.model_config.
        """
        optimizer = getattr(torch.optim, self.model_config["optimizer"][0])(self.parameters(), lr=self.learning_rate,
                                                                            **self.model_config["optimizer"][1])
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.model_config["learning_rate_scheduler"][0])(optimizer, **
        self.model_config["learning_rate_scheduler"][1])
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        This will be called for each batch of training images.
        It logs and returns metrics of the training step.

        Arguments:
            batch: tuple of torch.tensor: (torch.Size([b, c, h, w]), torch.Size([b]))
                    Contains a batch of images, and a batch of labels.
            batch_idx: int: the index of the batch in the dataloader.

        Returns:
            A dictionary. It has to include "loss": loss. Other entries are
            optional, and will be used in training_epoch_end.

        Logs:
            "train_loss": loss
        """
        image_batch, label_batch = batch
        model_output = self(image_batch)
        loss = self.criterion(model_output, label_batch)
        self.log("train_loss", loss)
        self.epoch_metrics["train"].append(dict(loss=loss.detach()))
        return {"loss": loss, "train_loss": loss.detach()}

    def on_train_epoch_end(self) -> None:  # TODO Klaus add type of outputs
        """
        This will be called after one epoch of training steps.
        Arguments: # TODO Klaus What is the type and shape of training_step_outputs?
            outputs: Collection of the return values of all training_steps.
        Logs:
            The average value of every given metric during the previous training epoch.
        """
        logs = {}
        for metric in ["loss"]:
            logs[f"train_avg_{metric}"] = torch.stack([x[metric] for x in self.epoch_metrics["train"]]).mean()
        self.epoch_metrics["train"].clear()
        self.log_dict(logs)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        This will be called for each batch of validation images.
        It logs and returns metrics of the validation step.

        Arguments:
            batch: tuple of torch.tensor: (torch.Size([b, c, h, w]), torch.Size([b]))
                    Contains a batch of images, and a batch of labels.
            batch_idx: int: the index of the batch in the dataloader.

        Returns:
            Optional: A dictionary. Entries will be used in val_epoch_end.

        Logs:
            "val_loss": loss
            "val_accuracy": accuracy (ratio of correctly predicted images)
        """
        image_batch, label_batch = batch
        model_output = self(image_batch)
        val_loss = self.criterion(model_output, label_batch)
        val_accuracy = self.accuracy(model_output.argmax(dim=1), label_batch)
        self.log("val_loss", val_loss)
        self.log("val_accuracy", val_accuracy)
        self.epoch_metrics["validation"].append(dict(loss=val_loss, accuracy=val_accuracy))
        return {"val_loss": val_loss, "val_accuracy": val_accuracy}

    def on_validation_epoch_end(self) -> None:
        """
        This will be called after one epoch of validation steps.
        Logs:
            The average value of every given metric during the previous validation epoch.
        """
        logs = {}
        for metric in ["loss", "accuracy"]:
            logs[f"val_avg_{metric}"] = torch.stack([x[metric] for x in self.epoch_metrics["validation"]]).mean()
        self.epoch_metrics["validation"].clear()
        self.log_dict(logs)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        image_batch, label_batch = batch
        model_output = self(image_batch)
        loss = self.criterion(model_output, label_batch)
        test_accuracy = self.accuracy(model_output.argmax(dim=1), label_batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", test_accuracy)
        self.epoch_metrics["test"].append(dict(loss=loss, accuracy=test_accuracy))
        return {"test_loss": loss, "test_accuracy": test_accuracy}

    def on_test_epoch_end(self) -> None:
        """
        This will be called after one epoch of test steps, i.e. the whole test.
        Logs:
            The average value of every given metric during the previous validation epoch.
        """
        logs = {}
        for metric in ["loss", "accuracy"]:
            logs[f"test_{metric}"] = torch.stack([x[metric] for x in self.epoch_metrics["test"]]).mean()
        self.log_dict(logs)

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
                     dataloader_idx: int = 0) -> torch.Tensor:
        image_batch, label_batch = batch
        model_output = self(image_batch)
        return model_output.argmax(dim=1)
