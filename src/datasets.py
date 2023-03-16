from typing import Callable

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from src import DATASETS


class GeneralDataModule(pl.LightningDataModule):
    """
    This class implements the general functionality of our data modules.
    It is subclassed by the classes for the specific datasets.

    Attributes:
        dataset_name: str: The name of the torchvision dataset class to be downloaded.
                E.g. "MNIST" in order to download torchvision.dataset.MNIST.
        val_set_size: int: The number of images in the validation set. The training
                set will contain the rest of the images downloaded from the training data.
        number_of_classes: number of classes in the dataset
        number_of_channels: number of channels in images in the dataset
                Default is 3!
        dl_kwargs: dict: The optional arguments given to the DataLoader initialisation.
                Do not include the "shuffle" argument, because it is different
                for the different dataloaders.
        transform: This transform will be performed on the images before they are used,
                and after they are saved locally.
        target_transform: This transform will be performed on the labels
                before they are used, and after they are saved locally.
    """

    def __init__(self, dataset_name: str, val_set_size: int, number_of_classes: int, number_of_channels: int = 3,
                 dl_kwargs: dict = None, transform: Callable = None,
                 target_transform: Callable = None, train_flag_is_bool: bool = True, limit_size: bool = False):
        super().__init__()
        self.dataset_name = dataset_name
        self.number_of_classes = number_of_classes
        self.number_of_channels = number_of_channels
        self.val_set_size = val_set_size
        self.dl_kwargs = dict(batch_size=512, num_workers=12)
        if dl_kwargs is not None:
            self.dl_kwargs.update(dl_kwargs)
        self.transform = transform
        self.target_transform = target_transform
        # The datasets will be assigned to in setup().
        self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset = None, None, None, None

        def datasetclass_kwargs_wrapper(*args, **kwargs):
            TorchvisionDataClass = getattr(torchvision.datasets, self.dataset_name)
            if train_flag_is_bool:
                return TorchvisionDataClass(*args, **kwargs)
            if "train" in kwargs:
                kwargs["split"] = "train" if kwargs["train"] else "test"
                del kwargs["train"]
                return TorchvisionDataClass(*args, **kwargs)

        self.dsc_kw_wrapper = datasetclass_kwargs_wrapper
        self.limit_size = limit_size

    def prepare_data(self):
        """
        This method is called before the data is used for training or testing.
        It downloads the datasets from torchvision onto the local machine.
        """
        # This returns, e.g., the class torchvision.datasets.MNIST, if dataset_name is "MNIST".
        self.dsc_kw_wrapper(root=DATASETS, train=True, download=True)
        self.dsc_kw_wrapper(root=DATASETS, train=False, download=True)

    def setup(self, stage: str = None):
        """
        This method is called before every training process, or test process.
        It loads the datasets downloaded in prepare_data(), splits the training
        data into training and validation sets, and performs transforms on
        the data.
        """
        # This returns, e.g., the class torchvision.datasets.MNIST, if dataset_name is "MNIST".

        # stage=="fit", "test", "predict": If the DataModule is used by pl.trainer
        # stage==None: If it is used outside of the Lightning infrastructure
        if stage == "fit" or stage is None:
            train_val_dataset = self.dsc_kw_wrapper(root=DATASETS, train=True,
                                                    download=True, transform=self.transform,
                                                    target_transform=self.target_transform)
            # Split the train_val_dataset into train and val datasets.
            gen = torch.Generator()
            gen.manual_seed(0xDEADBEEF // 1000)
            if not self.limit_size:
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_val_dataset,
                                                                                     [
                                                                                         len(train_val_dataset) - self.val_set_size,
                                                                                         self.val_set_size],
                                                                                     generator=gen)
            else:
                lens = [20000, self.val_set_size]
                lens.append(len(train_val_dataset) - sum(lens))
                self.train_dataset, self.val_dataset, _ = torch.utils.data.random_split(train_val_dataset, lens,
                                                                                        generator=gen)

        if stage == "test" or stage is None:
            self.test_dataset = self.dsc_kw_wrapper(root=DATASETS, train=False, download=True,
                                                    transform=self.transform, target_transform=self.target_transform)

        if stage == "predict" or stage is None:
            self.predict_dataset = self.dsc_kw_wrapper(root=DATASETS, train=False, download=True,
                                                       transform=self.transform,
                                                       target_transform=self.target_transform)

    # The following four methods return DataLoaders from the datasets
    # assigned in setup().
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False, **self.dl_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=False, **self.dl_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset, shuffle=False, **self.dl_kwargs)

    def get_test_dataset(self) -> Dataset:
        self.setup(stage="test")
        return self.test_dataset


# Now, use GeneralDataModule as a superclass to create a class each for the different datasets.
class MNISTDataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("MNIST", val_size, number_of_classes=10, number_of_channels=1, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]), target_transform=None, **kwargs)


# Small MNIST data module for fast testing of code.
class SmallMNISTDataModule(MNISTDataModule):
    def __init__(self, **kwargs):
        super().__init__(val_size=10, **kwargs)  # Initialize MnistDataModule

    def setup(self, stage: str = None):
        super().setup(stage)
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices=range(50))
        # self.val_dataset is already small from initialisation.
        self.test_dataset = torch.utils.data.Subset(self.test_dataset, indices=range(10))
        self.predict_dataset = torch.utils.data.Subset(self.predict_dataset, indices=range(50))


class FashionMNISTDataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("FashionMNIST", val_size, number_of_classes=10, number_of_channels=1,
                         transform=transforms.Compose(
                             [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]), target_transform=None,
                         **kwargs)


class CIFAR10DataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("CIFAR10", val_size, number_of_classes=10, number_of_channels=3, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                         target_transform=None, **kwargs)


class CIFAR100DataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("CIFAR100", val_size, number_of_classes=100, number_of_channels=3,
                         transform=transforms.Compose(
                             [transforms.ToTensor(),
                              transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
                         target_transform=None, **kwargs)


class USPSDataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("USPS", val_size, number_of_classes=10,
                         transform=transforms.ToTensor(),
                         target_transform=None, **kwargs)


class USPS28DataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("USPS", val_size, number_of_classes=10,
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))]),
                         target_transform=None, **kwargs)


class KMNISTDataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("KMNIST", val_size, number_of_classes=10, number_of_channels=1, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]), target_transform=None, **kwargs)


class Food101DataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("Food101", val_size, limit_size=True, number_of_classes=101, number_of_channels=3,
                         transform=transforms.Compose(
                             [transforms.ToTensor(), transforms.Resize((64, 64)),
                              transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))]),
                         target_transform=None, train_flag_is_bool=False, **kwargs)


class STL10DataModule(GeneralDataModule):
    def __init__(self, val_size=500, **kwargs):
        super().__init__("STL10", val_size, number_of_classes=10, number_of_channels=3,
                         transform=transforms.Compose(
                             [transforms.ToTensor(), transforms.Resize((48, 48)),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]),
                         target_transform=None, train_flag_is_bool=False, **kwargs)


class SVHNDataModule(GeneralDataModule):
    def __init__(self, val_size=5000, **kwargs):
        super().__init__("SVHN", val_size, number_of_classes=10, number_of_channels=3, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.43768, 0.44376, 0.47280), (0.19803, 0.20101, 0.19703))]),
                         target_transform=None, train_flag_is_bool=False, **kwargs)


def get_data_module(dataset_name: str, **data_module_kwargs) -> GeneralDataModule:
    """
    This function returns the class corresponding to the dataset_name.
    It is case-sensitive.

    Arguments:
        dataset_name: str: The name of the dataset whose DataModule should be
                    returned. This is case-sensitive.

    Returns:
        A DataModule object
    """
    data_module_name = dataset_name + "DataModule"
    assert data_module_name in globals(), \
        f"DataModule {data_module_name} did not exist, check you dataset name spelling!"
    return globals()[data_module_name](**data_module_kwargs)
