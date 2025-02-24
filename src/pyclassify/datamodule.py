"""Module for PyTorch Lightning DataModule."""

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning.pytorch as pl

class CIFAR10DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for CIFAR-10 dataset.

    This class handles data preparation, including downloading the CIFAR-10 dataset,
    applying transformations, splitting the dataset into training, validation, and testing sets,
    and providing data loaders for each set.

    Attributes:
        data_path (str): The directory where the dataset will be downloaded.
        batch_size (int): The batch size for data loaders.
        transform (torchvision.transforms.Compose): The set of transformations to apply to the dataset.
        train (torch.utils.data.Dataset): The training dataset after splitting.
        valid (torch.utils.data.Dataset): The validation dataset after splitting.
        test (torch.utils.data.Dataset): The testing dataset.

    Args:
        data_path (str, optional): The directory where the dataset is stored. Default is './data'.
        batch_size (int, optional): The batch size for data loaders. Default is 64.
    """
    def __init__(self, data_path='./data', batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.Resize((70, 70)), transforms.RandomCrop((64, 64)),
                transforms.ToTensor()])

    def prepare_data(self):
        """
        Downloads the CIFAR-10 dataset and sets up the data transformations.

        This method downloads the CIFAR-10 dataset if it doesn't exist in `data_path`
        and applies the following transformations:
        - Resize the images to (70, 70)
        - Randomly crop them to (64, 64)
        - Convert them to PyTorch tensors
        """
        datasets.CIFAR10(root=self.data_path, download=True)

    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing.

        This method splits the training dataset into training and validation subsets,
        and loads the testing dataset. The datasets use the transformations defined in `prepare_data`.

        Args:
            stage (str, optional): The current stage ('fit' or 'test'). Default is None.
        """
        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False,
        )
        # Split the training dataset into training (45,000) and validation (5,000)
        self.train, self.valid = random_split(train, lengths=[45000, 5000])
        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False,
        )

    def train_dataloader(self):
        """
        Returns the data loader for the training set.

        This method returns a DataLoader for the training set, with shuffling enabled
        and drop_last set to True to drop the last incomplete batch.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        """
        Returns the data loader for the validation set.

        This method returns a DataLoader for the validation set, with shuffling disabled
        and drop_last set to False to keep all batches.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        return valid_loader

    def test_dataloader(self):
        """
        Returns the data loader for the testing set.

        This method returns a DataLoader for the testing set, with shuffling disabled
        and drop_last set to False to keep all batches.

        Returns:
            DataLoader: DataLoader for the testing dataset.
        """
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        return test_loader
