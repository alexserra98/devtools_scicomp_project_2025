import lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .module import Classifier
from torch.utils.data import random_split

WHERE_TO_SAVE = "data"

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_path=WHERE_TO_SAVE, batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, download=True)
        self.transform = transforms.Compose(
            [transforms.Resize((70, 70)), transforms.RandomCrop((64, 64)),
                transforms.ToTensor()])
    def setup(self, stage=None):
        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False,
        )
        self.train, self.valid = random_split(train, lengths=[45000, 5000])
        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False,
        )
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
                                            