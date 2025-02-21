from lightning.pytorch.cli import LightningCLI
from src.pyclassify.model import AlexNet
from src.pyclassify.module import Classifier
from src.pyclassify.datamodule import CIFAR10DataModule

cli = LightningCLI(subclass_mode_data=True, subclass_mode_model=True)