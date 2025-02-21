import torch
import torch.nn as nn
from .model import AlexNet
import torchmetrics
import lightning as pl

class Classifier(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = AlexNet(num_classes=10)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=100)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
    
    def _classifier_step(self, batch):
        features, true_labels = batch
        logits = self.model(features)
        loss = nn.CrossEntropyLoss()(logits, true_labels)
        return loss, logits, true_labels
    
    def training_step(self, batch, batch_idx):
        loss, logits, _ = self._classifier_step(batch)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, true_labels = self._classifier_step(batch)
        self.val_accuracy(logits, true_labels)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, logits, true_labels = self._classifier_step(batch)
        self.test_accuracy(logits, true_labels)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        return loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
        
