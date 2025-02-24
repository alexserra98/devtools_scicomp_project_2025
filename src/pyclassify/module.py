"""Module for lightning modules."""

import torch
import torch.nn as nn
import torchmetrics
import lightning.pytorch as pl


class Classifier(pl.LightningModule):
    """
    PyTorch Lightning model wrapper for training, validation, and testing.

    This class implements the logic for training, validating, and testing a given
    neural network model using PyTorch Lightning's standardized training loop. It tracks
    accuracy for each phase and uses cross-entropy loss for classification tasks.

    Attributes:
        model (nn.Module): The neural network model to be trained and evaluated.
        train_accuracy (torchmetrics.Accuracy): Accuracy metric for tracking performance during training.
        val_accuracy (torchmetrics.Accuracy): Accuracy metric for tracking performance during validation.
        test_accuracy (torchmetrics.Accuracy): Accuracy metric for tracking performance during testing.

    Args:
        model (nn.Module): The PyTorch model to be trained and evaluated.
    """
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        # Accuracy metrics for training, validation, and testing
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.model.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.model.num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.model.num_classes)

    def _classifier_step(self, batch):
        """
        Shared logic for training, validation, and testing steps.

        This method computes the model's predictions (logits), calculates the cross-entropy loss,
        and determines the predicted labels.

        Args:
            batch (tuple): A tuple containing input features and true labels.

        Returns:
            tuple: A tuple containing the computed loss, the true labels, and the predicted labels.
        """
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, _):
        """
        Performs a single training step.

        This method uses the `_classifier_step` function to calculate the loss
        and update the training accuracy metric. It logs the loss and accuracy.

        Args:
            batch (tuple): A tuple containing input features and true labels.

        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        loss, true_labels, predicted_labels = self._classifier_step(batch)
        self.train_accuracy(predicted_labels, true_labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        """
        Performs a single validation step.

        This method uses the `_classifier_step` function to calculate the loss
        and update the validation accuracy metric. It logs the loss and accuracy.

        Args:
            batch (tuple): A tuple containing input features and true labels.
        """
        loss, true_labels, predicted_labels = self._classifier_step(batch)
        self.val_accuracy(predicted_labels, true_labels)
        self.log("val_loss", loss)
        self.log(
            "val_acc",
            self.val_accuracy,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, _):
        """
        Performs a single test step.

        This method uses the `_classifier_step` function to calculate the accuracy
        during the test phase and logs the accuracy metric.

        Args:
            batch (tuple): A tuple containing input features and true labels.
        """
        _, true_labels, predicted_labels = self._classifier_step(batch)
        self.test_accuracy(predicted_labels, true_labels)
        self.log("test_acc", self.test_accuracy, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        """
        Configures the optimizer used for training.

        Returns:
            torch.optim.Optimizer: The Adam optimizer with a learning rate of 0.0001.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer