"""Module for pytorch models."""

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet model for image classification.

    This implementation of AlexNet uses five convolutional layers,
    followed by three fully connected layers. It takes an input
    of size (3, 224, 224), typical for RGB images, and outputs
    class scores for `num_classes` categories.

    Attributes:
        num_classes (int): The number of output classes for classification.
        features (nn.Sequential): A sequential container of convolutional and pooling layers.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer to reduce feature maps.
        classifier (nn.Sequential): A sequential container of fully connected layers for classification.

    Args:
        num_classes (int, optional): Number of output classes. Default is 10.
    """
    def __init__(self, num_classes : int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass of the AlexNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, 224, 224), 
                                where N is the batch size, and 3 represents RGB channels.

        Returns:
            torch.Tensor: The output logits tensor of shape (N, num_classes), 
                            where N is the batch size and num_classes is the number of categories.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits