"""
EuroSAT Convolutional Neural Network (CNN) Architecture.

This module defines a custom CNN designed for multi-spectral satellite 
image classification. It uses sequential convolutional blocks with 
Batch Normalization and Dropout to achieve high accuracy on the EuroSAT dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EuroSATCNN(nn.Module):
    """
    Custom CNN for land use classification.
    
    The architecture consists of three convolutional blocks followed by 
    fully connected layers. Each block reduces the spatial dimensions 
    by half while increasing the feature depth.

    Attributes:
        conv1 (nn.Sequential): First block (Input -> 32 filters, MaxPool 2x2).
        conv2 (nn.Sequential): Second block (32 -> 64 filters, MaxPool 2x2).
        conv3 (nn.Sequential): Third block (64 -> 128 filters, MaxPool 2x2).
        flatten (nn.Flatten): Flattens 3D feature maps to 1D vectors.
        fc1 (nn.Linear): Fully connected layer with 256 neurons.
        dropout (nn.Dropout): Regularization layer (30% probability).
        fc2 (nn.Linear): Output layer mapped to the number of classes.
    """
    def __init__(self, in_channels=4, n_classes=10):
        """
        Initializes the CNN layers.

        Args:
            in_channels (int): Number of input bands (e.g., 4 for RGB + NDVI).
            n_classes (int): Number of target land use categories.
        """
        super(EuroSATCNN, self). __init__()
        
        # Bloc 1 : (in_channels x 64 x 64) -> (32 x 32 x 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Bloc 2 : (32 x 32 x 32) -> (64 x 16 x 16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Bloc 3 : (64 x 16 x 16) -> (128 x 8 x 8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Fully connected layers
        self.flatten = nn.Flatten() # Flattened size: 128 filters * 8 pixels * 8 pixels = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of satellite images.

        Returns:
            torch.Tensor: Logits for each class.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x