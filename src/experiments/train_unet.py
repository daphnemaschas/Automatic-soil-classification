"""
Training Script for SpaceNet 8 Segmentation.

This script orchestrates the training pipeline for the U-Net model:
1. Data loading and splitting (Image/Mask pairs).
2. Loss function definition (BCE + Dice Loss).
3. Training and validation loops with performance monitoring.
4. Model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, train_test_split
import pandas as pd
import os
import yaml
from tqdm import tqdm
import numpy as np

from src.models.unet import UNet
from src.data_loaders.segmentation_ds import SpaceNet8Dataset

class DiceLoss(nn.Module):
    """
    Computes the Dice Loss for binary segmentation tasks.
    
    Dice Loss is derived from the Sørensen–Dice coefficient and is 
    highly effective at handling class imbalance by maximizing the 
    spatial overlap between the prediction and the ground truth.

    Attributes:
        smooth (float): A small constant added to the numerator and 
            denominator to prevent division by zero and stabilize training.
    """
    def __init__(self, smooth=1.0):
        """
        Initializes the DiceLoss module.

        Args:
            smooth (float): Smoothing factor for numerical stability.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Calculates the Dice Loss between predictions and targets.

        Args:
            preds (torch.Tensor): Model output logits or probabilities. 
                Shape: (Batch, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks. 
                Shape: (Batch, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

class SegmentationExperiment:
    """
    Orchestrator for U-Net training and validation on SpaceNet 8 data.

    This class encapsulates the hardware setup, data preparation using 
    image-to-mask mapping, and the execution of the training loop using 
    a combination of Binary Cross-Entropy and Dice loss.

    Attributes:
        config (dict): Configuration parameters loaded from a YAML file.
        device (torch.device): Hardware accelerator (CUDA or CPU).
        model (nn.Module): The U-Net segmentation model.
        optimizer (torch.optim.Optimizer): Adam optimizer for weight updates.
        bce_loss (nn.Module): Standard Binary Cross Entropy with Logits.
        dice_loss (nn.Module): Custom Dice Loss for handling class imbalance.
        best_loss (float): Tracker for the lowest validation loss achieved.
    """
    def __init__(self, config_path="config.yaml"):
        """
        Initializes the experiment components and hardware settings.

        Args:
            config_path (str): Path to the YAML file containing model and data paths.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize U-Net
        self.model = UNet(
            in_channels=self.config['segmentation']['model']['in_channels'],
            out_channels=self.config['segmentation']['model']['n_classes']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['segmentation']['model']['learning_rate']
        )
        
        # Combination of Losses for better building footprint extraction
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
        self.best_loss = float('inf')

    def prepare_data(self):
        """
        Parses the mapping CSV and splits data into training and validation sets.

        It reconstructs the paths for the rasterized masks generated during 
        preprocessing and creates PyTorch DataLoaders.

        Returns:
            tuple: (train_loader, val_loader) instances.
        """
        df = pd.read_csv(self.config['segmentation']['data']['mapping_csv'])
        
        # Reconstruct paths for generated masks
        mask_dir = self.config['segmentation']['data']['mask_dir']
        img_paths = df['preimg'].tolist()
        mask_paths = [
            os.path.join(mask_dir, os.path.basename(p).replace('.tif', '_mask.png')) 
            for p in img_paths
        ]
        
        # Split data
        train_img, val_img, train_mask, val_mask = train_test_split(
            img_paths, mask_paths, test_size=0.2, random_state=42
        )
        
        train_ds = SpaceNet8Dataset(train_img, train_mask)
        val_ds = SpaceNet8Dataset(val_img, val_mask)
        
        train_loader = DataLoader(
            train_ds, batch_size=self.config['segmentation']['model']['batch_size'], 
            shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config['segmentation']['model']['batch_size'], 
            shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader

    def train_epoch(self, loader, epoch):
        """
        Runs a single training epoch through the provided DataLoader.

        Args:
            loader (DataLoader): The training data provider.
            epoch (int): Current epoch number for display.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
        
        for images, masks in loop:
            images = images.to(self.device)
            # Add channel dimension to masks for BCE loss (B, 1, H, W)
            masks = masks.unsqueeze(1).float().to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Hybrid Loss
            loss = self.bce_loss(outputs, masks) + self.dice_loss(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(loader)

    def validate(self, loader):
        """
        Evaluates the model on the validation set.

        Args:
            loader (DataLoader): The validation data provider.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.unsqueeze(1).float().to(self.device)
                outputs = self.model(images)
                loss = self.bce_loss(outputs, masks) + self.dice_loss(outputs, masks)
                total_loss += loss.item()
        return total_loss / len(loader)

    def run(self):
        """
        Main execution logic for the experiment.
        
        Orchestrates the full training cycle, validation, and saves the 
        best performing model weights to disk.
        """
        print(f"--- Starting Segmentation Training on {self.device} ---")
        train_loader, val_loader = self.prepare_data()
        
        epochs = self.config['segmentation']['model']['epochs']
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.config['segmentation']['model']['model_path'])
                print(f"Best Model Saved (Loss: {val_loss:.4f})")

if __name__ == "__main__":
    experiment = SegmentationExperiment()
    experiment.run()