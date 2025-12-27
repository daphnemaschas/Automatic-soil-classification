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
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, F1Score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import yaml
from tqdm import tqdm
import numpy as np

from src.models.unet import UNet
from src.data_loaders.segmentation_ds import SpaceNet8Dataset


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
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_f1 = 0.0

        metric_args = {'task': 'multiclass', 'num_classes': self.n_classes, 'average': 'macro'}
        
        self.acc_metric = Accuracy(**metric_args).to(self.device)
        self.prec_metric = Precision(**metric_args).to(self.device)
        self.f1_metric = F1Score(**metric_args).to(self.device)

    def _load_paths(self, mode='train'):
        if mode == 'train':
            csv_path = self.config['segmentation']['data']['train_mapping']
            mask_dir = self.config['segmentation']['data']['train_mask']
        else:
            csv_path = self.config['segmentation']['data']['test_mapping']
            mask_dir = self.config['segmentation']['data']['test_mask']

        df = pd.read_csv(csv_path)
        img_paths = df['preimg'].tolist()
        
        # On reconstruit le nom du masque : image.tif -> image_mask.png
        mask_paths = [
            os.path.join(mask_dir, os.path.basename(p).replace('.tif', '_mask.png')) 
            for p in img_paths
        ]
    
        return img_paths, mask_paths

    def prepare_data(self, mode='train'):
        """
        Parses the mapping CSV and splits data into training and validation sets.

        It reconstructs the paths for the rasterized masks generated during 
        preprocessing and creates PyTorch DataLoaders.

        Returns:
            tuple: (train_loader, val_loader) instances.
        """
        img_paths, mask_paths = self._load_paths(mode=mode)
        
        if mode == 'train':
            # Internal split for validation (Louisiana-East)
            t_img, v_img, t_mask, v_mask = train_test_split(
                img_paths, mask_paths, test_size=0.2, random_state=42
            )
            train_loader = DataLoader(
                SpaceNet8Dataset(t_img, t_mask), 
                batch_size=self.config['segmentation']['model']['batch_size'], 
                shuffle=True, num_workers=4
            )
            val_loader = DataLoader(
                SpaceNet8Dataset(v_img, v_mask), 
                batch_size=self.config['segmentation']['model']['batch_size'], 
                shuffle=False, num_workers=4
            )
            return train_loader, val_loader
        else:
            # Full loader for Louisiana-West
            return DataLoader(SpaceNet8Dataset(img_paths, mask_paths), batch_size=1)

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
            masks = masks.to(self.device).long()
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Hybrid Loss
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(loader)

    def validate(self, loader, mode="Validation"):
        """
        Evaluates the model on the validation set.

        Args:
            loader (DataLoader): The validation data provider.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0

        self.acc_metric.reset()
        self.prec_metric.reset()
        self.f1_metric.reset()

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device).long()

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                self.acc_metric.update(preds, masks)
                self.prec_metric.update(preds, masks)
                self.f1_metric.update(preds, masks)
        
        avg_loss = total_loss / len(loader)
        acc = self.acc_metric.compute()
        prec = self.prec_metric.compute()
        f1 = self.f1_metric.compute()

        print(f"\n[{mode}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | F1: {f1:.4f}")
        return avg_loss, f1
    
    def test(self):
        """
        Final evaluation on unseen Louisiana-West data.
        """
        print("\n--- Final Test Evaluation (Louisiana-West) ---")
        path = self.config['segmentation']['model']['model_path']
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Loaded best model from {path}")
        
        test_loader = self.prepare_data(mode='test')
        loss, f1 = self.validate(test_loader, mode="TEST")
        return loss, f1

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
            val_loss, val_f1 = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(self.model.state_dict(), self.config['segmentation']['model']['model_path'])
                print(f"Best Model Saved (Loss: {val_loss:.4f}, F1: {val_f1:.4f})")

if __name__ == "__main__":
    experiment = SegmentationExperiment()
    experiment.run()