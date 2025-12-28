"""
SpaceNet 8 Segmentation Dataset Module.

This module provides a specialized PyTorch Dataset class for handling 
satellite imagery and corresponding segmentation masks from the SpaceNet 8 dataset.
It utilizes rasterio for efficient geospatial data reading.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import rasterio
import numpy as np

class SpaceNet8Dataset(Dataset):
    """
    Dataset class for SpaceNet 8 binary or multi-class segmentation.
    
    This class loads satellite images (TIFF) and their corresponding 
    rasterized masks, preparing them for training with a U-Net or 
    similar segmentation architectures.

    Attributes:
        img_paths (list): List of file paths to the satellite images.
        mask_paths (list): List of file paths to the generated segmentation masks.
        transform (callable, optional): A function/transform (e.g., Albumentations) 
            that takes in an image and a mask and returns the transformed versions.
    """
    def __init__(self, img_paths, mask_paths, transform=None, target_size=(256, 256)): # (256, 256) to reduce the time spend treating each image
        """
        Initializes the dataset with image and mask file paths.

        Args:
            img_paths (list): Paths to the input .tif images.
            mask_paths (list): Paths to the corresponding .png or .tif masks.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        Fetches the image-mask pair at the given index.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: (image_tensor, mask_tensor) where image is scaled to [0, 1] 
                and mask contains class indices.
        """
        # Read the satellite image
        with rasterio.open(self.img_paths[index]) as src:
            img = src.read([1, 2, 3]).astype(np.float32) / 255.0
        
        # Read the mask
        mask = np.array(Image.open(self.mask_paths[index])).astype(np.int64)

        img_tensor = torch.from_numpy(img) # (3, H, W)
        mask_tensor = torch.from_numpy(mask) # (H, W)

        # Resize
        img_tensor = TF.resize(img_tensor, self.target_size, antialias=True)
        
        mask_tensor = TF.resize(mask_tensor.unsqueeze(0), self.target_size, 
                                interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        if self.transform:
            augmented = self.transform(image=img_tensor, mask=mask_tensor)
            img_tensor, mask_tensor = augmented['image'], augmented['mask']

        return img_tensor, mask_tensor