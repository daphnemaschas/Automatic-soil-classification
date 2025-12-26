import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tifffile import TiffFile

class ArealdData(Dataset):
    """PyTorch Dataset to iterate over satellite imagery."""
    def __init__(self, csv_file, root_dir, n_channels=4):
        """
        Args:
            csv_file: Path to the training CSV.
            root_dir: Directory where the class folders (Forest/, River/, etc.) are.
            n_channels: 4 for RGB + NDVI.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.n_channels = n_channels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get path and label
        img_rel_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        img_path = os.path.join(self.root_dir, img_rel_path)

        # Load and preprocess input
        raw_img = self.load_tiff(img_path)
        normalized_img = self.preprocess_image(raw_img)
        idx_map = self.calculate_spectral_indices(normalized_img)

        # Stack channels: RGB (B4, B3, B2) + NDVI
        x = np.stack([
            normalized_img[:,:,2], # Red
            normalized_img[:,:,1], # Green
            normalized_img[:,:,0], # Blue
            idx_map['ndvi']        # NDVI
        ], axis=0)

        # In this new implementation of the project we are doing solely classification and not segmentation. Hence we won't use masks anymore
        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.long)
    
    @staticmethod
    def load_tiff(filename):
        """Load a .tif file and return numpy array."""
        with TiffFile(filename) as tif:
            img = tif.asarray().astype("float32")
            if img.shape[0] < img.shape[1]:
                img = np.transpose(img, (1, 2, 0))
            return img # (H, W, Bands)
        
    def preprocess_image(self, img):
        """Standardize Sentinel-2 reflectance values."""
        img = img.astype("float32") / 10000.0 # TO VERIFY, though Sentinel-2 L2A data is usually 0-10000
        return np.clip(img, 0, 1)

    @staticmethod
    def calculate_spectral_indices(img):
        """
        Calculates various spectral indices for Sentinel-2 data.
        Sentinel-2 Band Mapping (0-indexed):
        B2 (Blue): 0 | B3 (Green): 1 | B4 (Red): 2 | B8 (NIR): 3 | B11 (SWIR1): 4 | B12 (SWIR2): 5
        Args:
            - img: (height, width, bands)
        """
        eps = 1e-8  # To avoid division by 0
        
        b2 = img[:, :, 0] # Blue
        b4 = img[:, :, 2] # Red
        b8 = img[:, :, 3] # NIR
        b11 = img[:, :, 4] # SWIR1
        
        indices = {}

        # 1. NDVI (Vegetation)
        indices['ndvi'] = (b8 - b4) / (b8 + b4 + eps)

        # 2. NDBI (Built-up / Urban)
        indices['ndbi'] = (b11 - b4) / (b11 + b4 + eps)

        # 3. NDMI (Moisture/Drought)
        indices['ndmi'] = (b8 - b11) / (b8 + b11 + eps)

        # 4. BAI (Burned Area Index)
        # Highlights burned areas based on reflectance decrease in NIR and increase in Red
        indices['bai'] = 1.0 / ((0.1 - b4)**2 + (0.06 - b8)**2 + eps)

        # 5. BSI (Bare Soil Index)
        indices['bsi'] = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)

        # 6. SWIR (Direct Heat/Fire detection), usually visualized directly using Band 12
        indices['swir_heat'] = img[:, :, 5] 

        return indices
    
    def extract_pixel_features(self, image_paths, n_subsamples=5000):
        """
        Extracts pixels from multiple images to create a training set for K-Means/SVM.
        Args:
            - n_subsamples: number of pixels to take per image to avoid memory crashes.
        """
        all_features = []
        
        for path in image_paths:
            img = self.preprocess_image(self.load_tiff(path))
            indices = self.calculate_spectral_indices(img)
            # Create a feature stack: RGB + NIR + NDVI + NDBI
            feature_stack = np.stack([
                img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3], # B2, B3, B4, B8
                indices['ndvi'], 
                indices['ndbi']
            ], axis=-1) # Shape: (H, W, 6)
            flat_pixels = feature_stack.reshape(-1, 6) # (H*W, 6)
            
            # Subsample to keep the dataset manageable
            n_pix = min(flat_pixels.shape[0], n_subsamples)
            indices_resample = np.random.choice(flat_pixels.shape[0], n_pix, replace=False)
            all_features.append(flat_pixels[indices_resample])
            
        return np.vstack(all_features)