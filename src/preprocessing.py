import os
import numpy as np
from tensorflow import keras
from tifffile import TiffFile

class ArealdData(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_input_img_paths):
            x[j] = self.load_tiff(path)

        for j, path in enumerate(batch_target_img_paths):
            y[j] = np.expand_dims(self.load_tiff(path), -1)

        return x, y

    @staticmethod
    def load_tiff(filename):
        """Load a .tif file and return numpy array."""
        with TiffFile(filename) as tif:
            return tif.asarray()