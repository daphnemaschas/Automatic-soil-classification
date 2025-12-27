"""
K-Means Clustering Experiment Module for Satellite Imagery.

This module implements an unsupervised learning approach to partition 
satellite image pixels into clusters based on spectral features (RGB, NIR) 
and calculated indices (NDVI, NDBI). It is useful for initial land cover 
exploration and segmenting raw satellite data.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.data_loaders.classification_ds import ArealData

class KMeansExperiment:
    """
    Orchestrates the K-Means clustering pipeline for remote sensing data.
    
    This class handles data preparation by subsampling pixels across multiple 
    images, scaling features, training the KMeans model, and visualizing 
    the resulting cluster maps.

    Attributes:
        config (dict): Configuration parameters from a YAML file.
        data_helper (ArealData): Instance of ArealData to handle TIFF loading and preprocessing.
        scaler (StandardScaler): Scikit-learn scaler to normalize spectral features.
        model (KMeans): The initialized K-Means clustering model.
    """
    def __init__(self, config_path="config.yaml"):
        """
        Initializes the experiment with configuration and scikit-learn components.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.data_helper = ArealData(
            csv_file=self.config['classification']['data']['train_dir'], 
            root_dir=self.config['classification']['data']['root_dir']
        )
        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=self.config['classification']['kmeans']['n_clusters'],
            random_state=self.config['classification']['kmeans']['random_state'],
            n_init=self.config['classification']['kmeans']['n_init']
        )

    def prepare_data(self):
        """
        Extracts and scales pixel-level features from a subset of the dataset.

        Iterates through the image repository, extracts spectral bands and indices,
        and prepares a flattened array suitable for unsupervised training.

        Returns:
            np.ndarray: Scaled feature matrix of shape (n_pixels, n_features).
        """
        root_dir = self.config['classification']['data']['root_dir']
        img_paths = glob.glob(os.path.join(root_dir, "**/*.tif"), recursive=True)

        n_samples = min(len(img_paths), 500) # We limit the nb of samples for Kmeans
        selected_paths = np.random.choice(img_paths, n_samples, replace=False)

        print(f"--- Extracting features from {len(img_paths)} images ---")
        X_train = self.data_helper.extract_pixel_features(
            selected_paths, 
            n_subsamples=self.config['classification']['kmeans']['batch_pixels']
        )
        return self.scaler.fit_transform(X_train)

    def run(self):
        """
        Executes the main pipeline: data preparation, model training, and visualization.
        """
        X_scaled = self.prepare_data()
        print("--- Training K-Means ---")
        self.model.fit(X_scaled)

        # Prediction on a sample for GitHub results
        self.visualize_sample()

    def visualize_sample(self):
        """
        Selects a random image and generates a side-by-side comparison of RGB vs Clusters.
        
        This method handles full image preprocessing, spectral index calculation, 
        and model prediction for every pixel in the selected image.
        """
        root_dir = self.config['classification']['data']['root_dir']
        img_paths = glob.glob(os.path.join(root_dir, "**/*.tif"), recursive=True)
        sample_path = np.random.choice(img_paths)
        
        # Process image
        raw_img = self.data_helper.load_tiff(sample_path)
        normalized_img = self.data_helper.preprocess_image(raw_img)
        indices = self.data_helper.calculate_spectral_indices(normalized_img)
        
        # Stack features
        feature_stack = np.stack([
            normalized_img[:,:,0], normalized_img[:,:,1], 
            normalized_img[:,:,2], normalized_img[:,:,3],
            indices['ndvi'], indices['ndbi']
        ], axis=-1)
        
        # Predict
        h, w, c = feature_stack.shape
        flat_img = feature_stack.reshape(-1, c)
        labels = self.model.predict(self.scaler.transform(flat_img))
        prediction_map = labels.reshape(h, w)

        # Plotting
        self._save_plot(normalized_img, prediction_map)

    def _save_plot(self, rgb_img, pred_map):
        """
        Plots and saves the comparison figure to the output directory.

        Args:
            rgb_img (np.ndarray): Normalized image for RGB visualization.
            pred_map (np.ndarray): 2D array of cluster labels.
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original RGB")
        plt.imshow(rgb_img[:,:,[2,1,0]]) # Show B4, B3, B2
        
        plt.subplot(1, 2, 2)
        plt.title(f"K-Means (k={self.config['classification']['kmeans']['n_clusters']})")
        plt.imshow(pred_map, cmap='terrain')
        
        out_path = os.path.join(self.config['classification']['data']['output_dir'], "kmeans_result.png")
        plt.savefig(out_path)
        print(f"--- Result saved to {out_path} ---")

if __name__ == "__main__":
    experiment = KMeansExperiment()
    experiment.run()