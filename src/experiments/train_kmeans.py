import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.preprocessing import ArealData

class KMeansExperiment:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.data_helper = ArealData(
            batch_size=1, 
            img_size=tuple(self.config['data']['img_size']), 
            input_img_paths=[], 
            target_img_paths=[]
        )
        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=self.config['kmeans']['n_clusters'],
            random_state=self.config['kmeans']['random_state'],
            n_init=self.config['kmeans']['n_init']
        )

    def prepare_data(self):
        """Extracts and scales features from the dataset."""
        train_dir = self.config['data']['train_dir']
        img_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.tif')]
        
        print(f"--- Extracting features from {len(img_paths)} images ---")
        X_train = self.data_helper.extract_pixel_features(
            img_paths, 
            n_subsamples=self.config['kmeans']['batch_pixels']
        )
        return self.scaler.fit_transform(X_train)

    def run(self):
        """Main pipeline: Train and Visualize."""
        X_scaled = self.prepare_data()
        print("--- Training K-Means ---")
        self.model.fit(X_scaled)

        # Prediction on a sample for GitHub results
        self.visualize_sample()

    def visualize_sample(self):
        """Generates a comparison plot between RGB and Clusters."""
        train_dir = self.config['data']['train_dir']
        sample_path = os.path.join(train_dir, os.listdir(train_dir)[0])
        
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
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original RGB")
        plt.imshow(rgb_img[:,:,[2,1,0]]) # Show B4, B3, B2
        
        plt.subplot(1, 2, 2)
        plt.title(f"K-Means (k={self.config['kmeans']['n_clusters']})")
        plt.imshow(pred_map, cmap='terrain')
        
        out_path = os.path.join(self.config['data']['output_dir'], "kmeans_result.png")
        plt.savefig(out_path)
        print(f"--- Result saved to {out_path} ---")

if __name__ == "__main__":
    experiment = KMeansExperiment()
    experiment.run()