# Earth Observer – Land Cover Classification with Sentinel-2

## Project Overview

This project explores the use of machine learning in satellite imagery analysis.
Using data from the Sentinel-2 mission, we analyze land cover to:

* Detect different types of vegetation cover
* Identify areas affected by fire
* Assess zones at potential risk

The models tested include both classical machine learning algorithms and deep learning architectures:

* K-Means (unsupervised clustering)
* Support Vector Machines (SVM)
* Decision Trees
* Convolutional Neural Networks (CNN)
* U-Net (semantic segmentation for satellite images)

---

## Dataset

* **Source**: Sentinel-2 imagery (ESA Copernicus mission)
* **Format**: `.tif` raster images (multi-spectral)
* **Labels**: Land cover classes such as:

  * Artificial surfaces and construction
  * Cultivated areas
  * Broadleaf forest
  * Coniferous forest
  * Herbaceous vegetation
  * Natural material surfaces
  * Permanent snow
  * Water bodies

Example visualization (RGB composite and NDVI):

```
results/example_rgb.png
results/example_ndvi.png
```

---

## Spectral Indices Used

To highlight vegetation and water conditions, we compute common indices:

* **NDVI** – Normalized Difference Vegetation Index
  [
  NDVI = \frac{ρ_{NIR} - ρ_{R}}{ρ_{NIR} + ρ_{R}}
  ]
* **MSI** – Moisture Stress Index
* **NDWI** – Normalized Difference Water Index
* **NDSI** – Normalized Difference Snow Index

---

## Project Structure

```
earth-observer/
│── data/                # Raw and processed datasets (not pushed to repo)
│── notebooks/           # Exploratory notebooks (EDA, tests, visualizations)
│── src/                 # Reusable source code
│   ├── preprocessing.py # Data loading, normalization, index calculation
│   ├── models.py        # ML and DL model definitions (KMeans, SVM, CNN, U-Net…)
│   ├── training.py      # Training pipelines and evaluation
│   ├── utils.py         # Helper functions (plots, metrics, etc.)
│── experiments/         # Scripts to run training with different models
│── results/             # Saved models, metrics, plots
│── requirements.txt     # Dependencies
│── README.md            # Project description
│── .gitignore           # Ignore datasets, cache, etc.
```

---

## Models Implemented

| Model             | Task                       | Advantages                          | Limitations                 |
| ----------------- | -------------------------- | ----------------------------------- | --------------------------- |
| **K-Means**       | Unsupervised clustering    | Simple, fast                        | No labels, rough boundaries |
| **SVM**           | Pixel-level classification | Works well on small data            | Not scalable                |
| **Decision Tree** | Classification             | Interpretable                       | Overfitting                 |
| **CNN**           | Supervised classification  | Good accuracy                       | Requires larger datasets    |
| **U-Net**         | Semantic segmentation      | State-of-the-art for remote sensing | Computationally heavy       |

---

## Results

* Accuracy metrics per model
* Confusion matrices
* Example prediction maps (CNN / U-Net)

```
results/predicted_map.png
```

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/earth-observer.git
cd earth-observer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run experiments

Train a CNN:

```bash
python experiments/train_cnn.py
```

Run U-Net segmentation:

```bash
python experiments/train_unet.py
```

---

## Future Work

* Improve generalization with data augmentation
* Use time-series Sentinel-2 images to detect fire spread over time
* Deploy a simple web application to visualize predictions

---

## Authors

Project carried out as part of **EI – Observer la Terre (Preligens partnership)**.