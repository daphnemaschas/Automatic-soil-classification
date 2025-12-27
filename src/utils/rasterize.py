"""
SpaceNet 8 Rasterization Utility Module.

This module provides tools to convert vector-based geospatial data (GeoJSON) 
into raster masks (PNG) aligned with specific satellite imagery. This is 
a crucial preprocessing step for semantic segmentation tasks where pixel-level 
labels are required.
"""
import imageio
import numpy as np
import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from tqdm import tqdm


def rasterize_spacenet8(csv_path, output_mask_dir):
    """
    Converts SpaceNet 8 GeoJSON vector labels into rasterized PNG masks.

    This function iterates through a mapping CSV, opens the corresponding 
    TIFF images to retrieve geospatial metadata (coordinate reference system 
    and transform), and projects the vector polygons onto a pixel grid.

    Args:
        csv_path (str): Path to the mapping CSV containing 'preimg' (image paths) 
            and 'label' (GeoJSON paths).
        output_mask_dir (str): Directory where the generated PNG masks will be saved.

    Returns:
        None: Saves generated masks directly to the specified directory.
    """
    df = pd.read_csv(csv_path)
    os.makedirs(output_mask_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rasterizing"):
        img_path = row['preimg']
        geojson_path = row['label']

        mask_name = os.path.basename(img_path).replace('.tif', '_mask.png')
        mask_path = os.path.join(output_mask_dir, mask_name)

        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            out_shape = (src.height, src.width)
            transform = src.transform

            # Load vector labels
            try:
                gdf = gpd.read_file(geojson_path)
                if gdf.empty:
                    mask = np.zeros(out_shape, dtype=np.uint8)
                else:
                    # Burn the geometry into a mask
                    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['class_id']))
                    mask = features.rasterize(shapes, out_shape=out_shape, transform=transform)
            except Exception as e:
                print(f"Erreur sur {geojson_path}: {e}")
                mask = np.zeros(out_shape, dtype=np.uint8)

            imageio.imwrite(mask_path, mask)
    
    print(f"Success: Masks saved in {output_mask_dir}.")