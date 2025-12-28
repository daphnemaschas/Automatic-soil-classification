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


def rasterize_spacenet8(csv_path, output_mask_dir, img_dir, annotations_dir):
    """
    Converts SpaceNet 8 GeoJSON vector labels into rasterized PNG masks.

    This function iterates through a mapping CSV, opens the corresponding 
    TIFF images to retrieve geospatial metadata (coordinate reference system 
    and transform), and projects the vector polygons onto a pixel grid.

    Args:
        csv_path (str): Path to the mapping CSV containing 'pre-event image' (image paths) 
            and 'label' (GeoJSON paths).
        output_mask_dir (str): Directory where the generated PNG masks will be saved.
        img_dir (str): Directory where .tif images are stored.
        annotations_dir (str): Directory where the annotations (geojson) are stored.

    Returns:
        None: Saves generated masks directly to the specified directory.
    """
    df = pd.read_csv(csv_path)
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rasterizing"):
        img_path = os.path.join(img_dir, row['pre-event image'])
        geojson_path = os.path.join(annotations_dir,row['label'])

        mask_name = os.path.basename(img_path).replace('.tif', '_mask.png')
        mask_path = os.path.join(output_mask_dir, mask_name)

        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            out_shape = (src.height, src.width)
            transform = src.transform
            img_crs = src.crs

            # Load vector labels
            try:
                gdf = gpd.read_file(geojson_path)
                if gdf.empty:
                    mask = np.zeros(out_shape, dtype=np.uint8)
                else:
                    gdf = gdf.to_crs(img_crs)
                    shapes = []

                    for _, feature in gdf.iterrows():
                        is_flooded = feature.get('flooded') is not None and feature.get('flooded') != ""

                        if not feature.geometry.is_valid:
                            print("Geometry not valid")
                            continue
                        
                        if feature.get('building') == 'yes':
                            val = 2 if is_flooded else 1 # 1:build, 2:flood_build
                            shapes.append((feature.geometry, val))

                        elif feature['highway'] is not None:
                            val = 4 if is_flooded else 3 # 3:road, 4:flood_road
                            shapes.append((feature.geometry.buffer(0.00005), val)) # To make highways visible
                        
                if shapes:
                    mask = features.rasterize(
                        shapes, 
                        out_shape=out_shape, 
                        transform=transform, 
                        fill=0, 
                        all_touched=True
                    )
                else:
                    mask = np.zeros(out_shape, dtype=np.uint8)
                
            except Exception as e:
                print(f"Erreur sur {geojson_path}: {e}")
                mask = np.zeros(out_shape, dtype=np.uint8)

            imageio.imwrite(mask_path, mask.astype(np.uint8))
    
    print(f"Success: Masks saved in {output_mask_dir}.")