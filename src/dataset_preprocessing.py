import os
import csv
import numpy as np
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer
from PIL import Image


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Computes Normalized Difference Vegetation Index (NDVI): (NIR - RED) / (NIR + RED).
    """
    red_copy = red.astype(float).copy()
    nir_copy = nir.astype(float).copy()

    # Prevent division by zero
    red_copy[red_copy == 0] = 1.0
    nir_copy[nir_copy == 0] = 1.0

    denom = nir_copy + red_copy
    denom[denom == 0] = 1e-6

    ndvi = (nir_copy - red_copy) / denom
    return ndvi


def transform_pixel_to_coordinates(row, col, src, orig_height, orig_width, height, width):
    """
    Converts raster pixel indices to geospatial Longitude and Latitude coordinates.
    """
    lon, lat = src.xy(
        row * (orig_height - 1) / max(height - 1, 1),
        col * (orig_width - 1) / max(width - 1, 1)
    )
    return lon, lat


def transform_coordinates_to_pixel(lat: float, lon: float, src) -> tuple:
    """
    Converts Longitude/Latitude (EPSG:4326) to raster row and col indices.
    """
    image_crs = src.crs
    latlon_crs = CRS.from_epsg(4326)

    if image_crs == latlon_crs:
        row, col = src.index(lon, lat)
    else:
        transformer = Transformer.from_crs(latlon_crs, image_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        row, col = src.index(x, y)

    return abs(row), abs(col)


def create_empty_csv_files(output_folder: str, header: dict):
    """
    Creates empty CSV files for each coordinate grid cell.
    """
    os.makedirs(output_folder, exist_ok=True)
    for row in range(header['Height']):
        for col in range(header['Width']):
            file_name = f"{row}_{col}.csv"
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header['Headers'])


def extract_pixel_values(image_path: str, output_folder: str, header: dict, width: int = None, height: int = None):
    """
    Extracts multi-spectral band values from a single satellite TIF image and writes to pixel CSV files.
    """
    with rasterio.open(image_path) as src:
        num_bands = src.count
        orig_height, orig_width = src.height, src.width

        w = width if width is not None else orig_width
        h = height if height is not None else orig_height

        for row in range(h):
            for col in range(w):
                lon, lat = transform_pixel_to_coordinates(row, col, src, orig_height, orig_width, h, w)
                pixel_info = {'X': row, 'Y': col, 'Longitude': lon, 'Latitude': lat}

                for band in range(1, num_bands + 1):
                    band_data = src.read(band)
                    pixel_info[f'Band_{band}'] = band_data[min(row, orig_height-1), min(col, orig_width-1)]

                file_name = f"{row}_{col}.csv"
                file_path = os.path.join(output_folder, file_name)
                write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

                with open(file_path, 'a', newline='') as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=header['Headers'])
                    if write_header:
                        writer.writeheader()
                    writer.writerow(pixel_info)


def process_images(image_folder: str, output_folder: str, width: int = None, height: int = None, progress_callback=None):
    """
    Processes all satellite .tif images in a folder and generates structured CSV files for each pixel time-series.
    """
    total_images = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.tif')]

    if not total_images:
        raise FileNotFoundError(f"No .tif images found in '{image_folder}'.")

    first_image_path = total_images[0]
    with rasterio.open(first_image_path) as src:
        num_bands = src.count
        orig_height, orig_width = src.height, src.width
        header = {
            'Height': height if height is not None else orig_height,
            'Width': width if width is not None else orig_width,
            'Headers': ['X', 'Y', 'Longitude', 'Latitude'] + [f'Band_{i}' for i in range(1, num_bands + 1)]
        }

    create_empty_csv_files(output_folder, header)

    for idx, image_path in enumerate(total_images):
        extract_pixel_values(image_path, output_folder, header, width, height)
        if progress_callback:
            progress_callback(idx + 1, len(total_images), os.path.basename(image_path))
