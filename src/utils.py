import os
import csv
import random
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import rasterio

from .dataset_preprocessing import transform_coordinates_to_pixel


def extract_pixel_values_by_coord(longitude: float, latitude: float, image_folder: str) -> list:
    """
    Extracts multi-band time-series pixel values for a single Lat/Lon coordinate across satellite rasters.
    """
    pixel_values = []
    for image_file in sorted(os.listdir(image_folder)):
        if image_file.endswith('.tif'):
            image_path = os.path.join(image_folder, image_file)
            with rasterio.open(image_path) as src:
                row, col = transform_coordinates_to_pixel(latitude, longitude, src)
                for band in src.indexes:
                    band_data = src.read(band)
                    row_idx = min(row, src.height - 1)
                    col_idx = min(col, src.width - 1)
                    pixel_values.append(band_data[row_idx, col_idx])
    return pixel_values


def process_csv_row(row: dict, image_folder: str, output_folder: str):
    """
    Processes a single metadata CSV row (with Latitude, Longitude, Crop) and extracts satellite time-series.
    """
    longitude = float(row['Longitude'])
    latitude = float(row['Latitude'])
    crop_type = row.get('Crop', row.get('Crop_Type', 'Unknown'))
    
    pixel_values = extract_pixel_values_by_coord(longitude, latitude, image_folder)
    
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{longitude:.4f}_{latitude:.4f}.csv")
    
    tif_files = [f for f in os.listdir(image_folder) if f.endswith('.tif')]
    num_tifs = max(len(tif_files), 1)
    num_bands = len(pixel_values) // num_tifs
    
    header = [f'Band_{i}' for i in range(1, num_bands + 1)] + ['Crop_Type']
    
    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(header)
        for i in range(0, len(pixel_values), num_bands):
            writer.writerow(pixel_values[i:i + num_bands] + [crop_type])


def process_csv_parallel(csv_file: str, image_folder: str, output_folder: str, num_cores: int = None):
    """
    Processes CSV records in parallel across CPU cores.
    """
    if num_cores is None:
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        
    with open(csv_file, 'r') as file:
        reader = list(csv.DictReader(file))
        
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        process_row_partial = partial(process_csv_row, image_folder=image_folder, output_folder=output_folder)
        list(executor.map(process_row_partial, reader))


def move_random_files(source_dir: str, dest_dir: str, num_files: int):
    """
    Randomly splits/moves a specified number of files from source_dir to dest_dir for validation sets.
    """
    os.makedirs(dest_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    selected_files = random.sample(csv_files, min(num_files, len(csv_files)))
    
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.move(source_path, dest_path)
