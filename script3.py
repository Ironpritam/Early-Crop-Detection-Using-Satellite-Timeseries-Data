import os
import csv
import numpy as np
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer

# Function to create empty CSV files with proper headers
def create_empty_csv_files(csv_file, output_folder):
    with open(csv_file, 'r') as file:
        # Add band information to the headers
        num_bands = 8  
        band_headers = [f"Band_{i}" for i in range(1, num_bands + 1)]
        headers = band_headers

        # Append coordinates and crop type to the headers
        headers += ['X', 'Y', 'Crop_Type']

        reader = csv.DictReader(file)
        for row in reader:
            # Create a separate folder for each crop type
            crop_type = row['Crop']
            crop_folder = os.path.join(output_folder, crop_type)
            os.makedirs(crop_folder, exist_ok=True)
            # Create an empty CSV file with proper headers
            file_name = f"{row['Longitude']},{row['Latitude']},.csv"
            file_path = os.path.join(crop_folder, file_name)
            with open(file_path, 'w', newline='') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(headers)

# Function to transform coordinates to pixel coordinates
def transform_coordinates(lat, lon,src):
#   with rasterio.open("/content/drive/MyDrive/acps/Acps project data/lowerArea/20221206.tif") as src:
      # Get the CRS of the image
      image_crs = src.crs

      # Loop through each set of coordinates
      # Define the CRS for longitude and latitude coordinates (typically EPSG:4326)
      latlon_crs = CRS.from_epsg(4326)

      # Transform the longitude and latitude coordinates to the CRS of the image
      transformer = Transformer.from_crs(latlon_crs, image_crs, always_xy=True)
      x, y = transformer.transform(lon, lat)

      # Convert the transformed coordinates to pixel coordinates
      row, col = src.index(x, y)

      return abs(row), abs(col)



def transform_coordinates(lat, lon, src):
    raster_crs = src.crs
    your_coordinates_crs = CRS.from_epsg(4326)

    if raster_crs == your_coordinates_crs:
        row, col = src.index(lon, lat)
        row = abs(row // 8)
        col = abs(col // 8)
        return row, col
    else:
        transformer = Transformer.from_crs(your_coordinates_crs, raster_crs, always_xy=True)
        new_lon, new_lat = transformer.transform(lon, lat)
        row, col = src.index(new_lon, new_lat)
        row = abs(row // 8)
        col = abs(col // 8)
        return row, col

# Function to extract pixel values from a single image
def extract_pixel_values(image_path, pixels):
    with rasterio.open(image_path) as src:
        # Iterate through each pixel
        for pixel in pixels:
            longitude, latitude = float(pixel['Longitude']), float(pixel['Latitude'])
            row, col = transform_coordinates(latitude, longitude, src)
            # Initialize 'Pixel_Values' key if not already present
            pixel['X'] = row
            pixel['Y'] = col
            # if 'Pixel_Values' not in pixel:
            pixel['Pixel_Values'] = []
            # Read pixel values for each band
            for band in src.indexes:
                band_data = src.read(band)
                pixel_value = band_data[row, col]
                pixel['Pixel_Values'].append(pixel_value)

# Function to process CSV file and update pixel values
def process_csv(csv_file, image_folder, output_folder):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        pixels = list(reader)

    # Create empty CSV files with proper headers
    create_empty_csv_files(csv_file, output_folder)

    # Iterate through each image in the folder
    for image_file in os.listdir(image_folder):
        # Check if the file is a .tif file
        if image_file.endswith('.tif'):
            image_path = os.path.join(image_folder, image_file)
            # Extract pixel values from the image and update the pixels list
            extract_pixel_values(image_path, pixels)

            # Write updated pixel values to CSV files
            for pixel in pixels:
                crop_type = pixel['Crop']
                crop_folder = os.path.join(output_folder, crop_type)
                file_name = f"{pixel['Longitude']},{pixel['Latitude']},.csv"
                file_path = os.path.join(crop_folder, file_name)
                with open(file_path, 'a', newline='') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(pixel['Pixel_Values'] + [pixel['X'], pixel['Y'], pixel['Crop']])
            print("CSV updated for", image_file)

# Example usage
csv_file = 'total_data.csv'
image_folder = 'lowerArea'
output_folder = 'output_csv_files_3'
process_csv(csv_file, image_folder, output_folder)
