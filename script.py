
import pandas as pd
import os
import csv
import numpy as np
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer

csv_path = r"/content/drive/MyDrive/acps/Acps project data/lowerFinal.csv"

# Read CSV file into a DataFrame


# Extract coordinates and crop
# coordinatesL = df[['Latitude', 'Longitude']].values.tolist()
# cropsL = df['Crop'].tolist()


# folder_path = r"/content/drive/MyDrive/acps/Acps project data/lowerArea"
# files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

def min_max_normCCC(x):
    return ((x-np.nanpercentile(x,2))/(np.nanpercentile(x,98)-np.nanpercentile(x,2)))

# Loop through each .tif file and visualize it
# for tif_file in files:
#     file_path = str(folder_path+'/'+tif_file)
#     with rio.open(file_path) as src:
#         # Read each band of the raster separately
#         r = src.read(6)  # red channel
#         r1 = r
#         g = src.read(4)  # green channel
#         b = src.read(2)  # blue channel
#         band1 = src.read(1)
#         band2 = src.read(2)
#         band3 = src.read(3)
#         band4 = src.read(4)
#         band5 = src.read(5)
#         band6 = src.read(6)
#         band7 = src.read(7)
#         band8 = src.read(8)

#         # ba = [src.read(i) for i in range(1,9,1)]
#         # ban = [min_max_normCCC(i) for i in ba]
#         # band = ba
#         # band_min_max = ban

#         # Normalize each band separately
#         r = min_max_normCCC(r)
#         g = min_max_normCCC(g)
#         b = min_max_normCCC(b)

#         # Stack the bands to form an RGB image
#         img_1 = np.dstack((r, g, b))

#         # Define the new shape
#         new_shape = (4640, 4400)

#         # Reshape the image using array slicing
#         reshaped_img = img_1[:new_shape[0], :new_shape[1], :]



def transform_coordinates(lat, lon, src):
#   with rasterio.open("/content/drive/MyDrive/acps/Acps project data/lowerArea/20221206.tif") as dataset:
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
      print(f"row:{row}, col:{col}")
      return row, col

def trans(lat, lon, src):
#   with rasterio.open(path) as dataset:
        #print(dataset.shape)
    raster_crs = src.crs
    # print(lon, " ", lat)
    # lon = 76.453007
    # lat = 30.961394
    your_coordinates_crs = CRS.from_epsg(4326)

    if raster_crs == your_coordinates_crs:
        row, col = src.index(lon, lat)
        row =abs(row//8)
        col = abs(col//8)

        if(col>=0):
          print(row, " ", col)
          return row, col
        #   #print(i, reshaped_ndvi_lower[row][col])
        #   results.append(reshaped_ndvi_lower[row][col])
    else:
        transformer = Transformer.from_crs(your_coordinates_crs, raster_crs, always_xy=True)
        new_lon, new_lat = transformer.transform(lon, lat)
        row, col = src.index(new_lon, new_lat)
        row = abs(row//8)
        col = abs(col//8)
        if col>=0:
          #print(i, reshaped_ndvi_lower[row][col])
          # results.append(reshaped_ndvi_lower[row][col])
          print(row, " ", col)
          return row, col

# Function to extract pixel values from time-series satellite images
def extract_pixel_values(longitude, latitude, image_folder):
    # Initialize list to store pixel values
    pixel_values = []
    # Iterate through each image in the folder
    for image_file in os.listdir(image_folder):
        # Check if the file is a .tif file
        if image_file.endswith('.tif'):
            # Open the image file
            image_path = os.path.join(image_folder, image_file)
            with rasterio.open(image_path) as src:
                # Get transform to convert latitude and longitude to image coordinates
                transform = src.transform
                # Get matrix coordinates
                x, y = transform_coordinates(latitude, longitude, src)
                # Read pixel values for each band
                for band in src.indexes:
                    band_data = src.read(band)
                    pixel_value = band_data[x, y]
                    pixel_values.append(pixel_value)
    return pixel_values




def process_csv(csv_file, image_folder, output_folder):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Extract information from CSV row
            longitude = float(row['Longitude'])
            latitude = float(row['Latitude'])
            crop_type = row['Crop']
            # Extract pixel values from time-series satellite images
            pixel_values = extract_pixel_values(longitude, latitude, image_folder)
            # Write to output CSV file
            output_file = os.path.join(output_folder, f"{longitude},{latitude},.csv")
            with open(output_file, 'w', newline='') as out_file:
                writer = csv.writer(out_file)
                # Write header
                num_bands = len(pixel_values) // len([f for f in os.listdir(image_folder) if f.endswith('.tif')])
                header = ['Band_' + str(i) for i in range(1, num_bands + 1)] + ['Crop_Type']
                writer.writerow(header)
                # Write pixel values and crop type
                for i in range(0, len(pixel_values), num_bands):
                    writer.writerow(pixel_values[i:i + num_bands] + [crop_type])
            print("csv created ...")





csv_file = 'total_data.csv'
image_folder = 'lowerArea'
output_folder = 'output_csv_files'
process_csv(csv_file, image_folder, output_folder)