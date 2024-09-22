import csv
import os
import rasterio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

def transform_coordinates(lat, lon, src):
    image_crs = src.crs
    latlon_crs = rasterio.crs.CRS.from_epsg(4326)
    transformer = rasterio.warp.Transformer.from_crs(latlon_crs, image_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = src.index(x, y)
    print(f"row:{row}, col:{col}")
    return row, col


def extract_pixel_values(longitude, latitude, image_folder):
    pixel_values = []
    for image_file in os.listdir(image_folder):
        if image_file.endswith('.tif'):
            image_path = os.path.join(image_folder, image_file)
            with rasterio.open(image_path) as src:
                transform = src.transform
                x, y = transform_coordinates(latitude, longitude, src)
                for band in src.indexes:
                    band_data = src.read(band)
                    pixel_value = band_data[x, y]
                    pixel_values.append(pixel_value)
    return pixel_values


def process_csv_row(row, image_folder, output_folder):
    longitude = float(row['Longitude'])
    latitude = float(row['Latitude'])
    crop_type = row['Crop']
    pixel_values = extract_pixel_values(longitude, latitude, image_folder)
    output_file = os.path.join(output_folder, f"{longitude},{latitude}.csv")
    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        num_bands = len(pixel_values) // len([f for f in os.listdir(image_folder) if f.endswith('.tif')])
        header = ['Band_' + str(i) for i in range(1, num_bands + 1)] + ['Crop_Type']
        writer.writerow(header)
        for i in range(0, len(pixel_values), num_bands):
            writer.writerow(pixel_values[i:i + num_bands] + [crop_type])
    print("CSV created for", output_file)


def process_csv_parallel(csv_file, image_folder, output_folder, num_cores=None):
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()  # Use all available cores by default
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            process_row_partial = partial(process_csv_row, image_folder=image_folder, output_folder=output_folder)
            executor.map(process_row_partial, reader)


if __name__ == "__main__":
    csv_file = 'total_data.csv'
    image_folder = 'G:\ACPS-Project\Help\lowerArea'
    output_folder = 'G:\ACPS-Project\Help\output_csv_files'
    process_csv_parallel(csv_file, image_folder, output_folder, 2)
