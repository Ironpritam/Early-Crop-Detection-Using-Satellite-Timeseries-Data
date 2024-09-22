import os
import csv
import rasterio
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image

# Function to create empty CSV files with proper headers
def create_empty_csv_files(output_folder, header):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create empty CSV files for each pixel coordinate
    for row in range(header['Height']):
        for col in range(header['Width']):
            file_name = f"{row}_{col}.csv"
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header['Headers'])

# Function to extract pixel values from a single image
def extract_pixel_values(image_path, output_folder, header, progress_bar, total_files, width=None, height=None):
    with rasterio.open(image_path) as src:
        num_bands = src.count
        orig_height, orig_width = src.height, src.width
        
        if width is None or height is None:
            width, height = orig_width, orig_height

        for row in range(height):
            for col in range(width):
                # Transform pixel coordinates to latitude and longitude
                lon, lat = transform_pixel_to_coordinates(row, col, src, orig_height, orig_width,height,width)

                # Initialize a dictionary to store pixel information
                pixel_info = {'X': row, 'Y': col, 'Longitude': lon, 'Latitude': lat}

                # Read pixel values for each band
                for band in range(1, num_bands + 1):
                    band_data = src.read(band)
                    pixel_value = band_data[row, col]
                    pixel_info[f'Band_{band}'] = pixel_value

                # Write pixel information to CSV
                file_name = f"{row}_{col}.csv"
                file_path = os.path.join(output_folder, file_name)
                with open(file_path, 'a', newline='') as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=header['Headers'])
                    if os.path.getsize(file_path) == 0:  # Write header if file is empty
                        writer.writeheader()
                    writer.writerow(pixel_info)

                # Update progress bar
                progress_bar['value'] += 1
                progress_bar.update()
                
                remaining_files = total_files - progress_bar['value']
                progress_bar['title'] = f"Remaining Images: {remaining_files}"

# Function to transform pixel coordinates to latitude and longitude
def transform_pixel_to_coordinates(row, col, src, orig_height, orig_width,height,width):
    # Get the CRS of the image
    image_crs = src.crs

    # Convert pixel coordinates to coordinates
    lon, lat = src.xy(row * (orig_height - 1) / (height - 1), col * (orig_width - 1) / (width - 1))

    return lon, lat

# Function to resize images
def resize_image(image_path, output_folder, width, height):
    with Image.open(image_path) as img:
        resized_img = img.resize((width, height), Image.ANTIALIAS)
        resized_path = os.path.join(output_folder, os.path.basename(image_path))
        resized_img.save(resized_path)
        return resized_path

# Function to process images and generate CSV files for each pixel
def process_images(image_folder, output_folder, progress_bar, width=None, height=None):
    # Get header information from the first image
    total_images = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.tif')]

    first_image = total_images[0]
    first_image_path = os.path.join(image_folder, first_image)
    with rasterio.open(first_image_path) as src:
        num_bands = src.count
        orig_height, orig_width = src.height, src.width
        header = {
            'Height': height if height is not None else orig_height,
            'Width': width if width is not None else orig_width,
            'Headers': ['X', 'Y', 'Longitude', 'Latitude'] + [f'Band_{i}' for i in range(1, num_bands + 1)]
        }

    # Create empty CSV files with proper headers
    create_empty_csv_files(output_folder, header)

    # Get total number of files to process
    total_files = len(total_images)

    # Iterate through each image in the folder
    for idx, image_file in enumerate(total_images):
        # Check if the file is a .tif file
        if image_file.endswith('.tif'):
            image_path = os.path.join(image_folder, image_file)
            # Resize image if width and height are provided
            if width is not None and height is not None:
                image_path = resize_image(image_path, output_folder, width, height)
            # Extract pixel values from the image and generate CSV files
            extract_pixel_values(image_path, output_folder, header, progress_bar, total_files, width, height)

            print(f"CSV generated for image {idx+1}/{total_files}: {image_file}")

# Function to handle the button click event
def generate_csv_dataset():
    # Create a Tkinter Toplevel window
    sec_window = tk.Toplevel()
    sec_window.title("Generate CSV Dataset")
    
    # Set background color
    sec_window.configure(bg="#2E1A47")
    sec_window.geometry("800x600")
    
    # Variables to store selected directories
    input_folder_var = tk.StringVar()
    output_folder_var = tk.StringVar()
    width_var = tk.StringVar()
    height_var = tk.StringVar()
    
    # Function to update text boxes with selected directories
    def update_input_folder(folder):
        input_folder_var.set(folder)
    
    def update_output_folder(folder):
        output_folder_var.set(folder)
    
    # Function to select input directory
    def browse_input_folder():
        folder = filedialog.askdirectory()
        update_input_folder(folder)
    
    # Function to select output directory
    def browse_output_folder():
        folder = filedialog.askdirectory()
        update_output_folder(folder)
    
    # Ask for the input image folder
    input_label = tk.Label(sec_window, text="Select Input Image Folder:",  bg="black",fg="white",font=('bold',10))
    input_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    input_entry = tk.Entry(sec_window, textvariable=input_folder_var, width=50)
    input_entry.grid(row=0, column=1, padx=10, pady=10)  # Check this line

    input_button = tk.Button(sec_window, text="Browse", command=browse_input_folder,bg="#4CAE55", fg="black")
    input_button.grid(row=0, column=2, padx=10, pady=10)

    # Ask for the output CSV folder
    output_label = tk.Label(sec_window, text="Select Output CSV Folder:", bg="black",fg="white",font=('bold',10))
    output_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    output_entry = tk.Entry(sec_window, textvariable=output_folder_var, width=50)
    output_entry.grid(row=1, column=1, padx=10, pady=10)  # Check this line

    output_button = tk.Button(sec_window, text="Browse", command=browse_output_folder,bg="#4CAE55", fg="black")
    output_button.grid(row=1, column=2, padx=10, pady=10)
    
    # Provide resizing width and height information
    resize_info_label = tk.Label(sec_window, text="Resizing parameters, otherwise default size will be considered which could be memory as well as computationally heavy!", bg="black",fg="white",font=('bold',10))
    resize_info_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)
    
    # Ask for the width of the images
    width_label = tk.Label(sec_window, text="Width:", bg="black",fg="white",font=('bold',10))
    width_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

    width_entry = tk.Entry(sec_window, textvariable=width_var, width=10)
    width_entry.grid(row=3, column=1, padx=10, pady=10)  # Check this line
    
    # Ask for the height of the images
    height_label = tk.Label(sec_window, text="Height:", bg="black",fg="white",font=('bold',10))
    height_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

    height_entry = tk.Entry(sec_window, textvariable=height_var, width=10)
    height_entry.grid(row=4, column=1, padx=10, pady=10)  # Check this line
    
    # Progress bar
    progress_bar = Progressbar(sec_window, orient=tk.HORIZONTAL, length=300, mode='determinate', maximum=100, value=0)
    progress_bar.grid(row=5, columnspan=3, padx=10, pady=10)
    
    # Function to start processing
    def start_processing():
        input_folder = input_folder_var.get()
        output_folder = output_folder_var.get()
        width = int(width_var.get()) if width_var.get() else None
        height = int(height_var.get()) if height_var.get() else None
        
        if not input_folder or not output_folder:
            messagebox.showerror("Error", "Please select both input image and output CSV folders.")
            return
        
        process_images(input_folder, output_folder, progress_bar, width, height)
        messagebox.showinfo("CSV Dataset Prepared", "CSV dataset prepared successfully!")
        sec_window.destroy()
    
    # Start button
    start_button = tk.Button(sec_window, text="Start Processing", command=start_processing, bg="#4CAE55", fg="black")
    start_button.grid(row=6, columnspan=3, padx=10, pady=10)

    # Set the window as a priority window
    sec_window.grab_set()
    # Wait for this window to be dealt with before going back to the parent window
    sec_window.wait_window()

# Example usage
if __name__ == "__main__":
    generate_csv_dataset()
