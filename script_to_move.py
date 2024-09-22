import os
import random
import shutil

def move_random_files(source_dir, dest_dir, num_files):
    # Get list of CSV files in the source directory
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    
    # Randomly select 'num_files' files from the list
    selected_files = random.sample(csv_files, min(num_files, len(csv_files)))
    
    # Move selected files to the destination directory
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.move(source_path, dest_path)
        print(f"Moved {file_name} to {dest_dir}")

# Example usage:
source_directory = "G:\ACPS-Project\Help\Dataset\Paddy"
destination_directory = "G:\ACPS-Project\Help\Paddy_extra"
num_files_to_move = 700

move_random_files(source_directory, destination_directory, num_files_to_move)