import os
import shutil
import math

def split_folder_contents(folder_path, num_subfolders):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Calculate the number of files per subfolder
    files_per_subfolder = math.ceil(len(files) / num_subfolders)

    # Create subfolders
    for i in range(num_subfolders):
        subfolder_name = f"subfolder_{i + 1}"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # Calculate the range of files for this subfolder
        start_index = i * files_per_subfolder
        end_index = (i + 1) * files_per_subfolder

        # Copy files to the subfolder
        for file_name in files[start_index:end_index]:
            file_path = os.path.join(folder_path, file_name)
            shutil.move(file_path, subfolder_path)

if __name__ == "__main__":
    folder_path = "/home/aro/Schreibtisch/train_set/images"  # Replace with the path to your folder
    num_subfolders = 300

    split_folder_contents(folder_path, num_subfolders)
