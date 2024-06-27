import pandas as pd
import os
import shutil

# List of CSV files
csv_files = ["0_labeled.csv", "1_labeled.csv", "2_labeled.csv", "3_labeled.csv"]

parent_folder = "../data/classes/"


# Function to copy images to the respective gender folder
def find_by_gender(csv_file, class_dir):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        # Extract img name by removing class label
        image_name = row['name'][2:]
        gender = row['gender']

        source_path = os.path.join(parent_folder, class_dir, image_name)

        if not os.path.isfile(source_path):
            print(f"Source file not found: {source_path}")
            continue

        # Create gender dir
        dest_dir = os.path.join(parent_folder, class_dir, gender)

        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, image_name)

        # Copy the image
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            print(f"Error copying {source_path} to {dest_path}: {e}")


# Mapping of CSV files
for csv_file in csv_files:
    # Extract the class directory num
    class_dir = csv_file.split('_')[0]
    find_by_gender(csv_file, class_dir)

