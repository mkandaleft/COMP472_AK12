import pandas as pd
import os
import shutil

# List of CSV files
csv_files = ["0_labeled.csv", "1_labeled.csv", "2_labeled.csv", "3_labeled.csv"]

parent_folder = "../data/classes/"


# Function to copy img to age folder
def classify_by_age(csv_file, class_dir):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        image_name = row['name'][2:]
        age = row['age']

        source_path = os.path.join(parent_folder, class_dir, image_name)

        if not os.path.isfile(source_path):
            print(f"Source file not found: {source_path}")
            continue

        # Determine the age group
        if age <= 18:
            age_group = "young"
        elif 19 <= age <= 59:
            age_group = "adult"
        else:
            age_group = "elder"

        # Create age group directory
        dest_dir = os.path.join(parent_folder, class_dir, age_group)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, image_name)

        # Copy the image
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            print(f"Error copying {source_path} to {dest_path}: {e}")


# Mapping of CSV files
for csv_file in csv_files:
    # Extract the class directory number
    class_dir = csv_file.split('_')[0]
    classify_by_age(csv_file, class_dir)
