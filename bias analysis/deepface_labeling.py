import os
import warnings
import pandas as pd
import cv2
from deepface import DeepFace
import gc
import shutil

warnings.filterwarnings("ignore")

parent_dir = "../data/classes/"

# Define dictionaries for each emotion class
data = {
    "0": {"name": [], "age": [], "gender": []},
    "1": {"name": [], "age": [], "gender": []},
    "2": {"name": [], "age": [], "gender": []},
    "3": {"name": [], "age": [], "gender": []},
}

# Gender dictionary to map results
gender_dict = {
    "Man": "m",
    "Woman": "f"
}


# Function to process image using DeepFace
def process_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image at path: {file_path}")

    if len(img.shape) == 2:  # Grayscale image 2 channels
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:  # Grayscale image 1 channel
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:  # Already RGB
        img_rgb = img

    result = DeepFace.analyze(img_rgb, actions=("age", "gender"), enforce_detection=False)

    age = result[0]["age"]
    gender = gender_dict.get(result[0]["dominant_gender"], "Unknown")

    return age, gender


# Define batch
batch_size = 10  # For processing issues


# Create gender-specific dir
def create_gender_dirs(base_path, gender):
    gender_dir = os.path.join(base_path, gender)
    if not os.path.exists(gender_dir):
        os.makedirs(gender_dir)
    return gender_dir


# Load images
for dir_name, category in data.items():
    dir_path = os.path.join(parent_dir, dir_name)
    if os.path.isdir(dir_path):
        files = os.listdir(dir_path)
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            for file_name in batch_files:
                file_path = os.path.join(dir_path, file_name)
                try:
                    age, gender = process_image(file_path)
                    category["name"].append(f"{dir_name}_{file_name}")
                    category["age"].append(age)
                    category["gender"].append(gender)

                    # gender_dir = create_gender_dirs(dir_path, gender)

                    # Move img to gender directory
                    # shutil.move(file_path, os.path.join(gender_dir, file_name))

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
            # Run garbage collection to free up memory (if OOM issue)
            gc.collect()

# Create and save dataframes
for emotion, df in data.items():
    pd.DataFrame(df).to_csv(f"{emotion}_age_gender.csv", index=False)
