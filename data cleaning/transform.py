import pandas as pd
import numpy as np
import os
from PIL import Image

# Load the CSV file
df = pd.read_csv('../data/extracted dataset/angry_happy_neutral.csv')

# Extract pixel values and convert them to numpy arrays
df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

# Create a directory for each emotion
emotion_labels = df['emotion'].unique()
output_dir = 'emotion_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for emotion in emotion_labels:
    emotion_dir = os.path.join(output_dir, str(emotion))
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)

# Save images into the corresponding emotion folder
for idx, row in df.iterrows():
    emotion = row['emotion']
    pixels = row['pixels'].reshape(48, 48)  # Reshape the pixels to 48x48
    img = Image.fromarray(pixels.astype('uint8'), mode='L')  # Create a grayscale image
    img_path = os.path.join(output_dir, str(emotion), f'{idx}.png')
    img.save(img_path)

print(f'Images have been successfully saved into {output_dir}')
