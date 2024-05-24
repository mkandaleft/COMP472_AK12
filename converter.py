import numpy as np
from PIL import Image
import csv
import os

# Load data from CSV
with open('fer2013.csv') as file:
    images_data = list(csv.reader(file))[1:]

image_height, image_width = 48, 48
emotion_array, processed_data, converted_images = [], [], []

target_emotions = {'0', '3', '6'}

# Create directories for each target emotion
for emotion in target_emotions:
    os.makedirs(emotion, exist_ok=True)

# Process each row
for row in images_data:
    emotion, pixels_string = row[0].strip(), row[1].strip()

    if emotion in target_emotions:
        emotion_array.append(emotion)

        # Convert pixel string to numpy array
        pixels_array = np.array(list(map(int, pixels_string.split())), dtype=np.uint8)
        pixels_array = pixels_array.reshape(image_height, image_width)
        processed_data.append([emotion, pixels_string])

        # np to PIL image
        converted_images.append(Image.fromarray(pixels_array))

# Save images with corresponding emotion label
for i, (emotion, image) in enumerate(zip(emotion_array, converted_images)):
    image.save(f'{emotion}/img_{i}-{emotion}.png')

# Save to CSV
with open('processed.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Emotion', 'Pixels'])
    writer.writerows(processed_data)
