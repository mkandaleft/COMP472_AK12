from imagededup.methods import CNN
from imagededup.utils import plot_duplicates
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = (15, 10)

# Initialize pretrained MobileNet
cnn = CNN()

# Define path to img
img_path = '../data/classes/0'

# Generate encodings using Global Average Pooling
encodings = cnn.encode_images(image_dir=img_path)

# Identify img that have duplicates
duplicates = cnn.find_duplicates(encoding_map=encodings, scores=True)

threshold = 0.97

# Filter duplicates based on the threshold
filtered_duplicates = {}
for key, value in duplicates.items():
    filtered_value = [dup for dup in value if dup[1] >= threshold]
    if filtered_value:
        filtered_duplicates[key] = filtered_value

print(len(filtered_duplicates))

# Iterate through the duplicates dictionary
for key, value in filtered_duplicates.items():
    if len(value) > 0:
        # Delete duplicate values
        for duplicate_tuple in value:
            duplicate = duplicate_tuple[0]  # Extract filename, need string
            duplicate_path = os.path.join(img_path, duplicate)
            if os.path.exists(duplicate_path):
                os.remove(duplicate_path)

# To visualize an arbitrary image and its duplicates
# plot_duplicates(image_dir=img_path,
#                 duplicate_map=duplicates,
#                 filename='1001.png')
