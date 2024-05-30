import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


# Calculate histogram per img
def get_histogram(img):
    # Convert greyscale img from 2D to 1D
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    return hist


# Path to directories
data_folders = ['../data/classes/0', '../data/classes/1', '../data/classes/2', '../data/classes/3']
class_names = ['Angry', 'Happy', 'Neutral', 'Focused']

# Placeholder for aggregated pixels
aggregated_histograms = {class_name: np.zeros(256) for class_name in class_names}

# Get all files in the parent directory
for data_folder, class_name in zip(data_folders, class_names):
    for file in os.listdir(data_folder):
        image_path = os.path.join(data_folder, file)
        # Read image already greyscale
        image = io.imread(image_path)
        histogram = get_histogram(image)
        # Sum the pixels into the corresponding aggregate histogram
        aggregated_histograms[class_name] += histogram

# Plotting the aggregated pixel intensity histogram
# Format adjustments
for class_name, aggregated_histogram in aggregated_histograms.items():
    plt.figure(figsize=(12, 8))  # Set figure size
    plt.bar(range(256), aggregated_histogram, color='black', edgecolor='black', alpha=0.7)
    plt.title(f'Pixel Intensity Distribution for {class_name}', fontsize=20, fontweight='bold')
    plt.xlabel('Gray Level', fontsize=13)
    plt.ylabel('Pixel Count', fontsize=13)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.savefig(f'aggregate_pixel_intensity_{class_name}.png', dpi=300)
    plt.show()
