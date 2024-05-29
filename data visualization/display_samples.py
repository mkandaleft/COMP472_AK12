import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the path to the parent directory containing the folders
parent_dir = './classes'

# Define the folder names (these should correspond to your class labels)
class_folders = ['0', '1', '2', '3']

# Number of sample images to display for each class
num_samples = 15
rows = 5
cols = 3

# Function to plot images and histograms in a grid
def plot_images_and_histograms_grid(class_folder, images, class_label):
    fig, axes = plt.subplots(rows, cols*2, figsize=(20, 10))
    fig.suptitle(f'Class {class_label}', fontsize=16)

    for idx, img in enumerate(images):
        row = idx // cols
        col = (idx % cols) * 2
        
        # Open the image
        img_path = os.path.join(class_folder, img)
        image = Image.open(img_path).convert('L')
        
        # Plot the image
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
        
        # Plot the pixel intensity histogram
        pixels = np.array(image).flatten()
        axes[row, col+1].hist(pixels, bins=256, range=(0, 256), color='gray')
        axes[row, col+1].set_xlim(0, 255)
        axes[row, col+1].set_ylim(0, max(axes[row, col+1].get_ylim()))
        axes[row, col+1].set_xlabel('Pixel Intensity')
        axes[row, col+1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Process each class folder
for class_label in class_folders:
    class_folder = os.path.join(parent_dir, class_label)
    if os.path.isdir(class_folder):
        # Get all .png or .jpg files in the class folder
        images = [f for f in os.listdir(class_folder) if (f.endswith('.png')or f.endswith('.jpg')) ]
        
        # Randomly select sample images
        sample_images = random.sample(images, min(num_samples, len(images)))
        
        # Plot the sample images and histograms in a grid
        plot_images_and_histograms_grid(class_folder, sample_images, class_label)
