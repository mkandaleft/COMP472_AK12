import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import cumulative_distribution

# Define dir
directories = {
    "Angry": "../data/classes/0",
    "Happy": "../data/classes/1",
    "Neutral": "../data/classes/2",
    "Focused": "../data/classes/3"
}


# Function gets cd per image per dir
# Used only to get an idea for avg differences in pxl intensity
def average_cd(directory):
    sum_cd_image = np.zeros(256)
    image_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".png"):

            file_path = os.path.join(directory, filename)
            # Read the image in grayscale mode
            image = imread(file_path, as_gray=True)
            # Check if the image is already in 255 format
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            # Compute the cumulative distribution of the image
            cd_image, bins_image = cumulative_distribution(image)
            # Adjust the cumulative distribution to account for all pixel values
            cd_image = np.insert(cd_image, 0, [0] * bins_image[0])
            cd_image = np.append(cd_image, [1] * (255 - bins_image[-1]))
            # Add  cd
            sum_cd_image += cd_image
            image_count += 1

    # Compute the average cumulative distribution
    average_cd_image = sum_cd_image / image_count
    return average_cd_image


# Plot the average cumulative distributions for each directory
plt.figure(figsize=(10, 6))

for label, directory in directories.items():
    avg_cd = average_cd(directory)
    plt.plot(avg_cd, linewidth=2, label=label)

plt.xlim(0, 255)
plt.ylim(0, 1)
plt.xlabel('Pixel Values')
plt.ylabel('Cumulative Probability')
plt.title('Average Cumulative Distributions for Emotion Classes')
plt.legend()
# plt.savefig('cd_all_classes.png', dpi=300)
plt.show()
