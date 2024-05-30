from skimage.io import imread, imsave
from skimage.exposure import cumulative_distribution
import numpy as np
import os


def load_from_folder(folder):
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(folder):
        if filename.lower().endswith(valid_extensions):
            img = imread(os.path.join(folder, filename), as_gray=True)
            if img is not None:
                images.append((img * 255).astype(np.uint8))
    return images


# Function used to get comparison histograms for the report
def get_histogram(img):
    # Convert greyscale img from 2D to 1D
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    return hist


# Calculate aggregate cd
# Fills in missing bins
def aggregate_cumulative_distribution(images):
    all_cumulative_distributions = []
    for img in images:
        cd, bins = cumulative_distribution(img)
        cd = np.insert(cd, 0, [0] * bins[0])  # fill 0 in index 0 - bins[0]
        cd = np.append(cd, [1] * (255 - bins[-1]))  # fill 1 in index bins[-1] - 255
        all_cumulative_distributions.append(cd)

    aggregate_cd = np.mean(all_cumulative_distributions, axis=0)
    return aggregate_cd


# Function to match target histogram with aggregate
def histogram_match(image, template_cd, aggregate_cd):
    pxl = np.arange(256)
    pxl_transform = np.interp(template_cd, aggregate_cd, pxl)
    return (np.reshape(pxl_transform[image.ravel()], image.shape)).astype(np.uint8)


# Choose source class and target class
source_path = '../data/classes/0'
tester_path = '../data/classes/3'

# Load all images from source_path
images = load_from_folder(source_path)

# Compute aggregate cumulative distribution
aggregate_cd = aggregate_cumulative_distribution(images)

# Process each image in the tester_path
for filename in os.listdir(tester_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(tester_path, filename)
        image = imread(image_path).astype(np.uint8)  # treat img as grayscale

        cd_image, bins_image = cumulative_distribution(image)
        cd_image = np.insert(cd_image, 0, [0] * bins_image[0])  # fill 0 in index 0 - bins_image[0]
        cd_image = np.append(cd_image, [1] * (255 - bins_image[-1]))  # fill 1 in index bins_image[-1] - 255

        matched_image = histogram_match(image, cd_image, aggregate_cd)

        # Save the matched image 
        # Overwrites original target file
        output_image_path = os.path.join(tester_path, filename)
        imsave(output_image_path, matched_image)

# Code to preview results used in report
# plt.figure(figsize=(10,6))
# plt.subplot(1,2,1)
# plt.title('Resized Image')
# plt.imshow(imageTemplate, cmap='gray')
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.title('Processed Image')
# plt.imshow(imageOut, cmap='gray')
# plt.axis('off')
# plt.savefig('cd_processed_img-final.png', dpi=300)
# plt.show()

# Plot the aggregate cumulative distribution
# plt.plot(pxl_transform, linewidth=4, color="grey", label='Matched CD')
# plt.xlabel('Pixel Values')
# plt.ylabel('Cumulative Probability')
# plt.xlim(0,255)
# plt.ylim(0,255)
# plt.legend()
# plt.savefig('cd_matched.png', dpi=300)
# plt.show()

# plt.plot(aggregate_cd, linewidth=5, color="black", label='Aggregate of Class 0')
# plt.plot(cdimageTemplate, linewidth=5, color="grey", label='Target Image')
# plt.xlim(0,255)
# plt.ylim(0,1)
# plt.xlabel('Pixel Values')
# plt.ylabel('Cumulative Probability')
# plt.legend()
# plt.savefig('cd_matching.png', dpi=300)
# plt.show()
