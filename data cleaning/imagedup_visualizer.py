from imagededup.methods import CNN
from imagededup.utils import plot_duplicates
import matplotlib.pyplot as plt
import os

def find_duplicates_in(directory, threshold=0.97):
    # Initialize the CNN method
    cnn = CNN()
    # Generate encodings for all images in the directory
    encodings = cnn.encode_images(image_dir=directory)
    # Find duplicate keys using adjusted minimum threshold
    duplicates = cnn.find_duplicates(encoding_map=encodings, scores=True)

    # Filter duplicates based on the threshold
    filtered_duplicates = {}
    for key, value in duplicates.items():
        filtered_value = [dup for dup in value if dup[1] >= threshold]
        if filtered_value:
            filtered_duplicates[key] = filtered_value

    return filtered_duplicates

# Function deletes images that correspond to dup keys
def delete_duplicates(duplicate_dict, directory):
    for key, values in duplicate_dict.items():
        for duplicate_tuple in values:
            duplicate = duplicate_tuple[0]  # Extract filename
            duplicate_path = os.path.join(directory, duplicate)
            if os.path.exists(duplicate_path):
                os.remove(duplicate_path)

# Define path to the images
path = "../data/classes/0"

# Find duplicates
locate_dup = find_duplicates_in(path)

if not locate_dup:
    print('No duplicates detected in the folder.')
else:
    # Visualize duplicates before deletion
    plt.rcParams['figure.figsize'] = (15, 10)
    for key in locate_dup.keys():
        plot_duplicates(image_dir=path,
                        duplicate_map=locate_dup,
                        filename=key)

    # Delete duplicates
    delete_duplicates(locate_dup, path)
    print(f'{len(locate_dup)} duplicates detected and deleted.')

