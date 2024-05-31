from imagededup.methods import CNN
from imagededup.utils import plot_duplicates
import matplotlib.pyplot as plt
import os

# Use imagedup package and its CNN cosine sim method to keep a dictionary of dups
def find_duplicates_in(directory):
    # Initialize the CNN method
    cnn = CNN()
    # Generate encodings for all images in the directory
    encodings = cnn.encode_images(image_dir=directory)
    # Find duplicate keys using adjusted minimum threshold
    duplicates = cnn.find_duplicates(encoding_map=encodings, min_similarity_threshold=0.97)

    # Initialize a dictionary to store found duplicates
    locate_dup = {}

    # Function find_duplicates from imagedup stores key-value and reverse key-value
    # Correction loop
    for key, values in duplicates.items():
        # if duplicate found
        if len(values) > 0:
            locate_dup[key] = values

    return locate_dup


# Function deletes images that correspond to dup keys
def delete_duplicate(dict, dir):
    for duplicate_array in dict.values():
        for img in duplicate_array:
            img_path = os.path.join(dir, img)
            if os.path.exists(img_path):
                os.remove(img_path)


path = "../data/classes/0"

locate_dup = find_duplicates_in(path)
# print("Imagedup identified duplicates:", locate_dup)

# Filter the duplicate dictionary to remove reverse pairs
processed = set()
unique_duplicates = {}

for key, duplicates in locate_dup.items():
    if key not in processed:
        unique_duplicates[key] = []
        for duplicate in duplicates:
            if duplicate not in processed:
                unique_duplicates[key].append(duplicate)
        processed.add(key)
        processed.update(duplicates)

# print(unique_duplicates)

# Visualize duplicates before deletion
# For verification of the program
# plt.rcParams['figure.figsize'] = (10, 6)
# for key in unique_duplicates.keys():
#     plot_duplicates(image_dir=path,
#                     duplicate_map=unique_duplicates,
#                     filename=key)


delete_duplicate(unique_duplicates, path)

if not unique_duplicates:
    print('No duplicates detected in the folder.')
