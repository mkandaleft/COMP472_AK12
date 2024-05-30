import os
import cv2


# Define function to read all files in folder and resize
# Overwrites all files in source folder
def resize_in_folder(source_folder, size=(48, 48)):
    # Iterate over all files (img format already)
    for filename in os.listdir(source_folder):
        # Construct full file path
        file_path = os.path.join(source_folder, filename)

        # Check if the file is an image
        if os.path.isfile(file_path):
            # Read the image as greyscale
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # Convert the image to greyscale
            if image is not None:
                # Resize the image testing
                # resized = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
                # resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
                resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)

                cv2.imwrite(file_path, resized)


# Define source folder (applied to focused class only)
source_folder = '../data/classes/3'

# Call the function to resize images
resize_in_folder(source_folder)
