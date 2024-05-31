import os
import cv2

possible_extensions = ('.png', '.jpg', '.jpeg', '.tiff')
parent_dir = '../data/classes'
class_folders = ['0', '1', '2', '3']


# Function to convert and save images in PNG format
def convert_to_png(folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith(possible_extensions):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            # Create the new filename with .png extension
            base_name = os.path.splitext(file)[0]
            new_name = base_name + '.png'
            new_path = os.path.join(folder_path, new_name)

            # Save the image in PNG format
            cv2.imwrite(new_path, img)

            # Remove the original file if it's not already a PNG
            if not file.lower().endswith('.png'):
                os.remove(img_path)


# Convert all images in the folder to PNG format
for class_folder in class_folders:
    folder_path = os.path.join(parent_dir, class_folder)
    if os.path.isdir(folder_path):
        convert_to_png(folder_path)
