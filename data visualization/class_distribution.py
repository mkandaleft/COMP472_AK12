import os
import matplotlib.pyplot as plt


parent_dir = './classes'


class_folders = ['0', '1', '2', '3']

# dictionary to hold the count of images in each class
class_counts = {class_folder: 0 for class_folder in class_folders}

# Count the number of images in each class folder
for class_folder in class_folders:
    folder_path = os.path.join(parent_dir, class_folder)
    if os.path.isdir(folder_path):
        class_counts[class_folder] = len([f for f in os.listdir(folder_path) if (f.endswith('.png') or f.endswith('.jpg'))])

# Plot the class distribution on a histogram
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution of Images')
plt.xticks(rotation=45)
plt.show()
