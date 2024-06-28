import os

root_dir = '../data/aged-merged/'

# Define the mapping of directories to suffixes
suffix_map = {
    'adult': 'a',
    'young': 'y',
    'elder': 'e'
    # 'female': 'f',
    # 'male': 'm'
}

# Iterate 
for age_group in os.listdir(root_dir):
    age_group_path = os.path.join(root_dir, age_group)

    if age_group in suffix_map and os.path.isdir(age_group_path):
        suffix = suffix_map[age_group]

        # Iterate through subdirectories (0, 1, 2, 3)
        for subdir in os.listdir(age_group_path):
            subdir_path = os.path.join(age_group_path, subdir)

            if os.path.isdir(subdir_path) and subdir.isdigit() and 0 <= int(subdir) <= 3:
                # Iterate through each image in the subdirectory
                for filename in os.listdir(subdir_path):
                    if filename.endswith(('.png', '.jpg')):
                        # Construct the new filename
                        new_filename = f"{subdir}_{os.path.splitext(filename)[0]}_{suffix}{os.path.splitext(filename)[1]}"

                        # Construct the full paths
                        old_filepath = os.path.join(subdir_path, filename)
                        new_filepath = os.path.join(subdir_path, new_filename)

                        # Rename the file
                        os.rename(old_filepath, new_filepath)


