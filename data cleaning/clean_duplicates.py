import os
from PIL import Image
import imagehash


def find_duplicates(path):
    img_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    hashes = {}
    duplicates = []

    for img in img_files:
        with Image.open(os.path.join(path, img)) as image:
            img_hash = imagehash.phash(image)
            if img_hash in hashes:
                duplicates.append(img)
                os.remove(os.path.join(path, img))  # Delete the duplicate file
            else:
                hashes[img_hash] = img

    return duplicates


img_path = './clean_angry'
duplicates = find_duplicates(img_path)

