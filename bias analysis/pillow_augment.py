import os
from PIL import Image, ImageOps


def horizontal_flip(image):
    return ImageOps.mirror(image)


def zoom_image(image, zoom_factor=1.4):
    width, height = image.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    if new_width <= 0 or new_height <= 0:
        raise ValueError("Zoom factor too high")

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    if right <= left or bottom <= top:
        raise ValueError("Calculated crop invalid")

    image = image.crop((left, top, right, bottom))
    return image.resize((width, height), Image.LANCZOS)


def process_images(input_folder, output_folder):
    """Process images: zoom and transpose"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            try:
                # Apply horizontal flip
                img = horizontal_flip(img)

                # Apply zoom
                img = zoom_image(img)

                output_path = os.path.join(output_folder, filename)
                img.save(output_path)
            except ValueError as e:
                print(f"Skipping {filename}: {e}")


input_folder = '../data/old data/labeled/age/elder/3'
output_folder = '../data/old data/labeled/age/elder/3/augmented'
process_images(input_folder, output_folder)
