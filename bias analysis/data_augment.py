import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Output dir inside input dir
input_dir = '../data/labeled/age/young/3'
output_dir = '../data/labeled/age/young/3/augmented'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Iterate through all images
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(input_dir, filename)
        img = load_img(img_path, color_mode='grayscale') 
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # Reshape the image to (1, height, width, channels)

        # Create batches of augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug-', save_format='png'):
            i += 1
            if i > 10:  # Save augmented samples per image 
                break
