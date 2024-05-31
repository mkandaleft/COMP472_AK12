import cv2
import matplotlib.pyplot as plt

# Demo of resizer.py on random image
# Load the image in grayscale
image_path = './data/classes/tester/unsplash_girl.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    raise ValueError("Image not found. Please check the path.")

size = (48, 48)

# Resize images using different interpolation methods
# bicubic = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
# bilinear = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
inter_area = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
inter_near = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
lanczos4 = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)

# Plot the original and resized images together
plt.figure(figsize=(15, 10))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original', fontsize=14)
plt.axis('off')

# plt.subplot(2, 3, 2)
# plt.imshow(bicubic, cmap='gray')
# plt.title('Bicubic Interpolation')
# plt.axis('off')

# plt.subplot(2, 3, 3)
# plt.imshow(bilinear, cmap='gray')
# plt.title('Bilinear Interpolation')
# plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(inter_area, cmap='gray')
plt.title('Area Interpolation', fontsize=14)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(inter_near, cmap='gray')
plt.title('Nearest Interpolation', fontsize=14)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(lanczos4, cmap='gray')
plt.title('Lanczos4 Interpolation', fontsize=14)
plt.axis('off')

plt.tight_layout()
# plt.savefig('interpolations.png', dpi=300)
plt.show()
