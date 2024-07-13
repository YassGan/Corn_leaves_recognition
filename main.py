import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Specify the folder containing the images
folder_path = './data/'

# Get the list of files in the folder
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Print the list of file names
print("Files in the folder:")
for file_name in file_names:
    print(file_name)

# Load the first image from the list (example)
image_path = os.path.join(folder_path, file_names[0])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError("Image file not found. Please check the path.")

# Convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute HOG features and HOG image on the grayscale image
hog_features, hog_image = hog(image_gray, orientations=1, pixels_per_cell=(12, 12),
                              cells_per_block=(8, 8), block_norm='L1',
                              visualize=True, transform_sqrt=True)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Display the original image and HOG visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('HOG Visualization (Shape)')
plt.imshow(hog_image_rescaled, cmap='gray')  # Display HOG image
plt.axis('off')

plt.tight_layout()
plt.show()
