import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.feature import hog


# Load and preprocess dataset
def load_images(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

data_dir = './smallData'
images, labels = load_images(data_dir)

print(labels)





# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def extract_hog_features(image):
    features, _ = hog(image, block_norm='L2-Hys', pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

def extract_features(images):
    features = []
    for img in images:
        hog_feat = extract_hog_features(img)
        features.append(hog_feat)
    return features

X_train_features = extract_features(X_train)
X_val_features = extract_features(X_val)
X_test_features = extract_features(X_test)

print("Sample of HOG features extracted from X_train:")
print(len(X_train_features))  # Print a sample of the extracted HOG features

print("Sample of HOG features extracted from X_val:")
print(len(X_val_features))  # Print a sample of the extracted HOG features

print("Sample of HOG features extracted from X_test:")
print(len(X_test_features))  # Print a sample of the extracted HOG features


