import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.feature import hog, local_binary_pattern



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
    features, _ = hog(image, orientations=9,pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features


def extract_lbp_features(image):
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp.flatten()






def extract_features(images):
    hog_features = []
    lbp_features = []
    for img in images:
        hog_feat = extract_hog_features(img)
        hog_features.append(hog_feat)
        
        lbp_feat = extract_lbp_features(img)
        lbp_features.append(lbp_feat)
        
    return hog_features, lbp_features
    



# Extract features
X_train_hog_features, X_train_lbp_features = extract_features(X_train)
X_val_hog_features, X_val_lbp_features = extract_features(X_val)
X_test_hog_features, X_test_lbp_features = extract_features(X_test)



print("Sample of X_train_hog_features  extracted from X_train:")
print((X_train_hog_features))  
FirstImage_Vector=X_train_hog_features[0]
print(FirstImage_Vector)

print("Sample of X_train_lbp_features  extracted from X_train:")
print((X_train_lbp_features[0])) 

print("The end of the code")
