import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# Function to train and evaluate SVM
def train_evaluate_svm(X_train, y_train, X_val, y_val, kernel='linear', C=1.0):
    clf = SVC(kernel=kernel, C=C, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return clf, accuracy

# Function to extract LBP features from images
def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    n_bins = P + 2
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

# Function to resize images with padding
def resize_with_padding(image, target_size):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    
    # Resize image
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))
    
    # Create a new image and place the resized image at the center
    new_image = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_size[0]) // 2
    x_offset = (target_size - new_size[1]) // 2
    new_image[y_offset:y_offset + new_size[0], x_offset:x_offset + new_size[1]] = resized_image
    
    return new_image

# Function to load images
def load_images(data_dir, image_size=256):
    images = []
    labels = []
    print("")
    print("Loading images from", data_dir, "...")
    try:
        for label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                if img is not None:
                    img = resize_with_padding(img, image_size)  # Resize and pad to the target size
                    images.append(img)
                    labels.append(label)
                else:
                    print(f"Warning: Unable to read image {img_path}")
    except Exception as e:
        print(f"Error loading images from {data_dir}: {e}")
    print(f"Loaded {len(images)} images and {len(labels)} labels.\n")
    return images, labels

# Function to extract features
def extract_features(images, P, R):
    lbp_features = []
    start_lbp_time = time.time()
    for img in images:
        lbp_feat = extract_lbp_features(img, P, R)
        lbp_features.append(lbp_feat)
    end_lbp_time = time.time()
    lbp_time = end_lbp_time - start_lbp_time
    lbp_features = np.array(lbp_features)
    return lbp_features, lbp_time

# Function to normalize features
def normalize_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

# Directory containing image data
data_dir = './Alldata'
images, labels = load_images(data_dir)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# List of parameters to test
lbps_params = [(8, 1), (16, 2)]  # (P, R)
svm_params = [('linear', 1.0), ('rbf', 1.0), ('rbf', 10.0)]

# Prepare to collect results
results = []

for P, R in lbps_params:
    for kernel, C in svm_params:
        print(f"Testing LBP (P={P}, R={R}) with SVM (kernel={kernel}, C={C})...")

        # Extract and normalize features
        X_train_lbp_features, train_lbp_time = extract_features(X_train, P, R)
        X_val_lbp_features, val_lbp_time = extract_features(X_val, P, R)
        X_test_lbp_features, test_lbp_time = extract_features(X_test, P, R)

        X_train_lbp_features = normalize_features(X_train_lbp_features)
        X_val_lbp_features = normalize_features(X_val_lbp_features)
        X_test_lbp_features = normalize_features(X_test_lbp_features)

        # Train and evaluate SVM model
        start_svm_time = time.time()
        svm_lbp_model, lbp_accuracy = train_evaluate_svm(X_train_lbp_features, y_train, X_val_lbp_features, y_val, kernel, C)
        end_svm_time = time.time()
        svm_time = end_svm_time - start_svm_time

        # Evaluate on test set
        y_test_pred = svm_lbp_model.predict(X_test_lbp_features)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Record results
        results.append({
            'LBP_P': P,
            'LBP_R': R,
            'SVM_kernel': kernel,
            'SVM_C': C,
            'Train_LBP_Time': train_lbp_time,
            'Val_LBP_Time': val_lbp_time,
            'Test_LBP_Time': test_lbp_time,
            'SVM_Time': svm_time,
            'Validation_Accuracy': lbp_accuracy,
            'Test_Accuracy': test_accuracy
        })

        print(f"Validation Accuracy: {lbp_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}\n")

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('experiment_results.csv', index=False)
print("Results saved to experiment_results.csv")
