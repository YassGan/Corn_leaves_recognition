import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
from itertools import product
import csv
from datetime import datetime

# Train and evaluate SVM
def train_evaluate_svm(X_train, y_train, X_val, y_val, kernel, C):
    clf = SVC(kernel=kernel, C=C, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return clf, accuracy

# Extract GLCM features from images
def extract_glcm_features(image, distances, angles):
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    features = []
    for prop in properties:
        features.extend(graycoprops(glcm, prop).flatten())
    return np.array(features)








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






data_dir = './Alldata'
images, labels = load_images(data_dir)

print("Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("Dataset split into training, validation, and test sets.\n")

def extract_features(images, distances, angles):
    glcm_features = []
    for img in images:
        glcm_feat = extract_glcm_features(img, distances, angles)
        glcm_features.append(glcm_feat)
    return np.array(glcm_features)

def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Define parameter ranges
glcm_distances = [[1], [2], [3]]
glcm_angles = [[0], [np.pi/4], [np.pi/2], [3*np.pi/4]]
svm_kernels = ['linear', 'rbf', 'poly']
svm_Cs = [0.1, 1, 10, 100]

best_accuracy = 0
best_params = {}

# Create CSV file for logging results
csv_filename = f"glcm_svm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_header = ['Distances', 'Angles', 'Kernel', 'C', 'Extraction Time', 'Training Time', 'Validation Accuracy']

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

    for distances, angles, kernel, C in product(glcm_distances, glcm_angles, svm_kernels, svm_Cs):
        print(f"\nTesting parameters: distances={distances}, angles={angles}, kernel={kernel}, C={C}")
        
        # Extract features
        start_time = time.time()
        X_train_glcm = extract_features(X_train, distances, angles)
        X_val_glcm = extract_features(X_val, distances, angles)
        extraction_time = time.time() - start_time
        print(f"Feature extraction time: {extraction_time:.2f} seconds")

        # Normalize features
        X_train_glcm_norm = normalize_features(X_train_glcm)
        X_val_glcm_norm = normalize_features(X_val_glcm)

        # Train and evaluate
        start_time = time.time()
        _, accuracy = train_evaluate_svm(X_train_glcm_norm, y_train, X_val_glcm_norm, y_val, kernel, C)
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Log results to CSV
        csv_writer.writerow([str(distances), str(angles), kernel, C, extraction_time, training_time, accuracy])
        csvfile.flush()  # Ensure data is written to file

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'distances': distances,
                'angles': angles,
                'kernel': kernel,
                'C': C
            }

print("\nBest parameters:")
print(f"GLCM distances: {best_params['distances']}")
print(f"GLCM angles: {best_params['angles']}")
print(f"SVM kernel: {best_params['kernel']}")
print(f"SVM C: {best_params['C']}")
print(f"Best validation accuracy: {best_accuracy:.4f}")

# Test the best model on the test set
print("\nEvaluating best model on test set...")
X_train_glcm_best = extract_features(X_train, best_params['distances'], best_params['angles'])
X_test_glcm_best = extract_features(X_test, best_params['distances'], best_params['angles'])

X_train_glcm_best_norm = normalize_features(X_train_glcm_best)
X_test_glcm_best_norm = normalize_features(X_test_glcm_best)

best_model, _ = train_evaluate_svm(X_train_glcm_best_norm, y_train, X_test_glcm_best_norm, y_test, best_params['kernel'], best_params['C'])
y_test_pred = best_model.predict(X_test_glcm_best_norm)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy with best parameters: {test_accuracy:.4f}')

# Append test results to CSV
with open(csv_filename, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Test Results', '', '', '', '', '', ''])
    csv_writer.writerow([str(best_params['distances']), str(best_params['angles']), best_params['kernel'], best_params['C'], '', '', test_accuracy])