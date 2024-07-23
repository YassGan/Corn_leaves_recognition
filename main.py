import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import time

data_folder_name = "./SmallData"
output_directory = 'Experiments_Results'

# Ensure the directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Train and evaluate SVM
def train_evaluate_svm(X_train, y_train, X_val, y_val, svm_params):
    clf = SVC(**svm_params)  # Pass parameters to SVC
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return clf, accuracy

# Extract HOG features from images
def extract_hog_features(image, hog_params):
    features, _ = hog(image, **hog_params, visualize=True)
    return features

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
    return images, labels

def extract_features(images, hog_params):
    hog_features = []
    start_hog_time = time.time()
    for img in images:
        hog_feat = extract_hog_features(img, hog_params)
        hog_features.append(hog_feat)
    end_hog_time = time.time()
    hog_time = end_hog_time - start_hog_time
    hog_features = np.array(hog_features)
    return hog_features, hog_time

def normalize_features(features, normalization_method):
    if normalization_method == 'standard':
        scaler = StandardScaler()
    elif normalization_method == 'minmax':
        scaler = MinMaxScaler()
    elif normalization_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported normalization method")
    
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def run_experiment(data_dir, image_sizes, hog_params_list, normalization_methods, split_sizes, svm_params_list):
    results = []  # List to store results for DataFrame
    
    for image_size in image_sizes:
        images, labels = load_images(data_dir, image_size=image_size)
        
        for split_size in split_sizes:
            X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=split_size, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            
            for hog_params in hog_params_list:
                print(f"Testing HOG parameters: {hog_params}")
                X_train_hog_features, train_hog_time = extract_features(X_train, hog_params)
                X_val_hog_features, val_hog_time = extract_features(X_val, hog_params)
                X_test_hog_features, test_hog_time = extract_features(X_test, hog_params)

                for normalization_method in normalization_methods:
                    print(f"Testing normalization method: {normalization_method}")
                    X_train_hog_features = normalize_features(X_train_hog_features, normalization_method)
                    X_val_hog_features = normalize_features(X_val_hog_features, normalization_method)
                    X_test_hog_features = normalize_features(X_test_hog_features, normalization_method)

                    for svm_params in svm_params_list:
                        print(f"Testing SVM parameters: {svm_params}")
                        start_time = time.time()
                        svm_hog_model, hog_accuracy = train_evaluate_svm(X_train_hog_features, y_train, X_val_hog_features, y_val, svm_params)
                        end_time = time.time()
                        
                        test_start_time = time.time()
                        y_test_pred = svm_hog_model.predict(X_test_hog_features)
                        test_accuracy = accuracy_score(y_test, y_test_pred)
                        test_end_time = time.time()
                        
                        # Collect results
                        results.append({
                            'Image Size': image_size,
                            'Split Size': split_size,
                            'HOG Params': str(hog_params),
                            'Normalization Method': normalization_method,
                            'SVM Params': str(svm_params),
                            'Train HOG Time': train_hog_time,
                            'Validation HOG Time': val_hog_time,
                            'Test HOG Time': test_hog_time,
                            'Train Time': end_time - start_time,
                            'Test Time': test_end_time - test_start_time,
                            'HOG Accuracy': hog_accuracy,
                            'Test Accuracy': test_accuracy
                        })
                        
                        print(f'HOG Features SVM Accuracy: {hog_accuracy:.4f}')
                        print(f'Test Accuracy with HOG Features: {test_accuracy:.4f}')
                        print("\n")
    # Save results to CSV file
    df = pd.DataFrame(results)
    output_file = os.path.join(output_directory, f'HOG_SVM_{os.path.basename(data_folder_name)}_experiment_results.csv')
    df.to_csv(output_file, index=False)

# Define parameters
image_sizes = [128, 256, 512]  # Different image sizes
hog_params_list = [
    {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2)},
    {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (2, 2)},
    {'orientations': 9, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)},
]
normalization_methods = ['standard', 'minmax', 'robust']  # Different normalization methods
split_sizes = [0.3, 0.4]  # Different data splitting sizes
svm_params_list = [
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'linear', 'C': 10},
    {'kernel': 'rbf', 'C': 1, 'gamma': 0.001},
    {'kernel': 'rbf', 'C': 1, 'gamma': 0.01},
]

# Run the experiment
run_experiment(data_folder_name, image_sizes, hog_params_list, normalization_methods, split_sizes, svm_params_list)
