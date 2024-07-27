import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import time
import csv
import multiprocessing
from functools import partial
from tqdm import tqdm

# Function definitions
def resize_with_padding(image, target_size):
    old_size = image.shape[:2]
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))
    
    new_image = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_size[0]) // 2
    x_offset = (target_size - new_size[1]) // 2
    new_image[y_offset:y_offset + new_size[0], x_offset:x_offset + new_size[1]] = resized_image
    
    return new_image

def load_images(data_dir, image_size=256):
    start_time = time.time()
    images = []
    labels = []
    total_images = sum([len(files) for r, d, files in os.walk(data_dir)])
    processed_images = 0
    try:
        for label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = resize_with_padding(img, image_size)
                    images.append(img)
                    labels.append(label)
                else:
                    print(f"Warning: Unable to read image {img_path}")
                processed_images += 1
                if processed_images % 100 == 0:
                    print(f"Processed {processed_images}/{total_images} images")
    except Exception as e:
        print(f"Error loading images from {data_dir}: {e}")
    end_time = time.time()
    print(f"load_images execution time: {end_time - start_time:.4f} seconds")
    return images, labels

def extract_glrlm_features_M(image):
    # Ensure image is of type uint8
    image = image.astype(np.uint8)
    
    # Calculate max_gray safely
    max_gray = int(np.max(image)) + 1
    max_run = np.max(image.shape)
    
    # Initialize GLRLM with a minimum size of 1 for each dimension
    glrlm = np.zeros((max(max_gray, 1), max(max_run, 1), 4), dtype=int)
    
    # Calculate GLRLM for 0, 45, 90, 135 degrees
    for angle in range(4):
        if angle == 0:
            for row in image:
                run_length = 1
                for i in range(1, len(row)):
                    if row[i] == row[i-1]:
                        run_length += 1
                    else:
                        glrlm[row[i-1], run_length-1, angle] += 1
                        run_length = 1
                glrlm[row[-1], run_length-1, angle] += 1
        elif angle == 1:
            for col in image.T:
                run_length = 1
                for i in range(1, len(col)):
                    if col[i] == col[i-1]:
                        run_length += 1
                    else:
                        glrlm[col[i-1], run_length-1, angle] += 1
                        run_length = 1
                glrlm[col[-1], run_length-1, angle] += 1
        elif angle == 2:
            for diag in [image.diagonal(i) for i in range(-image.shape[0]+1, image.shape[1])]:
                run_length = 1
                for i in range(1, len(diag)):
                    if diag[i] == diag[i-1]:
                        run_length += 1
                    else:
                        glrlm[diag[i-1], run_length-1, angle] += 1
                        run_length = 1
                glrlm[diag[-1], run_length-1, angle] += 1
        else:
            flipped = np.fliplr(image)
            for diag in [flipped.diagonal(i) for i in range(-flipped.shape[0]+1, flipped.shape[1])]:
                run_length = 1
                for i in range(1, len(diag)):
                    if diag[i] == diag[i-1]:
                        run_length += 1
                    else:
                        glrlm[diag[i-1], run_length-1, angle] += 1
                        run_length = 1
                glrlm[diag[-1], run_length-1, angle] += 1
    
    # Normalize GLRLM
    glrlm_sum = np.sum(glrlm)
    if glrlm_sum == 0:
        return np.zeros(11)  # Return zero features if GLRLM is empty
    p = glrlm / glrlm_sum

    # Calculate features
    g_indices = np.arange(max_gray)[:, np.newaxis, np.newaxis]
    r_indices = np.arange(max_run)[np.newaxis, :, np.newaxis]
    
    sre = np.sum(p / (r_indices**2 + 1e-6))
    lre = np.sum(p * r_indices**2)
    gln = np.sum(np.sum(p, axis=1)**2)
    rln = np.sum(np.sum(p, axis=0)**2)
    rp = np.sum(p)
    lgre = np.sum(p / (g_indices**2 + 1e-6))
    hgre = np.sum(p * g_indices**2)
    srlge = np.sum(p / ((r_indices**2 + 1e-6) * (g_indices**2 + 1e-6)))
    srhge = np.sum(p * g_indices**2 / (r_indices**2 + 1e-6))
    lrlge = np.sum(p * r_indices**2 / (g_indices**2 + 1e-6))
    lrhge = np.sum(p * r_indices**2 * g_indices**2)

    return np.array([sre, lre, gln, rln, rp, lgre, hgre, srlge, srhge, lrlge, lrhge])

def extract_features_parallel(images, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        glrlm_features_M = list(tqdm(pool.imap(extract_glrlm_features_M, images), total=len(images)))
    
    end_time = time.time()
    print(f"Parallel feature extraction time: {end_time - start_time:.4f} seconds")
    return glrlm_features_M

def normalize_features(features_list):
    start_time = time.time()
    all_features = np.array(features_list)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    end_time = time.time()
    return scaled_features

# Main execution
if __name__ == "__main__":
    # Load and preprocess dataset
    print("Starting dataset loading...")
    start_time = time.time()
    data_dir = './AllData'
    images, labels = load_images(data_dir)
    end_time = time.time()
    print(f"Total dataset loading time: {end_time - start_time:.4f} seconds")

    # Split dataset
    print("Starting dataset split...")
    start_time = time.time()
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    end_time = time.time()
    print(f"Dataset split time: {end_time - start_time:.4f} seconds")

    # Initialize CSV file for logging results
    csv_file = 'experiment_results.csv'
    csv_headers = [
        'Feature Extraction Technique', 'Feature Params', 'SVM Params', 
        'Normalization Method', 'Normalization Params', 
        'Feature Extraction Time', 'Normalization Time', 'Training Time', 
        'Validation Accuracy', 'Test Accuracy'
    ]

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)

    # List of SVM parameters
    svm_params_list = [
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 100.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'auto'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},  
        {'kernel': 'linear'}, 
        {'kernel': 'poly', 'degree': 2, 'C': 1.0},
        {'kernel': 'poly', 'degree': 3, 'C': 1.0},
        {'kernel': 'linear', 'C': 0.1},
        {'kernel': 'linear', 'C': 10.0},
        {'kernel': 'sigmoid', 'C': 1.0}
    ]

    # Experiment loop
    for svm_params in svm_params_list:
        # Extract GLRLM features
        print("Starting parallel feature extraction...")
        start_time = time.time()
        X_train_glrlm_features_M = extract_features_parallel(X_train)
        X_val_glrlm_features_M = extract_features_parallel(X_val)
        X_test_glrlm_features_M = extract_features_parallel(X_test)
        feature_extraction_time = time.time() - start_time

        # Normalize features
        print("Starting feature normalization...")
        start_time = time.time()
        X_train_glrlm_features_M = normalize_features(X_train_glrlm_features_M)
        X_val_glrlm_features_M = normalize_features(X_val_glrlm_features_M)
        X_test_glrlm_features_M = normalize_features(X_test_glrlm_features_M)
        normalization_time = time.time() - start_time

        # Train SVM classifier
        print("Starting SVM training...")
        start_time = time.time()
        clf = SVC(**svm_params)
        clf.fit(X_train_glrlm_features_M, y_train)
        training_time = time.time() - start_time

        # Evaluate classifier
        print("Starting classifier evaluation...")
        start_time = time.time()
        y_val_pred = clf.predict(X_val_glrlm_features_M)
        y_test_pred = clf.predict(X_test_glrlm_features_M)
        evaluation_time = time.time() - start_time

        # Collect metrics
        validation_accuracy = accuracy_score(y_val, y_val_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Log results
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'GLRLM', {}, svm_params, 'StandardScaler', {}, 
                feature_extraction_time, normalization_time, training_time, 
                validation_accuracy, test_accuracy
            ])

        print(f"Feature Extraction Time: {feature_extraction_time:.4f} seconds")
        print(f"Normalization Time: {normalization_time:.4f} seconds")
        print(f"Training Time: {training_time:.4f} seconds")
        print(f"Validation Accuracy: {validation_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")