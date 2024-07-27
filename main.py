import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import time

# Function definitions
import matplotlib.pyplot as plt

def resize_with_padding(image, target_size):
    start_time = time.time()
    
    # Display original image
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    
    old_size = image.shape[:2]
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))
    
    new_image = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_size[0]) // 2
    x_offset = (target_size - new_size[1]) // 2
    new_image[y_offset:y_offset + new_size[0], x_offset:x_offset + new_size[1]] = resized_image
    
    # Display resized image
    # plt.subplot(1, 2, 2)
    # plt.title("Resized Image")
    # plt.imshow(new_image, cmap='gray')
    # plt.axis('off')
    
    # plt.show()
    
    end_time = time.time()
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

def get_run_lengths_M(line, max_gray):
    start_time = time.time()
    run_lengths = [0] * max_gray
    length = 1
    for i in range(1, len(line)):
        if line[i] == line[i - 1]:
            length += 1
        else:
            run_lengths[line[i - 1]] += length
            length = 1
    run_lengths[line[-1]] += length
    end_time = time.time()
    return run_lengths

def extract_glrlm_features_M(image):
    start_time = time.time()
    max_gray = 256
    all_run_lengths = []
    for direction in [0, 1, 2, 3]:
        if direction == 0:
            for row in image:
                run_lengths = get_run_lengths_M(row, max_gray)
                all_run_lengths.append(run_lengths)
        elif direction == 1:
            for col in image.T:
                run_lengths = get_run_lengths_M(col, max_gray)
                all_run_lengths.append(run_lengths)
        elif direction == 2:
            for offset in range(-image.shape[0] + 1, image.shape[1]):
                diag = image.diagonal(offset)
                run_lengths = get_run_lengths_M(diag, max_gray)
                all_run_lengths.append(run_lengths)
        elif direction == 3:
            for offset in range(-image.shape[0] + 1, image.shape[1]):
                anti_diag = np.fliplr(image).diagonal(offset)
                run_lengths = get_run_lengths_M(anti_diag, max_gray)
                all_run_lengths.append(run_lengths)
    flat_run_lengths = np.concatenate(all_run_lengths)
    total_pixels = image.size
    SRE = np.sum((flat_run_lengths / total_pixels) ** 2)
    LRE = np.sum(flat_run_lengths ** 2) / total_pixels
    GLN = np.sum(np.sum(np.array(all_run_lengths), axis=0) ** 2) / (total_pixels ** 2)
    RLN = np.sum(np.sum(np.array(all_run_lengths), axis=1) ** 2) / (total_pixels ** 2)
    RP = np.sum(flat_run_lengths) / total_pixels
    end_time = time.time()
    return np.array([SRE, LRE, GLN, RLN, RP])

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

# Extract GLRLM features
def extract_features(images):
    start_time = time.time()
    glrlm_features_M = []
    total_images = len(images)
    for i, img in enumerate(images):
        glrlm_feat_M = extract_glrlm_features_M(img)
        glrlm_features_M.append(glrlm_feat_M)
        if (i + 1) % 100 == 0:
            print(f"Extracted features for {i + 1}/{total_images} images")
    end_time = time.time()
    print(f"extract_features execution time: {end_time - start_time:.4f} seconds")
    return glrlm_features_M

print("Starting feature extraction...")
start_time = time.time()
X_train_glrlm_features_M = extract_features(X_train)
X_val_glrlm_features_M = extract_features(X_val)
X_test_glrlm_features_M = extract_features(X_test)
end_time = time.time()
print(f"Total feature extraction time: {end_time - start_time:.4f} seconds")

# Ensure all feature vectors have the same length
feature_lengths = [len(fv) for fv in X_train_glrlm_features_M]
print(f"Feature lengths: {feature_lengths}")
if len(set(feature_lengths)) != 1:
    raise ValueError("Feature vectors have inconsistent lengths.")

# Normalize features
def normalize_features(features_list):
    start_time = time.time()
    all_features = np.array(features_list)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    end_time = time.time()
    return scaled_features

print("Starting feature normalization...")
start_time = time.time()
X_train_glrlm_features_M = normalize_features(X_train_glrlm_features_M)
X_val_glrlm_features_M = normalize_features(X_val_glrlm_features_M)
X_test_glrlm_features_M = normalize_features(X_test_glrlm_features_M)
end_time = time.time()
print(f"Feature normalization time: {end_time - start_time:.4f} seconds")

# Train SVM classifier
print("Starting SVM training...")
start_time = time.time()
clf = SVC(kernel='linear')
clf.fit(X_train_glrlm_features_M, y_train)
end_time = time.time()
print(f"SVM training time: {end_time - start_time:.4f} seconds")

# Evaluate classifier
print("Starting classifier evaluation...")
start_time = time.time()
y_val_pred = clf.predict(X_val_glrlm_features_M)
y_test_pred = clf.predict(X_test_glrlm_features_M)
end_time = time.time()
print(f"Classifier evaluation time: {end_time - start_time:.4f} seconds")

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))
