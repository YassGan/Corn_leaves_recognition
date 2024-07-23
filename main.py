import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# Train and evaluate SVM
def train_evaluate_svm(X_train, y_train, X_val, y_val):
    clf = SVC(kernel='linear', random_state=42)  # Linear kernel SVM
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return clf, accuracy

# Extract HOG features from images
def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
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

def extract_features(images, part_name):
    hog_features = []

    # Start timing HOG feature extraction
    print(f"Extracting HOG features for {part_name}...")
    start_hog_time = time.time()
    for img in images:
        hog_feat = extract_hog_features(img)
        hog_features.append(hog_feat)
    end_hog_time = time.time()
    hog_time = end_hog_time - start_hog_time

    # Convert list to numpy array for consistency
    hog_features = np.array(hog_features)
    print(f"HOG feature extraction for {part_name} completed.\n")
    return hog_features, hog_time



def normalize_features(features, part_name):
    print(f"Normalizing features for {part_name}...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print(f"Feature normalization for {part_name} completed.\n")

    return scaled_features


# Extract and normalize features
X_train_hog_features, train_hog_time = extract_features(X_train, "training")
X_val_hog_features, val_hog_time = extract_features(X_val, "validation")
X_test_hog_features, test_hog_time = extract_features(X_test, "test")

# Normalize features
X_train_hog_features = normalize_features(X_train_hog_features, "training")
X_val_hog_features = normalize_features(X_val_hog_features, "validation")
X_test_hog_features = normalize_features(X_test_hog_features, "test")

print("Training SVM model...")
# Train and evaluate SVM on HOG features
svm_hog_model, hog_accuracy = train_evaluate_svm(X_train_hog_features, y_train, X_val_hog_features, y_val)
print(f'HOG Features SVM Accuracy: {hog_accuracy:.4f}\n')

print("Evaluating model on test set...")
# Optionally, evaluate the model on test set if needed
y_test_pred = svm_hog_model.predict(X_test_hog_features)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy with HOG Features: {test_accuracy:.4f}\n')

# Print timing information
print("Time taken for HOG feature extraction on training set:", train_hog_time)
print("Time taken for HOG feature extraction on validation set:", val_hog_time)
print("Time taken for HOG feature extraction on test set:", test_hog_time)
