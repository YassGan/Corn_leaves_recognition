import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

# Train and evaluate SVM
def train_evaluate_svm(X_train, y_train, X_val, y_val):
    clf = SVC(kernel='rbf', C=10)  # Linear kernel SVM
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return clf, accuracy, y_pred

# Extract LBP features from images
def extract_lbp_features(image, P=16, R=2):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    n_bins = P + 2
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

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

data_dir = './DataToWorkWith'
images, labels = load_images(data_dir)

print("Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("Dataset split into training, validation, and test sets.\n")

def extract_features(images, part_name):
    lbp_features = []

    # Start timing LBP feature extraction
    print(f"Extracting LBP features for {part_name}...")
    start_lbp_time = time.time()
    for img in images:
        lbp_feat = extract_lbp_features(img)
        lbp_features.append(lbp_feat)
    end_lbp_time = time.time()
    lbp_time = end_lbp_time - start_lbp_time

    # Convert list to numpy array for consistency
    lbp_features = np.array(lbp_features)
    print(f"LBP feature extraction for {part_name} completed.\n")
    return lbp_features, lbp_time

def normalize_features(features, part_name):
    print(f"Normalizing features for {part_name}...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print(f"Feature normalization for {part_name} completed.\n")

    return scaled_features

# Extract and normalize features
X_train_lbp_features, train_lbp_time = extract_features(X_train, "training")
X_val_lbp_features, val_lbp_time = extract_features(X_val, "validation")
X_test_lbp_features, test_lbp_time = extract_features(X_test, "test")

print("A sample of a lbp feature ", X_train_lbp_features[0])

# Normalize features
X_train_lbp_features = normalize_features(X_train_lbp_features, "training")
X_val_lbp_features = normalize_features(X_val_lbp_features, "validation")
X_test_lbp_features = normalize_features(X_test_lbp_features, "test")

print("A sample of a normalized lbp feature ", X_train_lbp_features[0])

print("Training SVM model...")
# Train and evaluate SVM on LBP features
svm_lbp_model, lbp_accuracy, val_predictions = train_evaluate_svm(X_train_lbp_features, y_train, X_val_lbp_features, y_val)
print(f'LBP Features SVM Accuracy: {lbp_accuracy:.4f}\n')

print("Evaluating model on test set...")
# Optionally, evaluate the model on test set if needed
y_test_pred = svm_lbp_model.predict(X_test_lbp_features)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy with LBP Features: {test_accuracy:.4f}\n')

# Generate and print classification report and confusion matrix
print("Classification Report for Validation Set:")
print(classification_report(y_val, val_predictions))

print("Confusion Matrix for Validation Set :")
print(confusion_matrix(y_val, val_predictions))

print("Classification Report for Test Set:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix for Test Set:")
print(confusion_matrix(y_test, y_test_pred))

# Print timing information
print("Time taken for LBP feature extraction on training set:", train_lbp_time)
print("Time taken for LBP feature extraction on validation set:", val_lbp_time)
print("Time taken for LBP feature extraction on test set:", test_lbp_time)
