import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import time

# Train and evaluate SVM
def train_evaluate_svm(X_train, y_train, X_val, y_val):
    clf = SVC(kernel='rbf', C=10)  # RBF kernel SVM
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return clf, accuracy, y_pred

# Train and evaluate KNN
def train_evaluate_knn(X_train, y_train, X_val, y_val, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return knn, accuracy, y_pred

# Train and evaluate Random Forest
def train_evaluate_rf(X_train, y_train, X_val, y_val, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return rf, accuracy, y_pred

# Train and evaluate CNN
def train_evaluate_cnn(X_train, y_train, X_val, y_val, input_shape, num_classes, epochs=10, batch_size=32):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1)
    
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    return model, accuracy, y_pred_classes

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
    return np.array(images), np.array(labels)

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

# Main execution
if __name__ == "__main__":
    data_dir = './DataToWorkWith'
    images, labels = load_images(data_dir)

    print("Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print("Dataset split into training, validation, and test sets.\n")

    # Extract and normalize features for traditional ML models
    X_train_lbp_features, train_lbp_time = extract_features(X_train, "training")
    X_val_lbp_features, val_lbp_time = extract_features(X_val, "validation")
    X_test_lbp_features, test_lbp_time = extract_features(X_test, "test")

    X_train_lbp_features = normalize_features(X_train_lbp_features, "training")
    X_val_lbp_features = normalize_features(X_val_lbp_features, "validation")
    X_test_lbp_features = normalize_features(X_test_lbp_features, "test")

    # Prepare data for CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1) / 255.0
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1) / 255.0
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1) / 255.0

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)

    # Convert to categorical for CNN
    y_train_categorical = to_categorical(y_train_encoded)
    y_val_categorical = to_categorical(y_val_encoded)
    y_test_categorical = to_categorical(y_test_encoded)

    # Train and evaluate models
    models = {
        "SVM": (train_evaluate_svm, (X_train_lbp_features, y_train_encoded, X_val_lbp_features, y_val_encoded)),
        "KNN": (train_evaluate_knn, (X_train_lbp_features, y_train_encoded, X_val_lbp_features, y_val_encoded)),
        "Random Forest": (train_evaluate_rf, (X_train_lbp_features, y_train_encoded, X_val_lbp_features, y_val_encoded)),
        "CNN": (train_evaluate_cnn, (X_train_cnn, y_train_categorical, X_val_cnn, y_val_categorical, X_train_cnn.shape[1:], len(np.unique(y_train))))
    }

    results = {}

    for model_name, (train_evaluate_func, args) in models.items():
        print(f"\nTraining {model_name} model...")
        model, val_accuracy, val_predictions = train_evaluate_func(*args)
        print(f'{model_name} Validation Accuracy: {val_accuracy:.4f}')

        # Evaluate on test set
        if model_name == "CNN":
            test_predictions = np.argmax(model.predict(X_test_cnn), axis=1)
            test_accuracy = accuracy_score(y_test_encoded, test_predictions)
        else:
            test_predictions = model.predict(X_test_lbp_features)
            test_accuracy = accuracy_score(y_test_encoded, test_predictions)
        print(f'{model_name} Test Accuracy: {test_accuracy:.4f}')

        results[model_name] = {
            "model": model,
            "val_predictions": val_predictions,
            "test_predictions": test_predictions,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy
        }

    # Print detailed results
    for model_name, result in results.items():
        print(f"\n--- {model_name} Results ---")
        print(f"Validation Accuracy: {result['val_accuracy']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        
        print("\nClassification Report for Validation Set:")
        print(classification_report(y_val_encoded, result['val_predictions']))
        print("Confusion Matrix for Validation Set:")
        print(confusion_matrix(y_val_encoded, result['val_predictions']))
        
        print("\nClassification Report for Test Set:")
        print(classification_report(y_test_encoded, result['test_predictions']))
        print("Confusion Matrix for Test Set:")
        print(confusion_matrix(y_test_encoded, result['test_predictions']))

    # Print timing information
    print("\nTime taken for LBP feature extraction:")
    print(f"Training set: {train_lbp_time:.2f} seconds")
    print(f"Validation set: {val_lbp_time:.2f} seconds")
    print(f"Test set: {test_lbp_time:.2f} seconds")