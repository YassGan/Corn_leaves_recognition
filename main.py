import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
import time

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
    features, _ = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

def extract_lbp_features(image):
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp.flatten()

def extract_glcm_features(image):
    distances = [1]
    angles = [0]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    features = []
    for prop in properties:
        features.extend(graycoprops(glcm, prop).flatten())
    
    return np.array(features)







#This glrlm features are returned as a vector
def extract_glrlm_features_V(image):
    max_gray = 256  
    run_lengths = [0] * max_gray
    
    for direction in [0, 1, 2, 3]:  # Horizontal, Vertical, Diagonal, Anti-diagonal
        if direction == 0:  
            for row in image:
                run_lengths.extend(get_run_lengths_V(row, max_gray))
        elif direction == 1:  
            for col in image.T:
                run_lengths.extend(get_run_lengths_V(col, max_gray))
        elif direction == 2:  
            for offset in range(-image.shape[0] + 1, image.shape[1]):
                diag = image.diagonal(offset)
                run_lengths.extend(get_run_lengths_V(diag, max_gray))
        elif direction == 3:  
            for offset in range(-image.shape[0] + 1, image.shape[1]):
                anti_diag = np.fliplr(image).diagonal(offset)
                run_lengths.extend(get_run_lengths_V(anti_diag, max_gray))
    
    return np.array(run_lengths)


#This function works for returning the extracted functions in a vector
def get_run_lengths_V(line, max_gray):
    run_lengths = [0] * max_gray
    length = 1
    for i in range(1, len(line)):
        if line[i] == line[i - 1]:
            length += 1
        else:
            run_lengths[line[i - 1]] += length
            length = 1
    run_lengths[line[-1]] += length
    return run_lengths










def extract_glrlm_features_M(image):
    max_gray = 256  
    all_run_lengths = []
    
    for direction in [0, 1, 2, 3]:  # Horizontal, Vertical, Diagonal, Anti-diagonal
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
    
    return all_run_lengths

def get_run_lengths_M(line, max_gray):
    run_lengths = [0] * max_gray
    length = 1
    for i in range(1, len(line)):
        if line[i] == line[i - 1]:
            length += 1
        else:
            run_lengths[line[i - 1]] += length
            length = 1
    run_lengths[line[-1]] += length
    return run_lengths




















def extract_features(images):
    hog_features = []
    lbp_features = []
    glcm_features = []
    glrlm_features = []

    # Start timing HOG feature extraction
    start_hog_time = time.time()
    for img in images:
        hog_feat = extract_hog_features(img)
        hog_features.append(hog_feat)
    end_hog_time = time.time()
    hog_time = end_hog_time - start_hog_time

    # Start timing LBP feature extraction
    start_lbp_time = time.time()
    for img in images:
        lbp_feat = extract_lbp_features(img)
        lbp_features.append(lbp_feat)
    end_lbp_time = time.time()
    lbp_time = end_lbp_time - start_lbp_time

    # Start timing GLCM feature extraction
    start_glcm_time = time.time()
    for img in images:
        glcm_feat = extract_glcm_features(img)
        glcm_features.append(glcm_feat)
    end_glcm_time = time.time()
    glcm_time = end_glcm_time - start_glcm_time

    # Start timing GLRLM feature extraction
    start_glrlm_time = time.time()
    for img in images:
        glrlm_feat = extract_glrlm_features_V(img)
        glrlm_features.append(glrlm_feat)
    end_glrlm_time = time.time()
    glrlm_time = end_glrlm_time - start_glrlm_time

    return hog_features, lbp_features, glcm_features, glrlm_features, hog_time, lbp_time, glcm_time, glrlm_time

# Extract features
X_train_hog_features, X_train_lbp_features, X_train_glcm_features, X_train_glrlm_features, train_hog_time, train_lbp_time, train_glcm_time, train_glrlm_time = extract_features(X_train)
X_val_hog_features, X_val_lbp_features, X_val_glcm_features, X_val_glrlm_features, val_hog_time, val_lbp_time, val_glcm_time, val_glrlm_time = extract_features(X_val)
X_test_hog_features, X_test_lbp_features, X_test_glcm_features, X_test_glrlm_features, test_hog_time, test_lbp_time, test_glcm_time, test_glrlm_time = extract_features(X_test)

print("Sample of X_train_hog_features extracted from X_train:")
print(X_train_hog_features[0])

print("Sample of X_train_lbp_features extracted from X_train:")
print(X_train_lbp_features[0])

print("Sample of X_train_glcm_features extracted from X_train:")
print(X_train_glcm_features[0])

print("Sample of X_train_glrlm_features extracted from X_train:")
print(X_train_glrlm_features[0])

# Print timing information
print("Time taken for HOG feature extraction on training set:", train_hog_time)
print("Time taken for LBP feature extraction on training set:", train_lbp_time)
print("Time taken for GLCM feature extraction on training set:", train_glcm_time)
print("Time taken for GLRLM feature extraction on training set:", train_glrlm_time)

print("Time taken for HOG feature extraction on validation set:", val_hog_time)
print("Time taken for LBP feature extraction on validation set:", val_lbp_time)
print("Time taken for GLCM feature extraction on validation set:", val_glcm_time)
print("Time taken for GLRLM feature extraction on validation set:", val_glrlm_time)

print("Time taken for HOG feature extraction on test set:", test_hog_time)
print("Time taken for LBP feature extraction on test set:", test_lbp_time)
print("Time taken for GLCM feature extraction on test set:", test_glcm_time)
print("Time taken for GLRLM feature extraction on test set:", test_glrlm_time)

print("The end of the code")
