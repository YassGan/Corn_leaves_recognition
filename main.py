
import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from skimage.filters import gabor

import time





## This function resizes the input images while preserving their carecteristics
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






# Load and preprocess dataset
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

data_dir = './smallData'
images, labels = load_images(data_dir)

print(labels)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

def extract_lbp_features(image, radius=3, n_points=24):
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










##The gabor simple approach
# def extract_gabor_features(image):
#     frequencies = [0.1, 0.3, 0.5, 0.7, 0.9]
#     features = []
#     for frequency in frequencies:
#         filt_real, filt_imag = gabor(image, frequency=frequency)
#         features.append(filt_real.mean())
#         features.append(filt_real.var())
#         features.append(filt_imag.mean())
#         features.append(filt_imag.var())
#     return np.array(features)


def extract_gabor_features(image):
    frequencies = [0.1, 0.3, 0.5, 0.7, 0.9]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # orientations
    bandwidth = 1
    sigma_x = 4
    sigma_y = 4
    n_stds = 3
    offset = 0
    features = []
    for frequency in frequencies:
        for theta in thetas:
            filt_real, filt_imag = gabor(image, frequency=frequency, theta=theta, bandwidth=bandwidth, sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds, offset=offset)
            features.append(filt_real.mean())
            features.append(filt_real.var())
            features.append(filt_imag.mean())
            features.append(filt_imag.var())
    return np.array(features)





def extract_features(images):
    hog_features = []
    lbp_features = []
    glcm_features = []
    glrlm_features_V = []
    glrlm_features_M = []
    gabor_features = []


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

    # Start timing GLRLM feature extraction (Vector)
    start_glrlm_time_V = time.time()
    for img in images:
        glrlm_feat_V = extract_glrlm_features_V(img)
        glrlm_features_V.append(glrlm_feat_V)
    end_glrlm_time_V = time.time()
    glrlm_time_V = end_glrlm_time_V - start_glrlm_time_V

    # Start timing GLRLM feature extraction (Matrix)
    start_glrlm_time_M = time.time()
    for img in images:
        glrlm_feat_M = extract_glrlm_features_M(img)
        glrlm_features_M.append(glrlm_feat_M)
    end_glrlm_time_M = time.time()
    glrlm_time_M = end_glrlm_time_M - start_glrlm_time_M



    # Start timing Gabor feature extraction
    start_gabor_time = time.time()
    for img in images:
        gabor_feat = extract_gabor_features(img)
        gabor_features.append(gabor_feat)
    end_gabor_time = time.time()
    gabor_time = end_gabor_time - start_gabor_time

    return hog_features, lbp_features, glcm_features, gabor_features, glrlm_features_V, glrlm_features_M, hog_time, lbp_time, glcm_time, gabor_time, glrlm_time_V, glrlm_time_M




from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Choose either MinMaxScaler or StandardScaler
### This function normalizes the features extracted values after the treatment of extraction 

scaler = MinMaxScaler()  # or StandardScaler()

def normalize_features(features_list):
    all_features = np.vstack(features_list)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    
    split_idx = 0
    normalized_features = []
    for features in features_list:
        next_idx = split_idx + features.shape[0]
        normalized_features.append(scaled_features[split_idx:next_idx])
        split_idx = next_idx
    return normalized_features






# Extract features
X_train_hog_features, X_train_lbp_features, X_train_glcm_features, X_train_gabor_features, X_train_glrlm_features_V, X_train_glrlm_features_M, train_hog_time, train_lbp_time, train_glcm_time, train_gabor_time, train_glrlm_time_V, train_glrlm_time_M = extract_features(X_train)
X_val_hog_features, X_val_lbp_features, X_val_glcm_features, X_val_gabor_features, X_val_glrlm_features_V, X_val_glrlm_features_M, val_hog_time, val_lbp_time, val_glcm_time, val_gabor_time, val_glrlm_time_V, val_glrlm_time_M = extract_features(X_val)
X_test_hog_features, X_test_lbp_features, X_test_glcm_features, X_test_gabor_features, X_test_glrlm_features_V, X_test_glrlm_features_M, test_hog_time, test_lbp_time, test_glcm_time, test_gabor_time, test_glrlm_time_V, test_glrlm_time_M = extract_features(X_test)





# Normalized extracted features 
X_train_hog_features = normalize_features(X_train_hog_features)
X_train_lbp_features = normalize_features(X_train_lbp_features)
X_train_glcm_features = normalize_features(X_train_glcm_features)
X_train_gabor_features = normalize_features(X_train_gabor_features)
X_train_glrlm_features_V = normalize_features(X_train_glrlm_features_V)
X_train_glrlm_features_M = normalize_features(X_train_glrlm_features_M)

X_val_hog_features = normalize_features(X_val_hog_features)
X_val_lbp_features = normalize_features(X_val_lbp_features)
X_val_glcm_features = normalize_features(X_val_glcm_features)
X_val_gabor_features = normalize_features(X_val_gabor_features)
X_val_glrlm_features_V = normalize_features(X_val_glrlm_features_V)
X_val_glrlm_features_M = normalize_features(X_val_glrlm_features_M)

X_test_hog_features = normalize_features(X_test_hog_features)
X_test_lbp_features = normalize_features(X_test_lbp_features)
X_test_glcm_features = normalize_features(X_test_glcm_features)
X_test_gabor_features = normalize_features(X_test_gabor_features)
X_test_glrlm_features_V = normalize_features(X_test_glrlm_features_V)
X_test_glrlm_features_M = normalize_features(X_test_glrlm_features_M)

















print("")
print("Sample of X_train_hog_features extracted from X_train:")
print(X_train_hog_features[0])
print("")


print("Sample of X_train_lbp_features extracted from X_train:")
print(X_train_lbp_features[0])
print("")

print("Sample of X_train_glcm_features extracted from X_train:")
print(X_train_glcm_features[0])
print("")

print("Sample of X_train_gabor_features extracted from X_train:")
print(X_train_gabor_features[0])
print("")

print("Sample of X_train_glrlm_features_V extracted from X_train:")
print(X_train_glrlm_features_V[0])
print("")

print("Sample of X_train_glrlm_features_M extracted from X_train:")
print(X_train_glrlm_features_M[0])
print("")

# Print timing information
print("Time taken for HOG feature extraction on training set:", train_hog_time)
print("Time taken for LBP feature extraction on training set:", train_lbp_time)
print("Time taken for GLCM feature extraction on training set:", train_glcm_time)
print("Time taken for Gabor feature extraction on training set:", train_gabor_time)
print("Time taken for GLRLM feature extraction (Vector) on training set:", train_glrlm_time_V)
print("Time taken for GLRLM feature extraction (Matrix) on training set:", train_glrlm_time_M)
print(" ")

print("Time taken for HOG feature extraction on validation set:", val_hog_time)
print("Time taken for LBP feature extraction on validation set:", val_lbp_time)
print("Time taken for GLCM feature extraction on validation set:", val_glcm_time)
print("Time taken for Gabor feature extraction on validation set:", val_gabor_time)
print("Time taken for GLRLM feature extraction (Vector) on validation set:", val_glrlm_time_V)
print("Time taken for GLRLM feature extraction (Matrix) on validation set:", val_glrlm_time_M)
print(" ")

print("Time taken for HOG feature extraction on test set:", test_hog_time)
print("Time taken for LBP feature extraction on test set:", test_lbp_time)
print("Time taken for GLCM feature extraction on test set:", test_glcm_time)
print("Time taken for Gabor feature extraction on test set:", test_gabor_time)
print("Time taken for GLRLM feature extraction (Vector) on test set:", test_glrlm_time_V)
print("Time taken for GLRLM feature extraction (Matrix) on test set:", test_glrlm_time_M)