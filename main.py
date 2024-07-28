import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import pickle
import matplotlib.pyplot as plt

def predict_image_class(image_path, model_filename, scaler_filename):
    # Load the SVM model and StandardScaler object from files
    with open(model_filename, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open(scaler_filename, 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    # Read the image
    image = cv2.imread(image_path)

    # Show the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the target size
    target_size = 256
    old_size = gray_image.shape[:2]  # old_size is in (height, width) format
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized_image = cv2.resize(gray_image, (new_size[1], new_size[0]))
    new_image = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_size[0]) // 2
    x_offset = (target_size - new_size[1]) // 2
    new_image[y_offset:y_offset + new_size[0], x_offset:x_offset + new_size[1]] = resized_image

    # Extract LBP features from the image
    lbp = local_binary_pattern(new_image, 16, 2, method='uniform')
    n_bins = 16 + 2
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram

    # Normalize the features using the loaded scaler
    scaled_features = loaded_scaler.transform([hist])

    # Use the loaded model to predict the class
    predicted_class = loaded_model.predict(scaled_features)

    # Show the predicted class
    plt.title(f'Predicted class: {predicted_class[0]}')
    plt.show()

# Example usage
# image_path = 'test_App/Corn_Health (15).jpg'

# image_path = 'test_App/Corn_Common_Rust (1306).jpg'


image_path = 'test_App/Corn_Blight (1143).jpg'



model_filename = 'svm_lbp_model.pkl'
scaler_filename = 'scaler.pkl'
predict_image_class(image_path, model_filename, scaler_filename)