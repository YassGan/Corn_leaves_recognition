import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def predict_image_class(image_path, model_filename, scaler_filename):
    # Load the SVM model and StandardScaler object from files
    with open(model_filename, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open(scaler_filename, 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the target size
    target_size = 256
    old_size = gray_image.shape[:2]
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

    return predicted_class[0], image

class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier")
        self.master.geometry("600x400")

        self.model_filename = 'svm_lbp_model_Balanced_Data.pkl'
        self.scaler_filename = 'scaler_Balanced_Data.pkl'

        self.create_widgets()

    def create_widgets(self):
        # Frame for button
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(side=tk.TOP, pady=10)

        # Open Image button
        self.open_button = tk.Button(self.button_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(side=tk.LEFT, padx=5)

        # Image display
        self.image_label = tk.Label(self.master)
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Prediction result
        self.result_label = tk.Label(self.master, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.predict()

    def display_image(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((400, 300))  # Resize image to fit in the window
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference

    def predict(self):
        try:
            predicted_class, _ = predict_image_class(self.image_path, self.model_filename, self.scaler_filename)
            self.result_label.config(text=f"Predicted class: {predicted_class}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

def main():
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()