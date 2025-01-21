import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
MODEL_PATH = 'mnist_cnn_model.h5'  # Ensure you have this file in the same directory
model = load_model(MODEL_PATH)

# Title and description
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) to classify it using the trained model.")

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (if necessary)
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the model
    return image_array

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Display the prediction
    st.write(f"Predicted Digit: {predicted_digit}")

st.write("---")
st.write("Ensure the image is a clear, single digit for best results.")
