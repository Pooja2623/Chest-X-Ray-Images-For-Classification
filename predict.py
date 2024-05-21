import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define image dimensions
img_width, img_height = 150, 150

# Load the trained model
model = load_model('model.h5')

# Define class labels
class_labels = ['Normal','Pneumonia']

# Function to preprocess the uploaded image
def preprocess_image(img):
    # Convert the image to RGB if it's grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize the image to the required input shape
    img = img.resize((img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values
    return img


# Function to make predictions
def predict(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]
    return class_labels[predicted_class_index], confidence

def main():
    st.title('Chest X-ray Classifier')
    st.write('Upload a chest X-ray image to classify whether it is Normal or Pneumonia.')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        predicted_class, confidence = predict(image)
        st.write('Prediction:', predicted_class)
        st.write('Confidence:', confidence)

if __name__ == '__main__':
    main()
