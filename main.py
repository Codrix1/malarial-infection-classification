import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load your pre-trained model (replace with your model's path)
model = load_model('CNN_scratch.h5')
# model2 = load_model('MobileNet.h5')


# Define a function to preprocess the image as required by your model
def preprocess_image(img, target_size):
    img = img.resize(target_size)  # Resize the image to the target size
    img = np.array(img)  # Convert the image to a NumPy array
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Define a function to make predictions
def predict(image, model):
    processed_image = preprocess_image(image, target_size=(64, 64))  # Ensure the target size matches your model input
    prediction = model.predict(processed_image)
    return prediction


# Streamlit interface
st.title("Malaria Classification with Machine Learning")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Predict and display the result
    st.write("Classifying...")
    prediction = predict(image, model)
    # prediction2 = predict(image, model2)
    print(prediction)

    # Assuming the model returns a class index, you might need to map it to class names
    class_name = ""
    if prediction[0] > 0.5:
        class_name = "Parasitized"
    else:
        class_name = "Uninfected"

    st.write(f"CNN Prediction: {class_name}")
    # if prediction2[0] > 0.5:
    #     class_name = "Parasitized"
    # else:
    #     class_name = "Uninfected"
    # st.write(f"Mobile Net Prediction: {class_name}")
