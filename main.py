import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load your pre-trained model (replace with your model's path)
model = load_model('CNN_scratch.h5')
model2 = load_model('model.h5')
model3 = load_model('CNN_Dense_NN.h5')
model4 = load_model('vgg-16.h5')
model5 = load_model('inception_v3.h5')


def compute_metrics(predict):
    score = abs(predict - 0.5) * 100
    return round(score[0][0] * 2, 2)


# Define a function to preprocess the image as required by your model
def preprocess_image(img, target_size):
    img = img.resize(target_size)  # Resize the image to the target size
    img = np.array(img)  # Convert the image to a NumPy array
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Define a function to make predictions
def predict(image, model):
    if model != model5:
        processed_image = preprocess_image(image, target_size=(64, 64))  # Ensure the target size matches your model input
    else:
        processed_image = preprocess_image(image, target_size=(75, 75))  # Ensure the target size matches your model input
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
    st.write("Classifiers output:")
    prediction = predict(image, model)
    prediction2 = predict(image, model2)
    prediction3 = predict(image, model3)
    prediction4 = predict(image, model4)
    prediction5 = predict(image, model5)
    print(prediction)
    print(prediction2)
    print(prediction3)
    print(prediction4)
    print(prediction5)

    # Assuming the model returns a class index, you might need to map it to class names
    class_name = ""
    if prediction[0] > 0.5:
        class_name = "Parasitized"
    else:
        class_name = "Uninfected"

    st.write(f"CNN Prediction: {compute_metrics(prediction)}, {class_name}")
    if prediction2[0] > 0.5:
        class_name = "Parasitized"
    else:
        class_name = "Uninfected"
    st.write(f"Mobile Net Prediction: {compute_metrics(prediction2)}, {class_name}")
    if prediction3[0] > 0.5:
        class_name = "Parasitized"
    else:
        class_name = "Uninfected"
    st.write(f"CNN Dense Prediction: {compute_metrics(prediction3)}, {class_name}")
    if prediction4[0] > 0.5:
        class_name = "Parasitized"
    else:
        class_name = "Uninfected"
    st.write(f"VGG-16 Prediction: {compute_metrics(prediction4)}, {class_name}")
    if prediction5[0] > 0.5:
        class_name = "Parasitized"
    else:
        class_name = "Uninfected"
    st.write(f"Inception V3: {compute_metrics(prediction5)}, {class_name}")

