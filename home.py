import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load the trained model
try:
    model = load_model('model_new.h5')  # Path to your saved model
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Title of the app
st.title("Nail Disease Classification")
st.subheader("Upload an image of a nail to classify the disease.")
st.write("This application uses a trained deep learning model to identify nail diseases based on the uploaded image.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess the image
    img = image.load_img(uploaded_file, target_size=(160, 160))  # Ensure size matches the model input
    img_array = image.img_to_array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR format
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Class names (ensure this matches your model's class order)
    class_names = ['Acral_Lentiginous_Melanoma', 'Healthy_Nail', 'Onychogryphosis',
                   'blue_finger', 'clubbing', 'pitting']

    # Display the prediction
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Disease:** {class_names[predicted_class[0]]}")

    # Show the prediction probabilities
    st.write("### Prediction Probabilities:")
    for idx, class_name in enumerate(class_names):
        st.write(f"**{class_name}:** {predictions[0][idx]:.2f}")

    # Display the confidence of the predicted class
    confidence = predictions[0][predicted_class[0]]
    st.write(f"### Confidence in Prediction: {confidence:.2f}")

# Additional Styling
st.sidebar.header("About This App")
st.sidebar.write(
    "This application allows you to classify nail diseases using a trained deep learning model. "
    "Just upload an image of a nail, and the model will predict the disease type."
)

st.sidebar.header("Instructions")
st.sidebar.write(
    "1. Upload an image of a nail in JPG or PNG format.\n"
    "2. Click on 'Classify' to get the prediction.\n"
    "3. View the predicted disease and confidence rates."
)

st.sidebar.header("Contact")
st.sidebar.write("For any inquiries, please contact: example@example.com")
