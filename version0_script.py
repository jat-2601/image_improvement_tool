import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import zipfile
import os
from huggingface_hub import hf_hub_download

# Function to download the model from Hugging Face
def download_model():
    model_repo = "jayyap/upscale"  # Replace with the correct model repo if needed
    model_filename = "swinir_model.h5"  # Adjust based on the actual model file name
    model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
    return model_path

# Load the SwinIR model
def load_swinir_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Enhance image using SwinIR
def enhance_image_with_swinir(image, model):
    image = image.resize((image.width // 4, image.height // 4))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    enhanced_image = model.predict(image_array)
    enhanced_image = np.clip(enhanced_image[0] * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)

# Streamlit app layout
st.title("Image Enhancement Dashboard with SwinIR")
st.write("Upload low-resolution images to enhance their quality!")

# Download and load the model
model_path = download_model()
model = load_swinir_model(model_path)

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Processing images
if model and uploaded_files:
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        enhanced_image = enhance_image_with_swinir(image, model)

        enhanced_image_path = os.path.join(temp_dir, f"enhanced_{os.path.splitext(uploaded_file.name)[0]}.png")
        enhanced_image.save(enhanced_image_path)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image.resize((500, 750)), caption="Original Image", use_column_width=True)
        with col2:
            st.image(enhanced_image.resize((500, 750)), caption="Enhanced Image", use_column_width=True)

    zip_filename = "enhanced_images.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in os.listdir(temp_dir):
            zipf.write(os.path.join(temp_dir, file), file)

    with open(zip_filename, 'rb') as f:
        st.download_button(
            label="Download All Enhanced Images as ZIP",
            data=f,
            file_name=zip_filename,
            mime="application/zip"
        )

    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    st.success("Enhancement Complete!")
else:
    st.info("Please ensure the model is loaded correctly and upload images to get started.")
