import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import zipfile
import os

# Load the pretrained ESRGAN model from TensorFlow Hub
@st.cache_resource
def load_esrgan_model():
    model_url = "https://tfhub.dev/captain-pool/esrgan-tf2/1"  # ESRGAN model URL
    model = hub.load(model_url)
    return model

# Enhance image using ESRGAN
def enhance_image_with_esrgan(image, model):
    image = image.resize((image.width // 4, image.height // 4))  # Resize for ESRGAN input
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    enhanced_image = model(image_array)
    enhanced_image = np.clip(enhanced_image[0] * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)

# Streamlit app layout
st.title("Image Enhancement Dashboard with ESRGAN")
st.write("Upload low-resolution images to enhance their quality!")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True)

# Load the ESRGAN model
model = load_esrgan_model()

# Processing images
if uploaded_files:
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        # Enhance the image using ESRGAN
        enhanced_image = enhance_image_with_esrgan(image, model)

        # Save enhanced images to the temporary directory
        enhanced_image.save(os.path.join(temp_dir, f"enhanced_{uploaded_file.name.split('.')[0]}.png"))

        # Show original and enhanced images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image.resize((500, 750)), caption="Original Image", use_column_width=True)
        with col2:
            st.image(enhanced_image.resize((500, 750)), caption="Enhanced Image", use_column_width=True)

    # Create a ZIP file containing all enhanced images
    zip_filename = "enhanced_images.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    # Download button for the ZIP file
    with open(zip_filename, 'rb') as f:
        st.download_button(
            label="Download All Enhanced Images as ZIP",
            data=f,
            file_name=zip_filename,
            mime="application/zip"
        )

    # Clean up the temporary directory
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    st.success("Enhancement Complete!")
else:
    st.info("Please upload one or more images to get started.")
