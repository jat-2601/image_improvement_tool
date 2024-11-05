import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import zipfile
import os

# Function to load the ESRGAN model from a .h5 file
def load_esrgan_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Enhance image using ESRGAN
def enhance_image_with_esrgan(image, model):
    # Resize and normalize the image for ESRGAN input
    image = image.resize((image.width // 4, image.height // 4))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict and process the enhanced image
    enhanced_image = model.predict(image_array)
    enhanced_image = np.clip(enhanced_image[0] * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)

# Streamlit app layout
st.title("Image Enhancement Dashboard with ESRGAN")
st.write("Upload low-resolution images to enhance their quality!")

# Load the ESRGAN model
model_path = "model.h5"  # Ensure this path is correct
model = load_esrgan_model(model_path)

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Processing images
if model and uploaded_files:
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        # Enhance the image using ESRGAN
        enhanced_image = enhance_image_with_esrgan(image, model)

        # Save enhanced images to the temporary directory
        enhanced_image_path = os.path.join(temp_dir, f"enhanced_{os.path.splitext(uploaded_file.name)[0]}.png")
        enhanced_image.save(enhanced_image_path)

        # Display original and enhanced images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image.resize((500, 750)), caption="Original Image", use_column_width=True)
        with col2:
            st.image(enhanced_image.resize((500, 750)), caption="Enhanced Image", use_column_width=True)

    # Create a ZIP file containing all enhanced images
    zip_filename = "enhanced_images.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in os.listdir(temp_dir):
            zipf.write(os.path.join(temp_dir, file), file)

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
    st.info("Please ensure the model is loaded correctly and upload images to get started.")
