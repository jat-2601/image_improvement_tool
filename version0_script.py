import streamlit as st
import numpy as np
from PIL import Image
import torch
from transformers import SwinForImageClassification, AutoFeatureExtractor
import zipfile
import os
from huggingface_hub import login

# Authenticate with Hugging Face using the token from environment variable
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
else:
    st.error("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN environment variable.")

# Load the SwinIR model and feature extractor directly from Hugging Face
def load_model():
    try:
        model = SwinForImageClassification.from_pretrained("jayyap/swinir")
        feature_extractor = AutoFeatureExtractor.from_pretrained("jayyap/swinir")
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Enhance image using SwinIR
def enhance_image_with_swinir(image, model, feature_extractor):
    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        enhanced_image = model(**inputs).logits
    enhanced_image = np.clip(enhanced_image[0].numpy() * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)

# Streamlit app layout
st.title("Image Enhancement Dashboard with SwinIR")
st.write("Upload low-resolution images to enhance their quality!")

# Load the model
model, feature_extractor = load_model()

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Processing images
if model and feature_extractor and uploaded_files:
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        # Enhance the image using SwinIR
        enhanced_image = enhance_image_with_swinir(image, model, feature_extractor)

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
