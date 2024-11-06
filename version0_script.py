import streamlit as st
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionUpscalePipeline
import zipfile
import os
from huggingface_hub import login

# Streamlit app layout
st.title("Image Enhancement Dashboard with Stable Diffusion")
st.write("Upload low-resolution images to enhance their quality!")

# Simple authentication
username = st.text_input("Username")
password = st.text_input("Password", type="password")

# Hardcoded credentials (for demonstration purposes)
if username == "admin" and password == "password":
    st.success("Logged in successfully!")
    
    # Input for Hugging Face token
    hf_token = st.text_input("Enter your Hugging Face token:", type="password")

    # Authenticate with Hugging Face using the entered token
    if hf_token:
        try:
            login(token=hf_token)
            st.success("Successfully logged in to Hugging Face!")
        except Exception as e:
            st.error(f"Error logging in: {e}")

    # Load the Stable Diffusion Upscale model
    def load_model():
        try:
            model = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    # Enhance image using Stable Diffusion
    def enhance_image_with_stable_diffusion(image, model):
        enhanced_image = model(image).images[0]
        return enhanced_image

    # Load the model
    model = load_model()

    # File uploader for multiple images
    uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Processing images
    if model and uploaded_files:
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)

            # Enhance the image using Stable Diffusion
            enhanced_image = enhance_image_with_stable_diffusion(image, model)

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
    if username and password:
        st.error("Invalid username or password.")
