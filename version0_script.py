import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import zipfile
import os

# Define a simple ESRGAN model (for demonstration purposes)
class ESRGAN(nn.Module):
    def __init__(self):
        super(ESRGAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

# Load the pretrained ESRGAN model (dummy model for demonstration)
@st.cache_resource
def load_esrgan_model():
    model = ESRGAN()
    # Normally you would load the pretrained weights here
    # model.load_state_dict(torch.load("path_to_pretrained_model.pth"))
    return model

# Enhance image using ESRGAN
def enhance_image_with_esrgan(image, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        enhanced_image_tensor = model(image_tensor)
    enhanced_image = transforms.ToPILImage()(enhanced_image_tensor.squeeze())
    return enhanced_image

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
