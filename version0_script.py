import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ExifTags
from io import BytesIO


# Load a pre-trained super-resolution model (example using OpenCV)
@st.cache_resource
def load_super_resolution_model():
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = "espcn_x4.pb"  # Path to your model file
    sr.readModel(model_path)
    sr.setModel("espcn", 4)  # Set the model and scale
    return sr


# Enhance a single image with basic adjustments
def enhance_image(image, brightness, contrast, sharpness, hist_eq=False):
    if hist_eq:
        lab = image.convert("LAB")
        l, a, b = lab.split()
        l = ImageEnhance.Contrast(l).enhance(2)
        lab = Image.merge("LAB", (l, a, b))
        image = lab.convert("RGB")

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    return image


# Apply filters
def apply_filter(image, filter_type):
    if filter_type == "Grayscale":
        return image.convert("L")
    elif filter_type == "Sepia":
        sepia_image = np.array(image)
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        sepia_image = cv2.transform(sepia_image, sepia_filter)
        return Image.fromarray(np.clip(sepia_image, 0, 255).astype(np.uint8))
    return image


# Super-resolution enhancement
def super_resolve(image, sr_model):
    image_np = np.array(image)
    result = sr_model.upsample(image_np)
    return Image.fromarray(result)


# Streamlit app layout
st.title("Enhanced Image Pixel Enhancer Dashboard")
st.write("Upload low-resolution images, apply enhancements, and adjust image properties!")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True)

# Enhancement sliders
brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0)
hist_eq = st.checkbox("Apply Histogram Equalization")
filter_type = st.selectbox("Choose a Filter", ["None", "Grayscale", "Sepia"])

# Load the super-resolution model
sr_model = load_super_resolution_model()

# Processing images
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        # Display metadata if available
        st.write(f"### {uploaded_file.name}")
        metadata = {ExifTags.TAGS[k]: v for k, v in image._getexif().items() if
                    k in ExifTags.TAGS} if image._getexif() else None
        if metadata:
            st.write("Image Metadata:", metadata)

        # Enhance the image with basic adjustments and filters
        enhanced_image_basic = enhance_image(image, brightness, contrast, sharpness, hist_eq)
        enhanced_image_basic = apply_filter(enhanced_image_basic, filter_type)

        # Apply super-resolution
        enhanced_image_sr = super_resolve(image, sr_model)

        # Show original and enhanced images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(enhanced_image_basic, caption="Enhanced Image (Basic Adjustments)", use_column_width=True)

        # Show super-resolution enhanced image
        st.image(enhanced_image_sr, caption="Enhanced Image (Super-Resolution)", use_column_width=True)

        # Download enhanced images
        buf_basic = BytesIO()
        enhanced_image_basic.save(buf_basic, format="PNG")
        byte_im_basic = buf_basic.getvalue()

        buf_sr = BytesIO()
        enhanced_image_sr.save(buf_sr, format="PNG")
        byte_im_sr = buf_sr.getvalue()

        st.download_button(
            label=f"Download Enhanced Image (Basic Adjustments) - {uploaded_file.name}",
            data=byte_im_basic,
            file_name=f"enhanced_basic_{uploaded_file.name.split('.')[0]}.png",
            mime="image/png"
        )

        st.download_button(
            label=f"Download Enhanced Image (Super-Resolution) - {uploaded_file.name}",
            data=byte_im_sr,
            file_name=f"enhanced_sr_{uploaded_file.name.split('.')[0]}.png",
            mime="image/png"
        )

    st.success("Enhancement Complete!")
else:
    st.info("Please upload one or more images to get started.")
