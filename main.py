# Library
import cv2
from PIL import Image
import streamlit as st
from helper import detect_food

# Title
st.title("Yemek Tanımlama Sistemi")

# Header
st.header("Lütfen fotoğraf yükleyin: ")

# Files
file = st.file_uploader("", type=["png", "jpg", "jpeg"])

# Model
model_path = "models/bestt.pt"

# Images
if file is not None:
    # Original Image
    st.header("Orijinal Fotoğraf")
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Processed Image
    st.header("Sonuç:")
    detection_result, is_detected = detect_food(image, model_path)

    if is_detected != 0:
        st.image(detection_result, use_column_width=True)
        st.write("##### Yemek Tanımlandı!" )

    else:
        st.image(detection_result, use_column_width=True)
        st.write("#### Yemek Tanımlanmadı !" )
