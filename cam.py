import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🎨 deteksi warna dominan")

mode = st.radio("pilih mode input:", ["kamera", "upload foto"])

def get_dominant_color(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = img.reshape((-1, 3))
    img = np.float32(img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 1
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = centers[0].astype(int)
    return tuple(dominant_color)

if mode == "kamera":
    cam_choice = st.radio("pilih kamera:", ["default", "depan", "belakang"])

    # NOTE: st.camera_input belum support facingMode
    # jadi "depan/belakang" ini lebih ke UI aja
    img_file = st.camera_input("ambil gambar dari kamera")

    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, caption=f"gambar dari kamera {cam_choice}")

        color = get_dominant_color(img)
        st.write(f"warna dominan (BGR): {color}")
        st.color_picker("preview", value="#%02x%02x%02x" % (color[2], color[1], color[0]))

elif mode == "upload foto":
    uploaded_file = st.file_uploader("upload foto", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="foto yang diupload")

        color = get_dominant_color(img)
        st.write(f"warna dominan (BGR): {color}")
        st.color_picker("preview", value="#%02x%02x%02x" % (color[2], color[1], color[0]))
