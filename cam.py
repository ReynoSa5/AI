import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🎥 deteksi warna dominan realtime")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

def get_dominant_color(frame):
    img = cv2.resize(frame, (100, 100))
    img = img.reshape((-1, 3))
    img = np.float32(img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 1
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return tuple(map(int, centers[0]))

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    color = get_dominant_color(frame)
    cv2.rectangle(frame, (10, 10), (110, 110), color, -1)
    cv2.putText(frame, f"BGR: {color}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
