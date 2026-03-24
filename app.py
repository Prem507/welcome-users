import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import cv2
import os

st.set_page_config(page_title="Helmet Detection", layout="wide")

st.title("🪖 Helmet Detection System")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # make sure best.pt is in repo

model = load_model()

# ---------------- SELECT INPUT ----------------
option = st.radio("Choose Input Type", ["Image", "Video"])

# ================= IMAGE =================
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        results = model(img_array)
        annotated = results[0].plot()

        st.image(annotated, caption="Result", use_container_width=True)

        # detection logic
        detected = False
        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                name = model.names[int(cls)]
                if "helmet" in name.lower():
                    detected = True

        if detected:
            st.success("Helmet Detected ✅")
        else:
            st.error("No Helmet ❌")

# ================= VIDEO =================
if option == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            stframe.image(annotated, channels="BGR", use_container_width=True)

        cap.release()
        os.remove(tfile.name)
