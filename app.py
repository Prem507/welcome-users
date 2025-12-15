import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import cv2
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0b1c2d;
}
.stApp {
    background-color: #0b1c2d;
}
h1, h2, h3, p, label {
    color: white !important;
}
.stButton button {
    background-color: #38bdf8;
    color: black;
    border-radius: 8px;
}
.stFileUploader {
    background-color: #112d4e;
    padding: 10px;
    border-radius: 10px;
}

/* -------- TOAST POPUP FIX -------- */
div[data-testid="stToast"] {
    background-color: #38bdf8 !important;  /* popup background */
}
div[data-testid="stToast"] * {
    color: black !important;               /* popup text color */
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🪖 AI-Based Helmet Detection System")
st.write("Upload an **image or video** to detect helmet usage")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- INPUT TYPE ----------------
option = st.radio(
    "Select input type",
    ("Image Upload", "Video Upload")
)

# =====================================================
# ================= IMAGE UPLOAD ======================
# =====================================================
if option == "Image Upload":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)

        results = model(img_array)
        annotated = results[0].plot()

        st.image(
            annotated,
            caption="Detection Result",
            use_container_width=True
        )

        helmet_detected = False
        no_helmet_detected = False

        if results[0].boxes is not None:
            for cls_id in results[0].boxes.cls.tolist():
                class_name = model.names[int(cls_id)].lower()

                if "without" in class_name or "no helmet" in class_name:
                    no_helmet_detected = True
                if "helmet" in class_name and "without" not in class_name:
                    helmet_detected = True

        # -------- ALERT + POPUP --------
        if no_helmet_detected:
            st.error("🚨 ALERT: Helmet NOT detected")
            st.toast("🚨 Helmet NOT detected!", icon="🚨")

        elif helmet_detected:
            st.success("✅ Helmet detected")
            st.toast("✅ Helmet detected", icon="🪖")

        else:
            st.info("ℹ️ No rider detected")

# =====================================================
# ================= VIDEO UPLOAD ======================
# =====================================================
if option == "Video Upload":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            stframe.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()
        os.unlink(tfile.name)