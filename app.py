import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import cv2
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
# Deep space/dark blue theme with improved component styling
st.markdown("""
<style>
/* --- General Styling --- */
body {
    color: white;
    background-color: #0d1117; /* GitHub Dark Mode BG */
}
.stApp {
    background-color: #0d1117;
}

/* --- Headings and Text --- */
h1, h2, h3, p, label {
    color: white !important;
}

/* --- Streamlit Components --- */
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #161b22; /* Darker sidebar */
    padding: 1rem;
}

/* Button styling - professional blue */
.stButton button {
    background-color: #238636; /* GitHub green for 'go' action */
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    transition: background-color 0.3s;
}
.stButton button:hover {
    background-color: #2ea043;
}

/* File Uploader styling - subtle background */
[data-testid="stFileUploader"] {
    background-color: #161b22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #30363d;
}

/* Radio button (Input Type) styling */
[data-testid="stRadio"] label {
    background-color: #161b22;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 5px;
    border: 1px solid #30363d;
}
[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 10px;
}

/* Alert/Toast/Feedback styling for high contrast */
.stAlert {
    border-radius: 10px;
    padding: 15px;
    font-size: 1.1em;
}

/* Metric styling (if used) */
[data-testid="stMetric"] {
    background-color: #161b22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #30363d;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🪖 AI-Based Helmet Detection System")
st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    # Placeholder for model loading message
    with st.spinner('Loading YOLO model...'):
        return YOLO("best.pt")

model = load_model()

# ---------------- SIDEBAR FOR INPUT ----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # ---------------- INPUT TYPE ----------------
    option = st.radio(
        "Select input type",
        ("Image Upload", "Video Upload")
    )

    uploaded_file = None

    # =====================================================
    # ================= FILE UPLOAD AREA ==================
    # =====================================================

    if option == "Image Upload":
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

    if option == "Video Upload":
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"]
        )

# ---------------- MAIN CONTENT AREA ----------------

# Create a container for dynamic feedback
feedback_container = st.container()

# Create a container for the visual output
output_container = st.container()

# --- Custom Bounding Box Plotting Logic ---
# Function to get custom plotting arguments
def get_plot_args(results):
    """
    Sets custom plotting arguments for better visualization based on detection type.
    """
    # Default visual settings for clarity
    args = {
        'line_width': 3,         # Thicker boxes
        'font_size': 14,         # Larger text
        'font_thickness': 2,     # Thicker font
        'conf': False,           # Hide confidence score by default for cleaner labels
        'labels': True,          # Show class labels
    }

    # Custom colors based on class (if possible, but usually handled by YOLO's internal scheme)
    # We can rely on YOLO's internal color palette which is usually good.
    # We could manually override colors, but it requires more complex logic.
    # For now, we enhance the general drawing parameters (line_width, font_size, etc.)

    return args

if uploaded_file:
    # --- Image Processing ---
    if option == "Image Upload":
        with st.spinner("Processing image and detecting..."):

            # Read image
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)

            # YOLO prediction
            results = model(img_array)

            # --- IMPROVEMENT: Customize Bounding Box Plotting ---
            plot_args = get_plot_args(results)
            annotated = results[0].plot(**plot_args)
            # ---------------------------------------------------

            # Display output
            with output_container:
                st.subheader("🖼️ Detection Result")
                st.image(
                    annotated,
                    caption=f"Processed Image: {uploaded_file.name}",
                    use_container_width=True
                )

            # --- Feedback Logic (Unchanged) ---
            helmet_detected = False
            no_helmet_detected = False

            if results[0].boxes is not None:
                for cls_id in results[0].boxes.cls.tolist():
                    class_name = model.names[int(cls_id)].lower()

                    if "without" in class_name or "no helmet" in class_name or "noh" in class_name:
                        no_helmet_detected = True
                    elif "helmet" in class_name or "with_h" in class_name:
                        helmet_detected = True

            # --- Alert + Popup (Visual Update) ---
            with feedback_container:
                col1, col2, col3 = st.columns([1, 1, 1])

                if no_helmet_detected:
                    st.error("🚨 ALERT: Helmet **NOT** detected for one or more riders!")
                    st.toast("🚨 Helmet NOT detected!", icon="🚨")
                    with col1:
                        st.metric(label="Compliance Status", value="NON-COMPLIANT", delta="IMMEDIATE ACTION REQUIRED")

                elif helmet_detected and not no_helmet_detected:
                    st.success("✅ **SUCCESS**: All detected riders are wearing a helmet.")
                    st.toast("✅ Helmet detected", icon="🪖")
                    with col1:
                        st.metric(label="Compliance Status", value="COMPLIANT", delta="GOOD")

                elif helmet_detected and no_helmet_detected:
                    st.warning("⚠️ WARNING: Mixed results. Some riders are compliant, others are NOT.")
                    st.toast("⚠️ Mixed results detected!", icon="⚠️")
                    with col1:
                        st.metric(label="Compliance Status", value="PARTIALLY COMPLIANT", delta="NEEDS REVIEW")

                else:
                    st.info("ℹ️ No riders (or relevant objects) were detected in the image.")
                    with col1:
                        st.metric(label="Compliance Status", value="N/A", delta="NO RIDERS DETECTED")


    # --- Video Processing ---
    if option == "Video Upload":
        with feedback_container:
            st.warning("Video processing can be resource-intensive. Wait for the video stream to start.")

        # Your existing video logic (Unchanged)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        with output_container:
            st.subheader("🎥 Live Video Detection")
            stframe = st.empty() # Placeholder for the video feed

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO prediction
                results = model(frame)

                # --- IMPROVEMENT: Customize Bounding Box Plotting ---
                plot_args = get_plot_args(results)
                annotated = results[0].plot(**plot_args)
                # ---------------------------------------------------

                # Display frame
                stframe.image(
                    annotated,
                    channels="BGR",
                    caption=f"Processed Video Stream: {uploaded_file.name}",
                    use_container_width=True
                )

            # Cleanup
            cap.release()
            os.unlink(tfile.name)
            stframe.empty() # Clear the video frame after processing ends
            feedback_container.success("Video processing complete.")

else:
    # Initial instruction/Welcome message
    with output_container:
        st.info(f"""
        ### Welcome! 👋
        Please select your input type (**{option.split()[0]}**) in the sidebar and upload a file.

        The AI will instantly analyze the uploaded media to check for mandatory helmet usage.
        """)
