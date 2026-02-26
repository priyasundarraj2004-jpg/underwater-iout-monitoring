import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import random
import pandas as pd
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="IoUT Underwater Monitoring System",
    page_icon="🌊",
    layout="wide"
)

# ------------------ STYLE ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #000c1a, #001f3f, #003366, #001a33);
}
.main-header {
    font-size: 40px;
    color: #00f2ff;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">🌊 IoUT Marine Monitoring Control Center</div>',
    unsafe_allow_html=True
)

# ------------------ LOAD MODEL ------------------
MODEL_PATH = "runs/detect/train2/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found! Check best.pt path.")
    st.stop()

model = YOLO(MODEL_PATH)

# ------------------ IMAGE ENHANCEMENT FUNCTION ------------------
def enhance_underwater_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32)

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    img[:, :, 0] *= avg_gray / (avg_b + 1e-6)
    img[:, :, 1] *= avg_gray / (avg_g + 1e-6)
    img[:, :, 2] *= avg_gray / (avg_r + 1e-6)

    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙ Control Panel")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.25
)

monitoring_mode = st.sidebar.selectbox(
    "Monitoring Mode",
    ["Image Comparison Mode", "Video Detection Mode"]
)

st.sidebar.success("System Status: ACTIVE")

# ------------------ SENSOR SIMULATION ------------------
def generate_sensor_data():
    return {
        "Temperature (°C)": round(random.uniform(18, 30), 2),
        "Salinity (PSU)": round(random.uniform(30, 40), 2),
        "Depth (m)": round(random.uniform(50, 200), 2),
        "Pressure (kPa)": round(random.uniform(100, 300), 2)
    }

sensor_data = generate_sensor_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡 Temperature (°C)", sensor_data["Temperature (°C)"])
col2.metric("🧂 Salinity (PSU)", sensor_data["Salinity (PSU)"])
col3.metric("📏 Depth (m)", sensor_data["Depth (m)"])
col4.metric("🌊 Pressure (kPa)", sensor_data["Pressure (kPa)"])

st.markdown("---")

# ================== IMAGE MODE ==================
if monitoring_mode == "Image Comparison Mode":

    uploaded_file = st.file_uploader(
        "📤 Upload Underwater Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image_array = np.array(image)

        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

        if st.button("🚀 Run Full Analysis"):

            results_original = model(image_array, conf=confidence)
            annotated_original = results_original[0].plot()
            original_count = len(results_original[0].boxes)

            enhanced_image = enhance_underwater_image(image_array)

            results_enhanced = model(enhanced_image, conf=confidence)
            annotated_enhanced = results_enhanced[0].plot()
            enhanced_count = len(results_enhanced[0].boxes)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Before Enhancement")
                st.image(annotated_original, use_container_width=True)
                st.success(f"Objects Detected: {original_count}")

            with col2:
                st.subheader("After Enhancement")
                st.image(annotated_enhanced, use_container_width=True)
                st.success(f"Objects Detected: {enhanced_count}")

            st.markdown("### 📊 Detection Improvement")

            comparison_df = pd.DataFrame({
                "Stage": ["Original", "Enhanced"],
                "Detected Objects": [original_count, enhanced_count]
            })

            st.bar_chart(comparison_df.set_index("Stage"))

# ================== VIDEO MODE ==================
if monitoring_mode == "Video Detection Mode":

    st.header("🎥 Underwater Video Object Detection")

    uploaded_video = st.file_uploader(
        "Upload Underwater Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()

        total_detections = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            total_detections += len(results[0].boxes)

            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_frame, use_container_width=True)

        cap.release()

        st.success(f"Total Objects Detected in Video: {total_detections}")

# ================== MODEL EVALUATION ==================
st.markdown("---")
st.header("📊 Model Evaluation on Validation Dataset")

DATA_YAML = "data.yaml"

if os.path.exists(DATA_YAML):

    if st.button("📈 Run Model Evaluation"):

        with st.spinner("Evaluating model..."):
            metrics = model.val(data=DATA_YAML)

        precision = float(metrics.box.p.mean())
        recall = float(metrics.box.r.mean())
        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{precision:.3f}")
        col2.metric("Recall", f"{recall:.3f}")
        col3.metric("F1 Score", f"{f1:.3f}")

        col4, col5 = st.columns(2)
        col4.metric("mAP@0.5", f"{map50:.3f}")
        col5.metric("mAP@0.5:0.95", f"{map5095:.3f}")

        performance_df = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1 Score", "mAP@0.5", "mAP@0.5:0.95"],
            "Value": [precision, recall, f1, map50, map5095]
        })

        st.bar_chart(performance_df.set_index("Metric"))

else:
    st.warning("⚠ data.yaml not found.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "<center style='color:#00f2ff;'>© 2026 IoUT Underwater AI Monitoring | MSc Research Project</center>",
    unsafe_allow_html=True
)