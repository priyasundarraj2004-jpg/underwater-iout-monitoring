import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Underwater Object Detection",
                   layout="centered")

st.title("🌊 Underwater Object Detection System")
st.write("Upload an underwater image to detect objects.")

# -------------------------------
# Load YOLO Model (Cloud Safe)
# -------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")   # Cloud compatible model
    return model

model = load_model()

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    # -------------------------------
    # Run Detection
    # -------------------------------
    results = model(img_array)

    result = results[0]
    boxes = result.boxes

    detected_count = len(boxes)

    # Draw bounding boxes
    annotated_frame = result.plot()

    st.image(annotated_frame, caption="Detected Objects", use_column_width=True)

    st.success(f"✅ Total Objects Detected: {detected_count}")

    # -------------------------------
    # Show Detection Details
    # -------------------------------
    if detected_count > 0:
        st.subheader("Detection Details:")

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            st.write(f"• Object: **{class_name}** | Confidence: **{confidence:.2f}**")

    else:
        st.warning("⚠ No objects detected.")
