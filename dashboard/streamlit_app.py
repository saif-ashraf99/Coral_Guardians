import streamlit as st
import cv2
import numpy as np
import tempfile
import torch

from detection.inference import load_model, run_inference

# Load model once outside the main function to avoid repeated overhead
@st.cache_resource
def get_model(weights_path):
    return load_model(weights=weights_path, device='cpu')

def draw_boxes(image, detections, class_names):
    for det in detections:
        (x1, y1, x2, y2) = det["bbox"]
        cls = det["class"]
        conf = det["confidence"]
        label = f"{class_names[cls]}: {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main():
    st.title("Coral Guardians: Red Sea Underwater Monitoring")

    # Provide the path to your YOLOv5 weights
    weights_path = './detection/yolov5/runs/train/coral_model/weights/best.pt'
    class_names = ["coral", "fish", "debris"]  # Must match your data.yaml

    model = get_model(weights_path)

    uploaded_file = st.file_uploader("Upload an underwater image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        image = cv2.imread(tmp_path)
        st.image(image, caption="Original Image", channels="BGR")

        with st.spinner("Running inference..."):
            detections = run_inference(model, tmp_path)

        st.success("Inference complete!")

        # Draw bounding boxes
        drawn_image = draw_boxes(image.copy(), detections, class_names)
        st.image(drawn_image, caption="Detection Results", channels="BGR")

if __name__ == "__main__":
    main()
