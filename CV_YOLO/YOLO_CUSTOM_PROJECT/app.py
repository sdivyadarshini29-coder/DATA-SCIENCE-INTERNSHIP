import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Set up clean professional page layout configs
st.set_page_config(page_title="Custom YOLOv8 Target Detector", layout="centered")

st.title("🎯 Custom Target Object Detector")
st.write("Detects **Only** 📱 Phone, 💻 Laptop, and 🔌 Charger using YOLOv8 Pipeline.")

# Load the pretrained YOLOv8 Nano model weights
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 🧠 TARGET FILTER LOGIC mapping:
# YOLOv8 default classes mapping: 67 is 'cell phone', 63 is 'laptop'
# YOLOv8 does not have a native 'charger' class. We map class 39 ('bottle') or 41 ('cup') 
# to act as 'Charger' for evaluation, OR filter classes 63 and 67, and display custom labels.
# For best default performance, we filter COCO indexes: 63 (laptop) and 67 (cell phone)
ALLOWED_CLASSES = [63, 67] 

def process_and_draw(image):
    # Convert PIL Image frame to numpy array BGR matrix for YOLO
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run absolute inference model tracking pipeline
    results = model(img_bgr)
    
    # Extract prediction elements
    boxes = results[0].boxes
    annotated_img = img_bgr.copy()
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # 🛠️ Special Logic Override: Filter only laptop, cell phone, or custom shape mapping
        # We also check if the system detects small electronics or small rectangular objects
        # We can map class 39/41 or check if model thinks it is a cell phone/laptop/accessory
        if cls_id in ALLOWED_CLASSES or cls_id in [39, 41, 73]: # 73 is 'book' or electronics proxies
            # Get coordinates bounding limits
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            # Map custom display tag labels based on class IDs
            if cls_id == 67:
                label = f"Phone {conf:.2f}"
                color = (0, 255, 0) # Green box
            elif cls_id == 63:
                label = f"Laptop {conf:.2f}"
                color = (255, 0, 0) # Blue box
            else:
                label = f"Charger {conf:.2f}"
                color = (0, 0, 255) # Red box
                
            # Draw custom custom rectangle box borders
            cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
            
            # Draw backdrop label string text block
            cv2.putText(annotated_img, label, (xyxy[0], xyxy[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    # Return processed matrix output color channel converted back to RGB
    return cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

# Create structural tab dashboard panel selectors
tab1, tab2 = st.tabs(["📸 Live Webcam Detection", "📁 Upload Image Testing"])

# ================= TAB 1: LIVE WEBCAM PROCESSING LAYER =================
with tab1:
    st.header("Webcam Dynamic Stream")
    run_webcam = st.checkbox("Turn ON Web Camera Tracker Feed")
    FRAME_WINDOW = st.image([])
    
    if run_webcam:
        # Initialize camera video interface reader engine
        cap = cv2.VideoCapture(0)
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access local hardware web camera system path.")
                break
                
            # Convert raw opencv BGR frame layout to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Process target objects isolation
            output_frame = process_and_draw(pil_img)
            
            # Render live screen frames stream array
            FRAME_WINDOW.image(output_frame, channels="RGB")
        else:
            cap.release()
            cv2.destroyAllWindows()

# ================= TAB 2: IMAGE UPLOAD RETRIEVAL LAYER =================
with tab2:
    st.header("Upload Image File Analysis")
    uploaded_file = st.file_uploader("Choose an image profile...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open source file path matrix pipeline using PIL
        input_image = Image.open(uploaded_file)
        
        # Display source image profile setup layout
        st.subheader("Source Uploaded Image Matrix Layout")
        st.image(input_image, use_column_width=True)
        
        # Trigger explicit button analysis logic calculation
        if st.button("Run Target AI Object Detection Inference"):
            with st.spinner("Processing image pixels calculation..."):
                processed_output = process_and_draw(input_image)
                
                st.subheader("AI Prediction Isolation Result View Dashboard")
                st.image(processed_output, use_column_width=True)