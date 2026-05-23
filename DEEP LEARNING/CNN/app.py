import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ==========================================
# 1. PAGE LAYOUT CONFIGURATION
# ==========================================
st.set_page_config(page_title="Brain Tumor AI Detector", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor Detection AI Dashboard")
st.write("Upload a Medical Brain MRI Scan image to diagnose the presence of Tumor cells using our trained Core CNN Deep Learning Pipeline.")

# ==========================================
# 2. LOAD TRAINED CORE MODEL
# ==========================================
model_path = 'brain_tumor_cnn_model.h5'

@st.cache_resource
def load_our_model():
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_our_model()

if model is None:
    st.error(f"❌ Error: Trained model file '{model_path}' not found in root folder! Please run 'train.py' first.")
else:
    # ==========================================
    # 3. FILE UPLOADER & INTERACTIVE DISPLAY
    # ==========================================
    uploaded_file = st.file_uploader("Choose a Brain MRI Scan Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display user image using PIL layout structure natively
        user_img = Image.open(uploaded_file)
        
        st.write("### 📸 Uploaded MRI Scan Details:")
        st.image(user_img, caption="Target Scan Matrix Layer", use_column_width=True)
        
        # Diagnostic button trigger
        if st.button("🚀 Analyze Scan Model Matrix"):
            with st.spinner('AI analyzing the structural parameters...'):
                
                # Image Preprocessing using Pure Native RGB Array Mapping (No CV2)
                # Convert grayscale scanning variants safely to clear RGB channel matrix layout
                img_rgb = user_img.convert('RGB')
                img_resized = img_rgb.resize((128, 128))
                
                # Normalize values
                img_array = image.img_to_array(img_resized)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Run Model Prediction Inference
                prediction = model.predict(img_array)
                confidence_score = prediction[0][0]
                
                # ==========================================
                # 4. DISPLAY REAL-TIME PREDICTIONS
                # ==========================================
                st.write("---")
                st.subheader("📊 Diagnostic Report Result Summary:")
                
                if confidence_score > 0.5:
                    st.error(f"⚠️ **Result Status:** BRAIN TUMOR DETECTED! (Probability Score: {confidence_score:.4f})")
                    st.warning("Immediate clinical advice recommended for structural anomaly verification analysis layer mapping.")
                else:
                    st.success(f"✅ **Result Status:** HEALTHY BRAIN SCAN (NO TUMOR) (Probability Score: {confidence_score:.4f})")
                    st.info("The structural pattern exhibits standard layout metrics under target testing conditions data mapping parameters.")