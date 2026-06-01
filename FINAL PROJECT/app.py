import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Page setup for Advanced Retail
st.set_page_config(page_title="Kirana AI Pro", layout="wide", page_icon="🏪")

# Cyber-gradient professional retail UI theme styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0d111a;
        background-image: 
            radial-gradient(at 20% 20%, rgba(14, 42, 71, 0.6) 0px, transparent 40%),
            radial-gradient(at 80% 80%, rgba(31, 16, 74, 0.5) 0px, transparent 50%);
        background-attachment: fixed;
        color: #e2e8f0 !important;
    }
    h1 {
        background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    h2, h3 { color: #38bdf8 !important; }
    .stAlert { border-radius: 12px !important; }
    div[data-testid="stMetricValue"] { color: #10b981 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Cache YOLO model to avoid freezing
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # COCO weights containing person, apple, orange, bottle, banana etc.

model = load_yolo_model()

# Base layout tabs mapping exactly to your 3 core problems
st.markdown("<h1>🏪 Kirana Automation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>AI Smart Inventory Tracking, Queue Alert Systems, & Fallback Point-of-Sale Architecture.</p>", unsafe_allow_html=True)
st.write("---")

tab1, tab2, tab3 = st.tabs([
    "🔍 AI Stock Search & Restock Engine", 
    "🚨 Queue Counter & Autonomous Checkout", 
    "📋 Manual Emergency Billing System"
])

# Helper to generate dummy stream when webcam is closed
def offline_frame(msg):
    img = np.zeros((480, 640, 3), np.uint8) + 20
    cv2.putText(img, msg, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 116, 139), 2)
    return img

# ==========================================
# PROBLEM 1: AI STOCK SEARCH & RESTOCK ENGINE
# ==========================================
with tab1:
    st.subheader("Visual Inventory Query System")
    st.write("Type any target product below. The AI camera matrix will instantly scan the shelf to verify existing quantities.")
    
    # Input product text box from user
    search_query = st.text_input("Enter Product Name to Audit (e.g., apple, banana, orange, bottle, broccoli):", value="apple").lower().strip()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"🎥 Active Scanner Node targeting: **{search_query.upper()}**")
        run_stock_cam = st.checkbox("Boot Shelf Scanning Camera")
        stock_slot = st.image([])
        
        # Simulated base target dictionary for dynamic comparison
        target_inventory_limits = {"apple": 5, "orange": 4, "banana": 3, "bottle": 6, "broccoli": 3}
        min_limit = target_inventory_limits.get(search_query, 3) # Fallback limit threshold
        
        live_count = 0
        if run_stock_cam:
            cap = cv2.VideoCapture(0)
            while run_stock_cam:
                ret, frame = cap.read()
                if not ret: break
                
                results = model(frame)
                annotated = results[0].plot()
                
                # Extract detected labels from bounding boxes
                boxes = results[0].boxes
                labels = [model.names[int(c)] for c in boxes.cls]
                
                # Count current occurrences matching user search input
                live_count = labels.count(search_query)
                
                # Dynamic box indicator overlay on visual frame
                cv2.rectangle(annotated, (10, 10), (320, 60), (15, 23, 42), -1)
                cv2.putText(annotated, f"Live {search_query.upper()} Count: {live_count}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
                
                stock_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            stock_slot.image(offline_frame("SHELF CAMERA STANDBY"))
            
    with col2:
        st.subheader("Telemetry Analysis")
        st.metric(label=f"Current Visual Count of '{search_query.upper()}'", value=live_count)
        st.write(f"Minimum Stock Threshold Limit: **{min_limit} units**")
        
        if run_stock_cam:
            if live_count < min_limit:
                st.error(f"🚨 ALERT: Stock level ({live_count}) dropped below threshold ({min_limit})! Please replenish the {search_query} shelf immediately.")
            else:
                st.success(f"✅ STABLE: {search_query.upper()} levels are sufficient. No current restock needed.")


# ==========================================
# PROBLEM 2: QUEUE ALARM & SELF-CHECKOUT
# ==========================================
with tab2:
    st.subheader("Checkout Density Control & Customer Self-Pay Terminal")
    
    # Allow storekeeper to configure the max queue limit dynamically for evaluation testing
    queue_threshold = st.slider("Configure Max Allowed Queue Limit (Set to low for easy testing):", min_value=1, max_value=15, value=2)
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.write("🎥 **Counter Queue Surveillance Cam**")
        run_queue_cam = st.checkbox("Initialize Counter Monitoring Node")
        queue_slot = st.image([])
        
        current_queue_size = 0
        if run_queue_cam:
            cap = cv2.VideoCapture(0)
            while run_queue_cam:
                ret, frame = cap.read()
                if not ret: break
                
                # Class 0 maps strictly to "person" in COCO weights
                results = model(frame, classes=[0])
                annotated = results[0].plot()
                current_queue_size = len(results[0].boxes)
                
                # Check for threshold breaches
                if current_queue_size > queue_threshold:
                    # Flash red danger bounding line over frame
                    cv2.rectangle(annotated, (0,0), (frame.shape[1], frame.shape[2]), (0, 0, 255), 15)
                    cv2.putText(annotated, "QUEUE OVERFLOW TRIPPED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # HTML5 code to inject real sound alert through audio browser elements
                    st.markdown("""
                        <audio autoplay hidden>
                            <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
                        </audio>
                    """, unsafe_allow_html=True)
                
                queue_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            queue_slot.image(offline_frame("COUNTER CAMERA STANDBY"))
            
    with c2:
        st.subheader("Counter Diagnostics")
        st.metric("Live Queue Headcount", f"{current_queue_size} Persons")
        
        if current_queue_size > queue_threshold:
            st.markdown(f"<div style='background-color:rgba(239,68,68,0.2); padding:20px; border-radius:10px; border:1px solid #ef4444;'><h4 style='color:#ef4444; margin:0;'>⚠️ OVERCROWDED ALERT triggered!</h4><p style='margin:0; color:#fca5a5;'>Queue has reached {current_queue_size} people. Directing incoming users to Self-Checkout terminals.</p></div>", unsafe_allow_html=True)
        else:
            st.success("🟢 Queue clearance status optimal. Flow rate stable.")
            
        st.write("---")
        st.subheader("🤖 Self-Checkout Machine Terminal")
        st.write("No queue delay! Place items under scanner for rapid digital receipt generation.")
        if st.button("Simulate Auto-Scan Basket Pay"):
            st.info("Scanning completed. Total: ₹120. Payment link dispatched via UPI.")


# ==========================================
# PROBLEM 3: EMERGENCY MANUAL POS BILLING TABLE
# ==========================================
with tab3:
    st.subheader("Barcode Failure Bypass Matrix (Fallback POS Engine)")
    st.warning("Use this manual input ledger table if a product item's printed label/barcode is damaged or unscannable.")
    
    # Store standard item prices dictionary database lookup
    menu_prices = {
        "Tomato (1kg)": 40.0,
        "Potato (1kg)": 30.0,
        "Apple (1kg)": 150.0,
        "Fresh Milk (1L)": 28.0,
        "Cooking Oil (1L)": 120.0,
        "Rice Bag (5kg)": 290.0,
        "Biscuit Pack": 10.0
    }
    
    # Creating structured entry lines
    st.write("### Add Damaged Label Items Below:")
    
    billing_rows = []
    
    # Loop to generate input fields row by row for clean matrix formatting
    for i in range(4): # Generates 4 custom item entry vectors
        col_item, col_qty, col_rate = st.columns([2, 1, 1])
        
        with col_item:
            selected_product = st.selectbox(f"Select Item {i+1}:", ["-- None Selected --"] + list(menu_prices.keys()), key=f"prod_{i}")
        
        with col_qty:
            qty_input = st.number_input(f"Quantity:", min_value=0, max_value=50, value=0, step=1, key=f"qty_{i}")
            
        with col_rate:
            if selected_product != "-- None Selected --":
                base_rate = menu_prices[selected_product]
                st.text_input(f"Rate (₹):", value=f"{base_rate}", disabled=True, key=f"rate_{i}")
                # Compute total cost row calculation dynamically
                row_total = qty_input * base_rate
                billing_rows.append({"Product": selected_product, "Quantity": qty_input, "Rate (₹)": base_rate, "Total (₹)": row_total})
            else:
                st.text_input(f"Rate (₹):", value="0.0", disabled=True, key=f"rate_{i}")

    # Process invoice logs table visually if item rows exist
    if len(billing_rows) > 0:
        st.write("---")
        st.subheader("🧾 Generated Manual Fallback Invoice")
        df_invoice = pd.DataFrame(billing_rows)
        
        # Filter rows to only display active counts higher than zero
        df_active = df_invoice[df_invoice["Quantity"] > 0]
        
        if not df_active.empty:
            st.dataframe(df_active, use_container_width=True)
            
            grand_total_bill = df_active["Total (₹)"].sum()
            st.metric("Aggregate Calculated Ledger Bill", f"₹ {grand_total_bill:,.2f}")
            
            if st.button("Print Manual Bill & Collect Cash"):
                st.success(f"Cash ledger entries processed successfully for Amount: ₹ {grand_total_bill}")
        else:
            st.info("Modify quantity counters above to compute billing ledger dynamically.")