import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import sqlite3
import pandas as pd
import plotly.express as px
from PIL import Image
from datetime import datetime
import requests
from streamlit_lottie import st_lottie
import numpy as np
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart Waste Sorter", page_icon="♻️", layout="wide")

# --- DATABASE FUNCTIONS ---

DB_PATH = os.path.join(os.path.dirname(__file__), 'waste_data.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS waste_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT,
            waste_type TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(class_name, waste_type, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO waste_log (class_name, waste_type, confidence) VALUES (?, ?, ?)',
              (class_name, waste_type, confidence))
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM waste_log", conn)
    conn.close()
    return df

init_db()

# --- MAPPING LOGIC (Lowercase) ---
WASTE_MAPPING = {
    'biological': 'Biodegradable 🌿',
    'cardboard': 'Biodegradable 🌿',
    'paper': 'Biodegradable 🌿',
    'battery': 'Non-Biodegradable ⚠️',
    'clothes': 'Non-Biodegradable ⚠️',
    'glass': 'Non-Biodegradable ⚠️',
    'metal': 'Non-Biodegradable ⚠️',
    'plastic': 'Non-Biodegradable ⚠️',
    'shoes': 'Non-Biodegradable ⚠️',
    'trash': 'Non-Biodegradable ⚠️'
}

# --- HELPER: ANIMATION ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_bio = load_lottieurl("https://lottie.host/5a706692-2346-4444-93e5-827299042220/M4lq4iUkwP.json") 
lottie_nonbio = load_lottieurl("https://lottie.host/96230f81-5d97-4089-9e8c-572776c94412/1vVw9bB9tV.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('../outputs/models/best.pt')

try:
    model = load_model()
except:
    st.error("Model 'best.pt' not found.")
    st.stop()


def process_image(image_input, conf_threshold, col_display, col_result):
    # Prediksi
    results = model.predict(image_input, conf=conf_threshold)
    
    # Tampilkan Gambar Hasil
    res_plotted = results[0].plot()[:, :, ::-1]
    with col_display:
        st.image(res_plotted, use_container_width=True, caption="Detected Image")

    # Tampilkan Logic Sorting
    with col_result:
        st.subheader("Sorting Analysis")
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Logic Lowercase
            normalized_name = class_name.lower().strip()
            waste_type = WASTE_MAPPING.get(normalized_name, "Unknown")
            
            # Save DB
            save_to_db(class_name, waste_type, conf)
            
            # Display UI
            if "Biodegradable" in waste_type:
                st.success(f"🌱 **BIODEGRADABLE**")
                st.markdown(f"### Item: {class_name.upper()}")
                if lottie_bio: st_lottie(lottie_bio, height=180, key=f"anim_{datetime.now()}")
                st.info("Action: **GREEN BIN** (Compost)")
            else:
                st.error(f"⚙️ **NON-BIODEGRADABLE**")
                st.markdown(f"### Item: {class_name.upper()}")
                if lottie_nonbio: st_lottie(lottie_nonbio, height=180, key=f"anim_{datetime.now()}")
                st.info("Action: **YELLOW/RED BIN** (Recycle)")
            
            st.toast(f"Logged: {class_name}", icon="✅")
        else:
            st.warning("No trash detected in this frame.")

# --- MAIN PAGE UI ---
st.title("♻️ Smart Waste Sorting System")
st.markdown("### 3-Mode Intelligent Detection Dashboard")

# --- DASHBOARD ---
st.markdown("---")
df = load_data()
if not df.empty:
    c1, c2 = st.columns([1, 4])
    with c1:
        st.metric("Total Items", len(df))
    with c2:
        counts = df['class_name'].value_counts().reset_index()
        counts.columns = ['Item', 'Count']
        fig = px.bar(counts, x='Item', y='Count', title='Detected Items Frequency', 
                     text_auto=True, color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data yet. Start detecting!")
st.markdown("---")

# --- 3 TABS SYSTEM ---
tab1, tab2, tab3 = st.tabs(["📂 Upload File", "📸 Take Photo (Snapshot)", "📹 Real-Time Live"])

# === TAB 1: UPLOAD FILE ===
with tab1:
    col1_L, col1_R = st.columns([1, 1])
    with col1_L:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])
        conf_val1 = st.slider("Sensitivity", 0.0, 1.0, 0.4, key='c1')
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        # Panggil fungsi helper
        process_image(img, conf_val1, col1_L, col1_R)

# === TAB 2: TAKE PHOTO (SNAPSHOT) ===
with tab2:
    col2_L, col2_R = st.columns([1, 1])
    with col2_L:
        st.subheader("Camera Snapshot")
        # Widget Camera Input (Native Streamlit)
        camera_file = st.camera_input("Take a picture")
        conf_val2 = st.slider("Sensitivity", 0.0, 1.0, 0.4, key='c2')

    if camera_file:
        img = Image.open(camera_file)
        # Panggil fungsi helper (sama persis logicnya dengan Tab 1)
        process_image(img, conf_val2, col2_L, col2_R)

# === TAB 3: REAL-TIME LIVE ===
with tab3:
    st.subheader("Live Stream Processing")
    st.write("Detects objects in real-time. Auto-logs to database.")
    
    run_live = st.checkbox('Start Live Camera')
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    if run_live:
        while True:
            ret, frame = cap.read()
            if not ret: 
                st.error("Camera error.")
                break
            
            results = model(frame, conf=0.5)
            frame_plotted = results[0].plot()
            frame_rgb = cv2.cvtColor(frame_plotted, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            if len(results[0].boxes) > 0:
                name = model.names[int(results[0].boxes[0].cls[0])]
                w_type = WASTE_MAPPING.get(name.lower().strip(), "Unknown")
                
                if "Biodegradable" in w_type:
                    status_text.success(f"🌱 {name.upper()} (BIO)")
                else:
                    status_text.error(f"⚙️ {name.upper()} (NON-BIO)")
            else:
                status_text.info("Scanning...")
    else:
        cap.release()