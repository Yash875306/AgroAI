# ======================================
# IMPORTS
# ======================================
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import sqlite3
from datetime import datetime
import os

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="AgroAI - Tomato Disease Detection",
    layout="wide"
)

# ======================================
# DATABASE
# ======================================
DB = "agroai.db"

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_result(disease, confidence):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO detections (disease, confidence, timestamp) VALUES (?, ?, ?)",
        (disease, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def get_results():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT disease, confidence, timestamp FROM detections ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows

init_db()

# ======================================
# MODEL (CACHED)
# ======================================
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        return None
    return YOLO("best.pt")

model = load_model()

# ======================================
# INFERENCE
# ======================================
def run_detection(image):
    arr = np.array(image)
    results = model.predict(arr, conf=0.25, verbose=False)
    r = results[0]

    detections = []
    annotated = image

    if r.boxes and len(r.boxes):
        annotated = Image.fromarray(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
        for box in r.boxes:
            detections.append((
                model.names[int(box.cls.item())],
                float(box.conf.item())
            ))

    return detections or [("Healthy", 1.0)], annotated

# ======================================
# SIDEBAR NAVIGATION
# ======================================
st.sidebar.title("AgroAI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Detection", "Results", "About"]
)

# ======================================
# HOME
# ======================================
if page == "Home":
    st.title("Tomato Disease Detection System")

    st.write("""
    This application uses a YOLOv8 deep learning model to detect and classify
    diseases in tomato leaves.

    The system helps in:
    - Early disease identification  
    - Improving crop productivity  
    - Supporting smart agriculture  

    Upload an image in the Detection section to begin analysis.
    """)

# ======================================
# DETECTION
# ======================================
elif page == "Detection":
    st.title("Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload a tomato leaf image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        with col2:
            if st.button("Run Detection"):

                if model is None:
                    st.error("Model file 'best.pt' not found.")
                else:
                    with st.spinner("Running detection..."):
                        detections, annotated = run_detection(image)

                    st.subheader("Detection Output")
                    st.image(annotated, use_container_width=True)

                    st.subheader("Predictions")
                    for name, conf in detections:
                        st.write(f"{name} — Confidence: {conf:.2f}")
                        save_result(name, conf)

# ======================================
# RESULTS
# ======================================
elif page == "Results":
    st.title("Detection History")

    results = get_results()

    if not results:
        st.info("No detection records available.")
    else:
        for disease, conf, time in results:
            st.write(f"{disease} | {conf:.2f} | {time}")

# ======================================
# ABOUT
# ======================================
elif page == "About":
    st.title("About Project")

    st.write("""
    This project is developed as part of a final year academic project.

    Technologies used:
    - Streamlit for frontend
    - YOLOv8 (Ultralytics) for detection
    - OpenCV and PIL for image processing
    - SQLite for storing detection results

    Objective:
    To detect tomato leaf diseases using deep learning and assist
    in precision agriculture.
    """)
