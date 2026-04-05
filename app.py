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
# CUSTOM CSS (WHITE + GREEN THEME)
# ======================================
st.markdown("""
<style>
body {
    background-color: #f5f7f6;
}

.block-container {
    padding-top: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Titles */
h1, h2, h3 {
    color: #14532d;
}

/* Card */
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    background-color: #16a34a;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
}
.stButton>button:hover {
    background-color: #15803d;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #16a34a;
    border-radius: 10px;
    background: #f0fdf4;
}

/* Metrics */
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

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
# MODEL
# ======================================
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        return None
    return YOLO("best.pt")

model = load_model()

# ======================================
# DETECTION FUNCTION
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
page = st.sidebar.radio("", ["Home", "Detection", "Results", "About"])

# ======================================
# HOME
# ======================================
if page == "Home":
    st.title("Tomato Disease Detection System")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    This system uses YOLOv8 deep learning model to detect diseases in tomato leaves.

    It helps farmers and agricultural professionals:
    - Detect diseases early  
    - Improve crop yield  
    - Make data-driven decisions  

    Navigate to the Detection page to start analysis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# DETECTION
# ======================================
elif page == "Detection":
    st.title("Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload Tomato Leaf Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Output")

            if st.button("Run Detection"):

                if model is None:
                    st.error("Model file 'best.pt' not found")
                else:
                    with st.spinner("Processing..."):
                        detections, annotated = run_detection(image)

                    st.image(annotated, use_container_width=True)

                    st.markdown("### Predictions")
                    for name, conf in detections:
                        st.write(f"{name} — {conf:.2f}")
                        save_result(name, conf)

        st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# RESULTS
# ======================================
elif page == "Results":
    st.title("Detection History")

    data = get_results()

    if not data:
        st.info("No records available")
    else:
        total = len(data)
        avg_conf = sum(x[1] for x in data) / total

        col1, col2 = st.columns(2)
        col1.metric("Total Detections", total)
        col2.metric("Average Confidence", f"{avg_conf:.2f}")

        st.markdown('<div class="card">', unsafe_allow_html=True)

        for d in data:
            st.write(f"{d[0]} | {d[1]:.2f} | {d[2]}")

        st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# ABOUT
# ======================================
elif page == "About":
    st.title("About Project")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    This project is developed for tomato leaf disease detection using YOLOv8.

    Technologies:
    - Streamlit
    - YOLOv8 (Ultralytics)
    - OpenCV
    - SQLite

    Purpose:
    To assist in early disease detection and improve agricultural productivity.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
