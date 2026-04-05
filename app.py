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
import pandas as pd

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="AgroAI - Tomato Disease Detection",
    layout="wide"
)

# ======================================
# PREMIUM CSS
# ======================================
st.markdown("""
<style>
body { background-color:#f6f8f7; }

h1, h2, h3 { color:#14532d; }

/* Cards */
.card {
    background:white;
    padding:22px;
    border-radius:14px;
    box-shadow:0 4px 14px rgba(0,0,0,0.06);
    margin-bottom:20px;
}

/* Buttons */
.stButton>button {
    background:#16a34a;
    color:white;
    border-radius:8px;
    padding:10px 24px;
}
.stButton>button:hover { background:#15803d; }

/* Prediction Card */
.pred-card {
    padding:12px;
    border-radius:10px;
    border-left:5px solid #16a34a;
    background:#f0fdf4;
    margin-bottom:10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:white;
}

/* Table */
.dataframe {
    border-radius:10px;
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
# HOME (DASHBOARD STYLE)
# ======================================
if page == "Home":
    st.title("Tomato Disease Detection Dashboard")

    data = get_results()

    total = len(data)
    avg_conf = (sum(x[1] for x in data) / total) if total else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scans", total)
    col2.metric("Average Confidence", f"{avg_conf:.2f}")
    col3.metric("Model", "YOLOv8")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    This system detects tomato leaf diseases using YOLOv8 deep learning model.

    Upload an image in Detection section to analyze diseases and get insights.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# DETECTION
# ======================================
elif page == "Detection":
    st.title("Disease Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Input Image")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detection Output")

            if st.button("Run Detection"):
                if model is None:
                    st.error("Model not found")
                else:
                    with st.spinner("Analyzing..."):
                        detections, annotated = run_detection(image)

                    st.image(annotated, use_container_width=True)

                    st.subheader("Predictions")

                    for name, conf in detections:
                        st.markdown(f"""
                        <div class="pred-card">
                            <strong>{name}</strong><br>
                            Confidence: {conf:.2f}
                        </div>
                        """, unsafe_allow_html=True)

                        save_result(name, conf)

            st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# RESULTS (WITH ANALYTICS)
# ======================================
elif page == "Results":
    st.title("Detection Analytics")

    data = get_results()

    if not data:
        st.info("No data available")
    else:
        df = pd.DataFrame(data, columns=["Disease","Confidence","Time"])

        col1, col2 = st.columns(2)
        col1.metric("Total Records", len(df))
        col2.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}")

        st.markdown("### Disease Distribution")
        st.bar_chart(df["Disease"].value_counts())

        st.markdown("### Detection Records")
        st.dataframe(df, use_container_width=True)

# ======================================
# ABOUT
# ======================================
elif page == "About":
    st.title("About Project")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    AgroAI is a deep learning-based tomato disease detection system.

    Technologies:
    - YOLOv8 (Ultralytics)
    - Streamlit
    - OpenCV, NumPy, PIL
    - SQLite Database

    Objective:
    To assist farmers and researchers in early detection of plant diseases
    using AI-powered image analysis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
