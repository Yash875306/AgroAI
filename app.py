import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import sqlite3
from datetime import datetime
import pandas as pd
import os
from report import generate_report

# ======================================
# CONFIG
# ======================================
st.set_page_config(page_title="AgroAI", layout="wide")

# ======================================
# PREMIUM UI
# ======================================
st.markdown("""
<style>
body { background:#f6f8f7; }
.card {
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.05);
    margin-bottom:20px;
}
.stButton>button {
    background:#16a34a;
    color:white;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# DATABASE
# ======================================
conn = sqlite3.connect("data.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS detections (
disease TEXT, confidence REAL, time TEXT)
""")

def save(d, cval):
    c.execute("INSERT INTO detections VALUES (?,?,?)",
              (d, cval, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

def fetch():
    return c.execute("SELECT * FROM detections").fetchall()

# ======================================
# MODEL
# ======================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def detect(img):
    arr = np.array(img)
    res = model.predict(arr, conf=0.25)
    r = res[0]

    out = img
    det = []

    if r.boxes:
        out = Image.fromarray(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
        for b in r.boxes:
            det.append((model.names[int(b.cls)], float(b.conf)))

    return det or [("Healthy",1.0)], out

# ======================================
# NAVIGATION
# ======================================
page = st.sidebar.radio("Menu", ["Home","Detection","Live Camera","Results","About"])

# ======================================
# HOME
# ======================================
if page == "Home":
    st.title("AgroAI Dashboard")

    data = fetch()
    total = len(data)

    col1,col2 = st.columns(2)
    col1.metric("Total Scans", total)
    col2.metric("Model","YOLOv8")

    st.markdown('<div class="card">AI-based tomato disease detection system.</div>', unsafe_allow_html=True)

# ======================================
# DETECTION
# ======================================
elif page == "Detection":
    st.title("Image Detection")

    file = st.file_uploader("Upload Image")

    if file:
        img = Image.open(file)

        c1,c2 = st.columns(2)

        with c1:
            st.image(img)

        with c2:
            if st.button("Run Detection"):
                det, out = detect(img)
                st.image(out)

                for d in det:
                    st.write(d)
                    save(d[0], d[1])

                # PDF
                if st.button("Download Report"):
                    generate_report("report.pdf", det)
                    with open("report.pdf","rb") as f:
                        st.download_button("Download", f, file_name="report.pdf")

# ======================================
# LIVE CAMERA
# ======================================
elif page == "Live Camera":
    st.title("Live Camera Detection")

    run = st.checkbox("Start Camera")

    frame = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, img = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        det, out = detect(pil)
        frame.image(out)

    cap.release()

# ======================================
# RESULTS
# ======================================
elif page == "Results":
    st.title("Analytics")

    data = fetch()

    if data:
        df = pd.DataFrame(data, columns=["Disease","Confidence","Time"])

        st.metric("Total", len(df))
        st.bar_chart(df["Disease"].value_counts())
        st.dataframe(df)

# ======================================
# ABOUT
# ======================================
elif page == "About":
    st.title("About")

    st.write("""
    AgroAI - Tomato Disease Detection  
    Built using YOLOv8 and Streamlit  
    Includes real-time detection, analytics, and reporting
    """)
