# ============================
# IMPORTS
# ============================
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import sqlite3
from datetime import datetime
import os

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AgroAI - Tomato Disease Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================
# SESSION STATE
# ============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ✅ NEW: Read page from URL (FIX)
query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]

def nav(p):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

PAGE = st.session_state.page

# ============================
# DATABASE
# ============================
DB = "agroai.db"

def init_db():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            confidence REAL,
            severity TEXT,
            timestamp TEXT
        )
    """)
    con.commit()
    con.close()

def save_detection(disease, confidence, severity):
    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO detections (disease,confidence,severity,timestamp) VALUES (?,?,?,?)",
        (disease, confidence, severity, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    con.commit()
    con.close()

def get_history():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT disease,confidence,severity,timestamp FROM detections ORDER BY id DESC"
    ).fetchall()
    con.close()
    return rows

def clear_history_db():
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM detections")
    con.commit()
    con.close()

init_db()

# ============================
# MODEL
# ============================
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        return None
    return YOLO("best.pt")

model = load_model()

def run_inference(model, pil_image):
    arr = np.array(pil_image)
    results = model.predict(arr, conf=0.25, verbose=False)
    r = results[0]

    detections = []
    annotated = pil_image

    if r.boxes and len(r.boxes):
        annotated = Image.fromarray(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
        for box in r.boxes:
            detections.append((
                model.names[int(box.cls.item())],
                float(box.conf.item())
            ))

    return detections or [("Tomato_healthy", 1.0)], annotated

# ============================
# YOUR UI (UNCHANGED)
# ============================
# ⚠️ KEEP ALL YOUR CSS EXACTLY SAME ABOVE THIS (not repeating to save space)

# ============================
# TOPBAR (ONLY SCRIPT FIXED)
# ============================
def _a(p):
    return "active" if PAGE == p else ""

st.markdown(f"""
<div class="topbar">
  <div class="tb-brand">Agro<span>AI</span></div>
  <nav class="tb-nav">
    <button class="tb-btn {_a('home')}" onclick="window._go('home')">Home</button>
    <button class="tb-btn {_a('detect')}" onclick="window._go('detect')">Detect</button>
    <button class="tb-btn {_a('results')}" onclick="window._go('results')">Results</button>
    <button class="tb-btn {_a('about')}" onclick="window._go('about')">About</button>
  </nav>
  <div class="tb-spacer"></div>
  <span class="tb-pill">YOLOv8 &nbsp;&middot;&nbsp; 96.7% mAP</span>
</div>
<div class="main-wrap"></div>

<script>
window._go = function(page) {{
    const url = new URL(window.location);
    url.searchParams.set("page", page);
    window.location.href = url.toString();
}};
</script>
""", unsafe_allow_html=True)

# ============================
# PAGES (UNCHANGED LOGIC)
# ============================

# HOME
if PAGE == "home":
    st.title("Tomato Disease Detection System")
    st.write("Upload a tomato leaf image and detect diseases.")

# DETECT
elif PAGE == "detect":
    st.title("Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        if st.button("Run Detection"):
            if model is None:
                st.error("Model file best.pt not found")
            else:
                detections, annotated = run_inference(model, img)
                st.image(annotated, width=500)

                for name, conf in detections:
                    st.write(f"{name} - {conf:.2f}")
                    save_detection(name, conf, "Medium")

# RESULTS
elif PAGE == "results":
    st.title("Detection History")

    rows = get_history()

    if not rows:
        st.info("No detections yet")
    else:
        for r in rows:
            st.write(r)

        if st.button("Clear History"):
            clear_history_db()
            st.rerun()

# ABOUT
elif PAGE == "about":
    st.title("About")
    st.write("Final Year Project using YOLOv8 and Streamlit")
