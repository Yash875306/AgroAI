# =====================================
# IMPORTS
# =====================================
import streamlit as st
import sqlite3
import bcrypt
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Tomato Disease Detection",
    layout="wide"
)

# =====================================
# DARK THEME CSS
# =====================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.card {
    background-color: #161b22;
    padding: 2rem;
    border-radius: 12px;
    width: 380px;
    margin: auto;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.6);
}

.navbar {
    background-color: #161b22;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 20px;
}

button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# DATABASE (CLOUD SAFE)
# =====================================
DB_PATH = "database.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

def create_table():
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password BLOB
        )
    """)
    conn.commit()

create_table()

# =====================================
# AUTH FUNCTIONS
# =====================================
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def add_user(username, password):
    c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    data = c.fetchone()
    if data and verify_password(password, data[1]):
        return True
    return False

# =====================================
# LOAD YOLO MODEL (IMPORTANT FOR CLOUD)
# =====================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # MUST be in repo

# Lazy load (prevents crash if not needed yet)
model = None

# =====================================
# SESSION STATE
# =====================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"

# =====================================
# NAVBAR
# =====================================
def navbar():
    st.markdown('<div class="navbar">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("Home"):
        st.session_state.page = "Home"

    if col2.button("Detection"):
        st.session_state.page = "Detection"

    if col3.button("About"):
        st.session_state.page = "About"

    if col4.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# AUTH PAGE
# =====================================
def auth_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    option = st.radio("Select", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            try:
                add_user(username, password)
                st.success("Account created successfully")
            except:
                st.error("Username already exists")

    if option == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# HOME PAGE
# =====================================
def home_page():
    st.title("Tomato Disease Detection System")

    st.write("""
    This application uses YOLOv8 deep learning model to detect tomato leaf diseases.

    Features:
    - Image-based detection
    - Fast and accurate predictions
    - User authentication system

    Use this tool to identify diseases early and improve crop yield.
    """)

# =====================================
# DETECTION PAGE
# =====================================
def detection_page():
    global model

    st.title("Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, width=300)

        if st.button("Run Detection"):

            if model is None:
                with st.spinner("Loading model..."):
                    model = load_model()

            results = model(image)

            result_img = results[0].plot()

            st.image(result_img, width=500)

            st.subheader("Predictions")

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"{model.names[cls_id]} - {conf:.2f}")

# =====================================
# ABOUT PAGE
# =====================================
def about_page():
    st.title("About")

    st.write("""
    Final Year Project

    Technologies Used:
    - Streamlit
    - YOLOv8 (Ultralytics)
    - SQLite

    This system helps detect tomato plant diseases using AI.
    """)

# =====================================
# MAIN
# =====================================
if not st.session_state.logged_in:
    auth_page()
else:
    navbar()

    if st.session_state.page == "Home":
        home_page()

    elif st.session_state.page == "Detection":
        detection_page()

    elif st.session_state.page == "About":
        about_page()
