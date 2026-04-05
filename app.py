import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import hashlib

# ============ CONFIG ============
st.set_page_config(page_title="AgroAI", layout="wide")

# ============ SIMPLE AUTH ============
def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "login"

def go(p):
    st.session_state.page = p
    st.rerun()

# ============ MODEL LOAD ============
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # your trained model path

model = load_model()

# ============ NAV ============
if st.session_state.logged_in:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Home"): go("home")
    with col2:
        if st.button("Detection"): go("detect")
    with col3:
        if st.button("About"): go("about")
    with col4:
        if st.button("Logout"):
            st.session_state.logged_in = False
            go("login")

# ============================================================
# LOGIN
# ============================================================
if st.session_state.page == "login":
    st.title("Login")

    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in st.session_state.users and st.session_state.users[user] == hash_pw(pw):
            st.session_state.logged_in = True
            st.session_state.username = user
            go("home")
        else:
            st.error("Invalid credentials")

    if st.button("Go to Signup"):
        go("signup")

# ============================================================
# SIGNUP
# ============================================================
elif st.session_state.page == "signup":
    st.title("Signup")

    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Create Account"):
        if user in st.session_state.users:
            st.error("User exists")
        else:
            st.session_state.users[user] = hash_pw(pw)
            st.success("Account created")
            go("login")

# ============================================================
# HOME
# ============================================================
elif st.session_state.page == "home":
    st.title(f"Welcome {st.session_state.username} 👋")
    st.write("🌿 Tomato Disease Detection using YOLOv8")

# ============================================================
# DETECTION
# ============================================================
elif st.session_state.page == "detect":
    st.title("🌿 Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            img_np = np.array(img)

            results = model(img_np)

            plotted = results[0].plot()
            st.image(plotted, caption="Result", use_column_width=True)

            names = results[0].names
            boxes = results[0].boxes

            if boxes is not None:
                classes = boxes.cls.cpu().numpy()
                detected = [names[int(c)] for c in classes]

                st.success("Detected: " + ", ".join(set(detected)))

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "about":
    st.title("About")
    st.write("Final Year Project - AgroAI 🌿")
    st.write("YOLOv8 based Tomato Disease Detection System")
