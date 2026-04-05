import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import hashlib
import json
import os

# ================= CONFIG =================
st.set_page_config(page_title="AgroAI", layout="wide")

# ================= CSS =================
st.markdown("""
<style>
.stApp {
    background: #0b1120;
}

/* Center card */
.card {
    background: #111827;
    padding: 40px;
    border-radius: 12px;
    width: 400px;
    margin: auto;
    margin-top: 100px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

.title {
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 20px;
}

.stTextInput input {
    background: #1f2937;
    color: white;
    border-radius: 8px;
}

.stButton button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    height: 40px;
    width: 100%;
}

/* Navbar */
.nav {
    display: flex;
    justify-content: space-around;
    background: #111827;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ================= FILE STORAGE =================
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

# ================= SESSION =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "login"

def go(p):
    st.session_state.page = p
    st.rerun()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================= NAV =================
if st.session_state.logged_in:
    st.markdown('<div class="nav">', unsafe_allow_html=True)

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
            st.session_state.username = ""
            go("login")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# LOGIN
# ============================================================
if st.session_state.page == "login":
    users = load_users()  # 🔥 IMPORTANT FIX

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Login</div>', unsafe_allow_html=True)

    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in users and users[user] == hash_pw(pw):
            st.session_state.logged_in = True
            st.session_state.username = user
            go("home")
        else:
            st.error("Invalid credentials")

    if st.button("Create account"):
        go("signup")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SIGNUP
# ============================================================
elif st.session_state.page == "signup":
    users = load_users()  # 🔥 IMPORTANT

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Create Account</div>', unsafe_allow_html=True)

    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        if user in users:
            st.error("User already exists")
        else:
            users[user] = hash_pw(pw)
            save_users(users)
            st.success("Account created")
            go("login")

    if st.button("Back to login"):
        go("login")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# HOME
# ============================================================
elif st.session_state.page == "home":
    st.markdown(f"""
    <h2 style='text-align:center;color:white'>
    Welcome {st.session_state.username}
    </h2>
    <p style='text-align:center;color:gray'>
    Tomato disease detection system using YOLOv8
    </p>
    """, unsafe_allow_html=True)

# ============================================================
# DETECTION
# ============================================================
elif st.session_state.page == "detect":
    st.title("Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(img, width=400)

        if st.button("Run Detection"):
            img_np = np.array(img)

            with st.spinner("Processing..."):
                results = model(img_np)

            plotted = results[0].plot()

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(plotted, width=400)

            names = results[0].names
            boxes = results[0].boxes

            if boxes is not None:
                classes = boxes.cls.cpu().numpy()
                detected = [names[int(c)] for c in classes]

                st.success(", ".join(set(detected)))

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "about":
    st.title("About")
    st.write("AgroAI - Final Year Project")
    st.write("YOLOv8 based plant disease detection system")
