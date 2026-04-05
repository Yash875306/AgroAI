import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import hashlib
import json
import os

# ================= CONFIG =================
st.set_page_config(page_title="AgroAI", layout="wide")

# ================= CSS (PRO UI) =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
}
h1,h2,h3,h4,p {
    text-align: center;
    color: white;
}
.stButton button {
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
.stTextInput input {
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================= USER STORAGE =================
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

users = load_users()

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
    return YOLO("best.pt")   # make sure best.pt is present

model = load_model()

# ================= NAV =================
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
    st.title("🔐 Login")

    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in users and users[user] == hash_pw(pw):
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
    st.title("📝 Create Account")

    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Create Account"):
        if user in users:
            st.error("User already exists")
        else:
            users[user] = hash_pw(pw)
            save_users(users)
            st.success("Account created successfully ✅")
            go("login")

# ============================================================
# HOME
# ============================================================
elif st.session_state.page == "home":
    st.markdown(f"""
    <h1>🌿 AgroAI</h1>
    <h3>Welcome {st.session_state.username} 👋</h3>
    <p style='color:gray'>
    AI-powered Tomato Disease Detection using YOLOv8
    </p>
    """, unsafe_allow_html=True)

# ============================================================
# DETECTION
# ============================================================
elif st.session_state.page == "detect":
    st.title("🌿 Tomato Disease Detection")

    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(img, width=450)

        if st.button("Run Detection"):
            img_np = np.array(img)

            with st.spinner("Detecting..."):
                results = model(img_np)

            plotted = results[0].plot()

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(plotted, width=450)

            names = results[0].names
            boxes = results[0].boxes

            if boxes is not None:
                classes = boxes.cls.cpu().numpy()
                detected = [names[int(c)] for c in classes]

                st.success("Detected: " + ", ".join(set(detected)))

                for box in boxes:
                    conf = float(box.conf[0])
                    st.write(f"Confidence: {conf:.2f}")

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "about":
    st.title("📌 About Project")
    st.write("Final Year Project - AgroAI 🌿")
    st.write("YOLOv8 based Tomato Disease Detection System")
