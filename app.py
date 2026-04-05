import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import hashlib

# ============ IN-MEMORY DB ============
def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

def init_db():
    if "user_db" not in st.session_state:
        st.session_state.user_db = {}

def add_user(username, email, password):
    if username in st.session_state.user_db:
        return False
    if any(v["email"] == email for v in st.session_state.user_db.values()):
        return "email_exists"
    st.session_state.user_db[username] = {"email": email, "password": hash_pw(password)}
    return True

def check_user(username, password):
    u = st.session_state.user_db.get(username)
    return u is not None and u["password"] == hash_pw(password)

# ============ PAGE CONFIG ============
st.set_page_config(page_title="AgroAI", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

# ============ SESSION INIT ============
init_db()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

def go(page):
    st.session_state.current_page = page
    st.rerun()

query = st.session_state.current_page
if st.session_state.logged_in and query in ["login", "signup"]:
    query = "home"
    st.session_state.current_page = "home"

# ============ CSS ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.stApp { background: #0a1628 !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAVBAR ── */
.navbar {
    position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 9999;
    background: rgba(10,22,40,0.97);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    display: flex; align-items: center; padding: 0 40px; gap: 8px;
}
.nav-logo { color: white; font-size: 18px; font-weight: 700; margin-right: 24px; white-space: nowrap; }
.nav-logo span { color: #4f8ef7; }
.nav-spacer { flex: 1; }
.nav-link {
    color: rgba(255,255,255,0.55); font-size: 14px; font-weight: 500;
    padding: 7px 14px; border-radius: 7px; cursor: pointer;
    border: none; background: transparent;
    transition: all 0.15s; white-space: nowrap;
}
.nav-link:hover { color: white; background: rgba(255,255,255,0.06); }
.nav-link.active { color: #4f8ef7; background: rgba(79,142,247,0.12); }
.nav-btn-primary {
    background: #4f8ef7; color: white; font-size: 14px; font-weight: 600;
    padding: 8px 20px; border-radius: 8px; border: none; cursor: pointer;
    transition: background 0.15s; white-space: nowrap;
}
.nav-btn-primary:hover { background: #3a7de8; }
.nav-user {
    color: rgba(255,255,255,0.7); font-size: 13px;
    padding: 7px 14px; border-radius: 8px;
    background: rgba(79,142,247,0.08);
    border: 1px solid rgba(79,142,247,0.18); white-space: nowrap;
}
.nav-btn-logout {
    color: #fca5a5; font-size: 13px; font-weight: 500;
    padding: 7px 14px; border-radius: 7px; cursor: pointer;
    border: 1px solid rgba(239,68,68,0.2);
    background: rgba(239,68,68,0.08); transition: all 0.15s;
}
.nav-btn-logout:hover { background: rgba(239,68,68,0.15); }

.main-wrap { margin-top: 64px; }

/* ── INPUTS ── */
.stTextInput > label { color: rgba(255,255,255,0.5) !important; font-size: 13px !important; font-weight: 500 !important; }
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important; color: white !important; font-size: 15px !important;
}
.stTextInput input:focus { border-color: #4f8ef7 !important; box-shadow: 0 0 0 3px rgba(79,142,247,0.1) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: #4f8ef7 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 11px !important; font-size: 15px !important;
    font-weight: 600 !important; width: 100% !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #3a7de8 !important; }

/* ── AUTH ── */
.auth-outer {
    min-height: calc(100vh - 64px);
    display: flex; align-items: center; justify-content: center; padding: 40px 20px;
}
.auth-card {
    background: #0d1f3c; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 40px; width: 100%; max-width: 400px;
}
.auth-title { color: white; font-size: 24px; font-weight: 700; margin-bottom: 6px; }
.auth-sub { color: rgba(255,255,255,0.35); font-size: 14px; margin-bottom: 24px; }
.auth-divider { height: 1px; background: rgba(255,255,255,0.06); margin: 18px 0; }
.auth-footer-txt { color: rgba(255,255,255,0.3); font-size: 14px; text-align: center; margin-bottom: 8px; }

/* ── HOME ── */
.hero { text-align: center; padding: 80px 20px 56px; max-width: 700px; margin: 0 auto; }
.hero-badge {
    display: inline-block; background: rgba(79,142,247,0.08); color: #4f8ef7;
    border: 1px solid rgba(79,142,247,0.2); padding: 5px 16px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 24px;
}
.hero-title { color: white; font-size: 52px; font-weight: 700; line-height: 1.1; letter-spacing: -2px; margin-bottom: 18px; }
.hero-title span { color: #4f8ef7; }
.hero-sub { color: rgba(255,255,255,0.38); font-size: 17px; line-height: 1.8; }
.stats-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; padding: 0 48px; margin: 48px 0; }
.stat-card { background: #0d1f3c; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 24px; text-align: center; }
.stat-num { color: #4f8ef7; font-size: 30px; font-weight: 700; margin-bottom: 6px; }
.stat-lbl { color: rgba(255,255,255,0.3); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.section { padding: 0 48px; margin-bottom: 56px; }
.section-title { color: white !important; font-size: 20px; font-weight: 600; margin-bottom: 16px; }
.disease-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 8px; }
.disease-item {
    background: #0d1f3c; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 13px 16px;
    color: rgba(255,255,255,0.6); font-size: 14px;
    display: flex; align-items: center; gap: 10px;
}
.disease-dot { width: 6px; height: 6px; background: #4f8ef7; border-radius: 50%; flex-shrink: 0; }

/* ── DETECTION ── */
.page-wrap { padding: 40px 48px; }
.detect-card { background: #0d1f3c; border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 24px; }
.card-label { color: rgba(255,255,255,0.28); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 16px; }
.result-item { background: rgba(79,142,247,0.07); border: 1px solid rgba(79,142,247,0.15); border-radius: 10px; padding: 14px 18px; margin: 8px 0; }
.result-name { color: white; font-size: 15px; font-weight: 600; margin-bottom: 4px; }
.result-conf { color: rgba(255,255,255,0.38); font-size: 13px; }

/* ── RESULTS ── */
.perf-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 32px; }
.perf-card { background: #0d1f3c; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 24px; text-align: center; }
.perf-num { color: #4f8ef7; font-size: 28px; font-weight: 700; margin-bottom: 6px; }
.perf-lbl { color: rgba(255,255,255,0.3); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }

/* ── ABOUT ── */
.about-card { background: #0d1f3c; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 24px 28px; margin-bottom: 12px; }
.about-card h3 { color: white !important; font-size: 16px; font-weight: 600; margin-bottom: 10px; }
.about-card p { color: rgba(255,255,255,0.45) !important; font-size: 14px; line-height: 1.9; }

/* ── FOOTER ── */
.footer { text-align: center; padding: 28px; color: rgba(255,255,255,0.15); font-size: 13px; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 48px; }

[data-testid="stFileUploader"] { background: rgba(255,255,255,0.02) !important; border: 2px dashed rgba(79,142,247,0.2) !important; border-radius: 10px !important; }
h1,h2,h3,p { color: white !important; }
a { text-decoration: none !important; }
</style>
""", unsafe_allow_html=True)

# ============ NAVBAR (Pure HTML — no Streamlit buttons) ============
nav_items = [("home","Home"), ("detection","Detection"), ("results","Results"), ("about","About")]
nav_links_html = ""
for pg, label in nav_items:
    active = "active" if query == pg else ""
    nav_links_html += f'<a class="nav-link {active}" href="?nav={pg}" onclick="void(0)">{label}</a>'

if st.session_state.logged_in:
    right_html = f'<span class="nav-user">👤 {st.session_state.username}</span>'
else:
    right_html = ""

st.markdown(f"""
<div class="navbar">
  <div class="nav-logo">🌿 Agro<span>AI</span></div>
  <div class="nav-spacer"></div>
  {nav_links_html}
  {right_html}
</div>
<div class="main-wrap">
""", unsafe_allow_html=True)

# ── Hidden nav triggers using Streamlit columns (invisible, keyboard-only fallback) ──
# Use st.session_state buttons in a hidden row
with st.container():
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1:
        if st.button("⌂ Home", key="b_home"): go("home")
    with c2:
        if st.button("🔬 Detection", key="b_detect"): go("detection")
    with c3:
        if st.button("📊 Results", key="b_results"): go("results")
    with c4:
        if st.button("ℹ️ About", key="b_about"): go("about")
    with c5:
        if not st.session_state.logged_in:
            if st.button("🔑 Login", key="b_login"): go("login")
    with c6:
        if st.session_state.logged_in:
            if st.button("🚪 Logout", key="b_logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                go("login")

# Style those nav buttons to be minimal / subtle
st.markdown("""
<style>
/* Make the functional nav buttons look like clean nav links */
[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    background: transparent !important;
    color: rgba(255,255,255,0.55) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    border-radius: 7px !important;
    border: none !important;
    width: auto !important;
}
[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ============ PAGES ============

# ── LOGIN ──
if query == "login":
    st.markdown('<div class="auth-outer">', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div class="auth-card">
          <div class="auth-title">Welcome Back 👋</div>
          <div class="auth-sub">Sign in to access AgroAI</div>
          <div class="auth-divider"></div>
        </div>""", unsafe_allow_html=True)
        username = st.text_input("Username", key="l_user", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="l_pass", placeholder="Enter your password")
        if st.button("Sign In →", key="login_btn"):
            if not username or not password:
                st.error("Please fill all fields")
            elif check_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                go("home")
            else:
                st.error("❌ Invalid username or password")
        st.markdown('<div class="auth-divider"></div><div class="auth-footer-txt">Don\'t have an account?</div>', unsafe_allow_html=True)
        if st.button("Create Account", key="goto_signup"):
            go("signup")
    st.markdown('</div>', unsafe_allow_html=True)

# ── SIGNUP ──
elif query == "signup":
    st.markdown('<div class="auth-outer">', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div class="auth-card">
          <div class="auth-title">Create Account 🌱</div>
          <div class="auth-sub">Join AgroAI — it's free!</div>
          <div class="auth-divider"></div>
        </div>""", unsafe_allow_html=True)
        new_user  = st.text_input("Username", key="s_user", placeholder="Choose a username")
        new_email = st.text_input("Email", key="s_email", placeholder="your@email.com")
        new_pass  = st.text_input("Password", type="password", key="s_pass", placeholder="Min 6 characters")
        conf_pass = st.text_input("Confirm Password", type="password", key="s_conf", placeholder="Repeat password")
        if st.button("Create Account →", key="signup_btn"):
            if not new_user or not new_email or not new_pass:
                st.error("Please fill all fields")
            elif new_pass != conf_pass:
                st.error("Passwords do not match")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters")
            else:
                result = add_user(new_user, new_email, new_pass)
                if result == "email_exists":
                    st.error("Email already registered")
                elif result:
                    st.success("✅ Account created! Please sign in.")
                    go("login")
                else:
                    st.error("Username already taken")
        st.markdown('<div class="auth-divider"></div><div class="auth-footer-txt">Already have an account?</div>', unsafe_allow_html=True)
        if st.button("Sign In", key="goto_login"):
            go("login")
    st.markdown('</div>', unsafe_allow_html=True)

# ── HOME ──
elif query == "home":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">✨ Powered by YOLOv8 Deep Learning</div>
        <div class="hero-title">Tomato Disease<br><span>Detection System</span></div>
        <div class="hero-sub">Upload a tomato leaf image and get instant AI-powered disease diagnosis with 96.7% accuracy</div>
    </div>
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-num">96.7%</div><div class="stat-lbl">Accuracy</div></div>
        <div class="stat-card"><div class="stat-num">10</div><div class="stat-lbl">Diseases</div></div>
        <div class="stat-card"><div class="stat-num">10,853</div><div class="stat-lbl">Training Images</div></div>
        <div class="stat-card"><div class="stat-num">3.6ms</div><div class="stat-lbl">Inference Speed</div></div>
    </div>
    """, unsafe_allow_html=True)
    diseases = [
        "🦠 Tomato Bacterial Spot",       "🍂 Tomato Early Blight",
        "💧 Tomato Late Blight",           "🌫️ Tomato Leaf Mold",
        "🔵 Tomato Septoria Leaf Spot",    "🕷️ Tomato Spider Mites",
        "🎯 Tomato Target Spot",           "🟡 Tomato Yellow Leaf Curl Virus",
        "✅ Tomato Healthy",               "🧬 Tomato Mosaic Virus"
    ]
    items = "".join([f'<div class="disease-item"><div class="disease-dot"></div>{d}</div>' for d in diseases])
    st.markdown(f'<div class="section"><div class="section-title">🔍 Detectable Diseases</div><div class="disease-grid">{items}</div></div>', unsafe_allow_html=True)

# ── DETECTION ──
elif query == "detection":
    if not st.session_state.logged_in:
        go("login")
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔬 Disease Detection</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="detect-card"><div class="card-label">📷 Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['jpg','jpeg','png'], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        else:
            st.markdown("<p style='color:rgba(255,255,255,0.2);text-align:center;padding:60px 0;font-size:14px;'>📁 Drag & drop or click to upload</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="detect-card"><div class="card-label">🤖 AI Detection Result</div>', unsafe_allow_html=True)
        if uploaded_file:
            with st.spinner("🔍 Analyzing leaf..."):
                model = YOLO('best.pt')
                img_np = np.array(image)
                results = model.predict(img_np, conf=0.25)
                result_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                st.image(result_img, use_column_width=True)
            found = False
            for r in results:
                for box in r.boxes:
                    found = True
                    cls  = model.names[int(box.cls)]
                    conf = float(box.conf)
                    st.markdown(f'<div class="result-item"><div class="result-name">🦠 {cls}</div><div class="result-conf">Confidence: {conf:.2%}</div></div>', unsafe_allow_html=True)
            if not found:
                st.success("✅ Leaf appears healthy! No disease detected.")
        else:
            st.markdown("<p style='color:rgba(255,255,255,0.2);text-align:center;padding:60px 0;font-size:14px;'>← Upload an image to begin</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── RESULTS ──
elif query == "results":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="perf-grid">
        <div class="perf-card"><div class="perf-num">96.7%</div><div class="perf-lbl">mAP50</div></div>
        <div class="perf-card"><div class="perf-num">3.6ms</div><div class="perf-lbl">Speed</div></div>
        <div class="perf-card"><div class="perf-num">1,960</div><div class="perf-lbl">Test Images</div></div>
        <div class="perf-card"><div class="perf-num">5</div><div class="perf-lbl">Epochs</div></div>
    </div>""", unsafe_allow_html=True)
    classes = [
        ("Tomato Bacterial Spot","94.1%","95.3%"),("Tomato Early Blight","94.4%","95.7%"),
        ("Tomato Late Blight","93.2%","94.1%"),("Tomato Leaf Mold","91.5%","92.8%"),
        ("Tomato Septoria Leaf Spot","92.7%","93.4%"),("Tomato Spider Mites","90.3%","91.6%"),
        ("Tomato Target Spot","89.8%","90.5%"),("Tomato Yellow Leaf Curl Virus","51.2%","96.5%"),
        ("Tomato Mosaic Virus","86.2%","96.6%"),("Tomato Healthy","99.7%","99.5%"),
    ]
    rows = "".join([
        f'<div class="disease-item" style="justify-content:space-between;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:center;gap:10px;"><div class="disease-dot"></div>{n}</div>'
        f'<span style="color:#4f8ef7;font-weight:600;font-size:13px;">P:{p} · R:{r}</span></div>'
        for n,p,r in classes
    ])
    st.markdown(f'<div>{rows}</div></div>', unsafe_allow_html=True)

# ── ABOUT ──
elif query == "about":
    st.markdown("""
    <div class="page-wrap">
    <p class="section-title">ℹ️ About AgroAI</p>
    <div class="about-card"><h3>🌿 What is AgroAI?</h3>
    <p>AgroAI is an advanced tomato leaf disease detection system built using YOLOv8 deep learning. It enables farmers and researchers to instantly identify tomato diseases from leaf images with high accuracy.</p></div>
    <div class="about-card"><h3>⚙️ Technology Stack</h3>
    <p>YOLOv8 · Python 3.11 · PyTorch · Streamlit · OpenCV · Pillow · Streamlit Cloud</p></div>
    <div class="about-card"><h3>📁 Dataset</h3>
    <p>Trained on 10,853 annotated tomato leaf images spanning 10 disease categories from the PlantVillage dataset.</p></div>
    <div class="about-card"><h3>📖 How to Use</h3>
    <p>1. Create a free account or sign in<br>2. Go to Detection page<br>3. Upload a tomato leaf image<br>4. Get instant AI-powered diagnosis</p></div>
    </div>""", unsafe_allow_html=True)

# ============ FOOTER ============
st.markdown('<div class="footer">🌿 AgroAI — Advanced Tomato Disease Detection · Powered by YOLOv8 · Made with ❤️</div>', unsafe_allow_html=True)
