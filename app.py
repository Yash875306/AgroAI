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
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {
    font-family: 'Sora', sans-serif !important;
    box-sizing: border-box;
    margin: 0; padding: 0;
}

/* ── GLOBAL ── */
.stApp { background: #060d1a !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── INPUTS ── */
.stTextInput > label {
    color: rgba(255,255,255,0.45) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus {
    border-color: #3d7fff !important;
    box-shadow: 0 0 0 3px rgba(61,127,255,0.12) !important;
    outline: none !important;
}
.stTextInput input::placeholder { color: rgba(255,255,255,0.2) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: #3d7fff !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: #2d6ee8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(61,127,255,0.35) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(61,127,255,0.03) !important;
    border: 2px dashed rgba(61,127,255,0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(61,127,255,0.5) !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #3d7fff !important; }

/* ── ALERTS ── */
.stAlert { border-radius: 10px !important; }

/* ── MISC ── */
h1,h2,h3,h4,p,li { color: white !important; }
a { text-decoration: none !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #060d1a; }
::-webkit-scrollbar-thumb { background: rgba(61,127,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ============ NAVBAR HTML ============
nav_pages = [("home","Home"), ("detection","Detection"), ("results","Results"), ("about","About")]
links_html = ""
for pg, lbl in nav_pages:
    active = "active" if query == pg else ""
    links_html += f'<span class="nl {active}" data-page="{pg}">{lbl}</span>'

if st.session_state.logged_in:
    user_pill = f'<span class="nav-user">👤 {st.session_state.username}</span>'
else:
    user_pill = ""

st.markdown(f"""
<style>
.navbar {{
    position: fixed; top: 0; left: 0; right: 0; height: 60px; z-index: 9999;
    background: rgba(6,13,26,0.96);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: center;
    padding: 0 36px; gap: 4px;
}}
.nav-logo {{
    color: white; font-size: 17px; font-weight: 800;
    margin-right: 28px; letter-spacing: -0.5px;
    display: flex; align-items: center; gap: 8px;
}}
.nav-logo .accent {{ color: #3d7fff; }}
.nav-logo .badge {{
    background: rgba(61,127,255,0.15); color: #3d7fff;
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    padding: 2px 7px; border-radius: 4px; border: 1px solid rgba(61,127,255,0.2);
    text-transform: uppercase;
}}
.nav-spacer {{ flex: 1; }}
.nl {{
    color: rgba(255,255,255,0.45); font-size: 13px; font-weight: 500;
    padding: 6px 14px; border-radius: 7px; cursor: pointer;
    transition: all 0.15s; letter-spacing: 0.1px;
}}
.nl:hover {{ color: white; background: rgba(255,255,255,0.06); }}
.nl.active {{ color: #3d7fff; background: rgba(61,127,255,0.1); }}
.nav-user {{
    color: rgba(255,255,255,0.6); font-size: 12px; font-weight: 500;
    padding: 6px 14px; border-radius: 8px;
    background: rgba(61,127,255,0.07);
    border: 1px solid rgba(61,127,255,0.15);
    margin-left: 8px;
}}
.main-wrap {{ margin-top: 60px; }}
</style>
<div class="navbar">
  <div class="nav-logo">🌿 Agro<span class="accent">AI</span> <span class="badge">YOLOv8</span></div>
  <div class="nav-spacer"></div>
  {links_html}
  {user_pill}
</div>
<div class="main-wrap"></div>
""", unsafe_allow_html=True)

# ============ FUNCTIONAL NAV BUTTONS ============
st.markdown("""
<style>
/* Compact nav button row */
div[data-testid="stHorizontalBlock"]:first-of-type {{
    position: fixed; top: 10px; left: 180px; z-index: 10000;
    gap: 2px !important;
}}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {{
    background: transparent !important;
    color: transparent !important;
    font-size: 1px !important;
    padding: 22px 40px !important;
    border: none !important;
    box-shadow: none !important;
    width: 80px !important;
    cursor: pointer !important;
    border-radius: 7px !important;
}}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {{
    background: rgba(255,255,255,0.03) !important;
    transform: none !important;
    box-shadow: none !important;
}}
</style>
""", unsafe_allow_html=True)

with st.container():
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        if st.button("Home", key="b_home"): go("home")
    with c2:
        if st.button("Detection", key="b_detect"): go("detection")
    with c3:
        if st.button("Results", key="b_results"): go("results")
    with c4:
        if st.button("About", key="b_about"): go("about")
    with c5:
        if not st.session_state.logged_in:
            if st.button("Login", key="b_login"): go("login")
    with c6:
        if st.session_state.logged_in:
            if st.button("Logout", key="b_logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                go("login")

# ── Logout button styling (visible) ──
if st.session_state.logged_in:
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:first-of-type div:nth-child(6) .stButton > button {
        background: rgba(239,68,68,0.1) !important;
        color: #fca5a5 !important;
        border: 1px solid rgba(239,68,68,0.2) !important;
        font-size: 12px !important;
        padding: 6px 16px !important;
        width: auto !important;
        position: fixed !important;
        top: 12px !important;
        right: 36px !important;
        z-index: 10001 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stHorizontalBlock"]:first-of-type div:nth-child(6) .stButton > button:hover {
        background: rgba(239,68,68,0.2) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:first-of-type div:nth-child(5) .stButton > button {
        background: #3d7fff !important;
        color: white !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        padding: 6px 18px !important;
        width: auto !important;
        position: fixed !important;
        top: 12px !important;
        right: 36px !important;
        z-index: 10001 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stHorizontalBlock"]:first-of-type div:nth-child(5) .stButton > button:hover {
        background: #2d6ee8 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
#  PAGES
# ============================================================

# ── LOGIN PAGE ──
if query == "login":
    st.markdown("""
    <style>
    .auth-wrap {
        min-height: calc(100vh - 60px);
        display: flex; align-items: center; justify-content: center;
        padding: 40px 20px;
        background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(61,127,255,0.07) 0%, transparent 70%);
    }
    .auth-card {
        background: #0b1629;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 44px 40px;
        width: 100%; max-width: 420px;
    }
    .auth-title { color: white !important; font-size: 26px; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.5px; }
    .auth-sub { color: rgba(255,255,255,0.3) !important; font-size: 14px; margin-bottom: 28px; }
    .auth-sep { height: 1px; background: rgba(255,255,255,0.06); margin: 24px 0; }
    .auth-foot { color: rgba(255,255,255,0.25) !important; font-size: 13px; text-align: center; margin-bottom: 10px; }
    </style>
    <div class="auth-wrap">
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
        <div class="auth-card">
          <div class="auth-title">Welcome Back 👋</div>
          <div class="auth-sub">Sign in to access AgroAI</div>
          <div class="auth-sep"></div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", key="l_user", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="l_pass", placeholder="Enter your password")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("Sign In →", key="login_btn"):
            if not username or not password:
                st.error("Please fill all fields.")
            elif check_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                go("home")
            else:
                st.error("Invalid username or password.")

        st.markdown('<div class="auth-sep"></div><div class="auth-foot">Don\'t have an account?</div>', unsafe_allow_html=True)
        if st.button("Create Account", key="goto_signup"):
            go("signup")

    st.markdown('</div>', unsafe_allow_html=True)

# ── SIGNUP PAGE ──
elif query == "signup":
    st.markdown("""
    <style>
    .auth-wrap {
        min-height: calc(100vh - 60px);
        display: flex; align-items: center; justify-content: center;
        padding: 40px 20px;
        background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(61,127,255,0.07) 0%, transparent 70%);
    }
    .auth-sep { height: 1px; background: rgba(255,255,255,0.06); margin: 24px 0; }
    .auth-foot { color: rgba(255,255,255,0.25) !important; font-size: 13px; text-align: center; margin-bottom: 10px; }
    </style>
    <div class="auth-wrap">
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
        <div style="background:#0b1629;border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:44px 40px;">
          <div style="color:white;font-size:26px;font-weight:800;margin-bottom:6px;letter-spacing:-0.5px;">Create Account 🌱</div>
          <div style="color:rgba(255,255,255,0.3);font-size:14px;margin-bottom:28px;">Join AgroAI — it's free!</div>
          <div style="height:1px;background:rgba(255,255,255,0.06);margin-bottom:24px;"></div>
        </div>
        """, unsafe_allow_html=True)

        new_user  = st.text_input("Username", key="s_user", placeholder="Choose a username")
        new_email = st.text_input("Email", key="s_email", placeholder="your@email.com")
        new_pass  = st.text_input("Password", type="password", key="s_pass", placeholder="Minimum 6 characters")
        conf_pass = st.text_input("Confirm Password", type="password", key="s_conf", placeholder="Repeat password")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("Create Account →", key="signup_btn"):
            if not new_user or not new_email or not new_pass:
                st.error("Please fill all fields.")
            elif new_pass != conf_pass:
                st.error("Passwords do not match.")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                result = add_user(new_user, new_email, new_pass)
                if result == "email_exists":
                    st.error("Email already registered.")
                elif result:
                    st.success("✅ Account created! Please sign in.")
                    go("login")
                else:
                    st.error("Username already taken.")

        st.markdown('<div class="auth-sep"></div><div class="auth-foot">Already have an account?</div>', unsafe_allow_html=True)
        if st.button("Sign In", key="goto_login"):
            go("login")

    st.markdown('</div>', unsafe_allow_html=True)

# ── HOME PAGE ──
elif query == "home":
    st.markdown("""
    <style>
    .home-bg {
        background: radial-gradient(ellipse 80% 40% at 50% -5%, rgba(61,127,255,0.12) 0%, transparent 60%);
        padding-bottom: 60px;
    }
    .hero { text-align: center; padding: 72px 20px 52px; max-width: 680px; margin: 0 auto; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 7px;
        background: rgba(61,127,255,0.08); color: #5d9bff;
        border: 1px solid rgba(61,127,255,0.18);
        padding: 5px 16px; border-radius: 20px;
        font-size: 11px; font-weight: 700; letter-spacing: 1.2px;
        text-transform: uppercase; margin-bottom: 28px;
    }
    .hero-title {
        color: white !important; font-size: 54px; font-weight: 800;
        line-height: 1.08; letter-spacing: -2.5px; margin-bottom: 20px;
    }
    .hero-title .accent { color: #3d7fff; }
    .hero-sub { color: rgba(255,255,255,0.32) !important; font-size: 16px; line-height: 1.85; max-width: 520px; margin: 0 auto; }

    .stats-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; padding: 0 52px; margin: 44px 0 52px; }
    .stat-box {
        background: #0b1629; border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px; padding: 24px 16px; text-align: center;
        transition: border-color 0.2s, transform 0.2s;
    }
    .stat-box:hover { border-color: rgba(61,127,255,0.25); transform: translateY(-2px); }
    .stat-num { color: #3d7fff !important; font-size: 32px; font-weight: 800; letter-spacing: -1px; margin-bottom: 6px; }
    .stat-lbl { color: rgba(255,255,255,0.25) !important; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; }

    .section { padding: 0 52px; margin-bottom: 52px; }
    .section-head {
        color: white !important; font-size: 18px; font-weight: 700;
        margin-bottom: 18px; letter-spacing: -0.3px;
        display: flex; align-items: center; gap: 10px;
    }
    .disease-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 8px; }
    .disease-chip {
        background: #0b1629; border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px; padding: 12px 16px;
        color: rgba(255,255,255,0.5) !important; font-size: 13px;
        display: flex; align-items: center; gap: 10px;
        transition: all 0.15s;
    }
    .disease-chip:hover { border-color: rgba(61,127,255,0.2); color: rgba(255,255,255,0.75) !important; }
    .disease-dot { width: 6px; height: 6px; background: #3d7fff; border-radius: 50%; flex-shrink: 0; }

    .cta-section { padding: 0 52px; margin-bottom: 20px; }
    .cta-box {
        background: linear-gradient(135deg, rgba(61,127,255,0.12) 0%, rgba(61,127,255,0.04) 100%);
        border: 1px solid rgba(61,127,255,0.2); border-radius: 16px;
        padding: 36px; display: flex; align-items: center; justify-content: space-between;
    }
    .cta-text { color: white !important; font-size: 20px; font-weight: 700; margin-bottom: 6px; }
    .cta-sub { color: rgba(255,255,255,0.35) !important; font-size: 14px; }
    </style>
    <div class="home-bg">
      <div class="hero">
        <div class="hero-badge">✨ Powered by YOLOv8 Deep Learning</div>
        <div class="hero-title">Tomato Disease<br><span class="accent">Detection System</span></div>
        <div class="hero-sub">Upload a tomato leaf image and get instant AI-powered disease diagnosis with exceptional accuracy</div>
      </div>
      <div class="stats-row">
        <div class="stat-box"><div class="stat-num">96.7%</div><div class="stat-lbl">Accuracy</div></div>
        <div class="stat-box"><div class="stat-num">10</div><div class="stat-lbl">Diseases</div></div>
        <div class="stat-box"><div class="stat-num">10,853</div><div class="stat-lbl">Training Images</div></div>
        <div class="stat-box"><div class="stat-num">3.6ms</div><div class="stat-lbl">Inference Speed</div></div>
      </div>
    """, unsafe_allow_html=True)

    diseases = [
        "🦠 Tomato Bacterial Spot",    "🍂 Tomato Early Blight",
        "💧 Tomato Late Blight",        "🌫️ Tomato Leaf Mold",
        "🔵 Tomato Septoria Leaf Spot", "🕷️ Tomato Spider Mites",
        "🎯 Tomato Target Spot",        "🟡 Tomato Yellow Leaf Curl Virus",
        "✅ Tomato Healthy",            "🧬 Tomato Mosaic Virus",
    ]
    chips = "".join([f'<div class="disease-chip"><div class="disease-dot"></div>{d}</div>' for d in diseases])
    st.markdown(f"""
      <div class="section">
        <div class="section-head">🔍 Detectable Diseases</div>
        <div class="disease-grid">{chips}</div>
      </div>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.markdown("""
      <div class="cta-section">
        <div class="cta-box">
          <div>
            <div class="cta-text">Ready to detect diseases?</div>
            <div class="cta-sub">Create a free account and start analyzing your crops instantly.</div>
          </div>
        </div>
      </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Get Started →", key="home_cta"):
                go("signup")

    st.markdown('</div>', unsafe_allow_html=True)

# ── DETECTION PAGE ──
elif query == "detection":
    if not st.session_state.logged_in:
        go("login")

    st.markdown("""
    <style>
    .page-pad { padding: 36px 52px 52px; }
    .page-title { color: white !important; font-size: 22px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
    .page-sub { color: rgba(255,255,255,0.25) !important; font-size: 13px; margin-bottom: 28px; }
    .detect-card {
        background: #0b1629; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 24px; height: 100%;
    }
    .card-lbl {
        color: rgba(255,255,255,0.22) !important; font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 18px;
    }
    .upload-hint {
        color: rgba(255,255,255,0.15) !important; text-align: center;
        padding: 60px 0; font-size: 13px;
    }
    .result-pill {
        background: rgba(61,127,255,0.07); border: 1px solid rgba(61,127,255,0.15);
        border-radius: 12px; padding: 14px 18px; margin: 8px 0;
    }
    .result-name { color: white !important; font-size: 15px; font-weight: 700; margin-bottom: 5px; }
    .result-conf { color: rgba(255,255,255,0.35) !important; font-size: 12px; font-weight: 500; }
    .conf-bar { height: 4px; background: rgba(61,127,255,0.15); border-radius: 2px; margin-top: 10px; }
    .conf-fill { height: 4px; background: #3d7fff; border-radius: 2px; }
    </style>
    <div class="page-pad">
      <div class="page-title">🔬 Disease Detection</div>
      <div class="page-sub">Upload a clear tomato leaf image for AI-powered analysis</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div style="padding:0 52px 0 52px"><div class="detect-card"><div class="card-lbl">📷 Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['jpg','jpeg','png'], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        else:
            st.markdown('<div class="upload-hint">📁 Drag & drop or click to upload<br><span style="font-size:11px;color:rgba(255,255,255,0.1)">Supports JPG, JPEG, PNG</span></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="padding:0 52px 0 0"><div class="detect-card"><div class="card-lbl">🤖 AI Detection Results</div>', unsafe_allow_html=True)
        if uploaded_file:
            with st.spinner("Analyzing leaf..."):
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
                    pct  = int(conf * 100)
                    st.markdown(f"""
                    <div class="result-pill">
                      <div class="result-name">🦠 {cls}</div>
                      <div class="result-conf">Confidence: {conf:.2%}</div>
                      <div class="conf-bar"><div class="conf-fill" style="width:{pct}%"></div></div>
                    </div>""", unsafe_allow_html=True)
            if not found:
                st.success("✅ No disease detected — leaf appears healthy!")
        else:
            st.markdown('<div class="upload-hint">← Upload an image to begin analysis</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

# ── RESULTS PAGE ──
elif query == "results":
    st.markdown("""
    <style>
    .page-pad { padding: 36px 52px 52px; }
    .page-title { color: white !important; font-size: 22px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
    .page-sub { color: rgba(255,255,255,0.25) !important; font-size: 13px; margin-bottom: 28px; }
    .perf-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 32px; }
    .perf-box {
        background: #0b1629; border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px; padding: 24px; text-align: center;
    }
    .perf-num { color: #3d7fff !important; font-size: 30px; font-weight: 800; letter-spacing: -1px; margin-bottom: 6px; }
    .perf-lbl { color: rgba(255,255,255,0.25) !important; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; }
    .class-row {
        background: #0b1629; border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px; padding: 14px 18px;
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 8px; transition: border-color 0.15s;
    }
    .class-row:hover { border-color: rgba(61,127,255,0.2); }
    .class-name { color: rgba(255,255,255,0.55) !important; font-size: 13px; display: flex; align-items: center; gap: 10px; }
    .class-dot { width: 6px; height: 6px; background: #3d7fff; border-radius: 50%; flex-shrink: 0; }
    .class-metrics { color: #3d7fff !important; font-size: 12px; font-weight: 700; }
    .section-head { color: white !important; font-size: 16px; font-weight: 700; margin-bottom: 16px; letter-spacing: -0.3px; }
    </style>
    <div class="page-pad">
      <div class="page-title">📊 Model Performance</div>
      <div class="page-sub">Evaluation metrics across all disease categories</div>
      <div class="perf-grid">
        <div class="perf-box"><div class="perf-num">96.7%</div><div class="perf-lbl">mAP50</div></div>
        <div class="perf-box"><div class="perf-num">3.6ms</div><div class="perf-lbl">Speed</div></div>
        <div class="perf-box"><div class="perf-num">1,960</div><div class="perf-lbl">Test Images</div></div>
        <div class="perf-box"><div class="perf-num">5</div><div class="perf-lbl">Epochs</div></div>
      </div>
      <div class="section-head">Per-Class Results</div>
    """, unsafe_allow_html=True)

    classes = [
        ("Tomato Bacterial Spot",        "94.1%","95.3%"),
        ("Tomato Early Blight",          "94.4%","95.7%"),
        ("Tomato Late Blight",           "93.2%","94.1%"),
        ("Tomato Leaf Mold",             "91.5%","92.8%"),
        ("Tomato Septoria Leaf Spot",    "92.7%","93.4%"),
        ("Tomato Spider Mites",          "90.3%","91.6%"),
        ("Tomato Target Spot",           "89.8%","90.5%"),
        ("Tomato Yellow Leaf Curl Virus","51.2%","96.5%"),
        ("Tomato Mosaic Virus",          "86.2%","96.6%"),
        ("Tomato Healthy",               "99.7%","99.5%"),
    ]
    for name, prec, rec in classes:
        st.markdown(f"""
        <div class="class-row">
          <div class="class-name"><div class="class-dot"></div>{name}</div>
          <div class="class-metrics">P: {prec} &nbsp;·&nbsp; R: {rec}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── ABOUT PAGE ──
elif query == "about":
    st.markdown("""
    <style>
    .page-pad { padding: 36px 52px 52px; }
    .page-title { color: white !important; font-size: 22px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
    .page-sub { color: rgba(255,255,255,0.25) !important; font-size: 13px; margin-bottom: 28px; }
    .about-block {
        background: #0b1629; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 26px 30px; margin-bottom: 12px;
    }
    .about-block h3 { color: white !important; font-size: 15px; font-weight: 700; margin-bottom: 12px; }
    .about-block p { color: rgba(255,255,255,0.38) !important; font-size: 14px; line-height: 1.9; }
    .about-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-top: 12px; }
    .about-stat {
        background: rgba(61,127,255,0.05); border: 1px solid rgba(61,127,255,0.12);
        border-radius: 10px; padding: 20px; text-align: center;
    }
    .about-stat-num { color: #3d7fff !important; font-size: 24px; font-weight: 800; margin-bottom: 4px; }
    .about-stat-lbl { color: rgba(255,255,255,0.25) !important; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    .step { display: flex; gap: 14px; margin-bottom: 16px; align-items: flex-start; }
    .step-num {
        background: rgba(61,127,255,0.12); color: #3d7fff !important;
        width: 28px; height: 28px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; font-weight: 700; flex-shrink: 0;
    }
    .step-text { color: rgba(255,255,255,0.45) !important; font-size: 14px; line-height: 1.6; padding-top: 4px; }
    </style>
    <div class="page-pad">
      <div class="page-title">ℹ️ About AgroAI</div>
      <div class="page-sub">Advanced tomato disease detection powered by deep learning</div>

      <div class="about-block">
        <h3>🌿 What is AgroAI?</h3>
        <p>AgroAI is a tomato leaf disease detection system built on YOLOv8 deep learning. It enables farmers and researchers to instantly identify tomato diseases from leaf images with high accuracy, helping protect crops and improve agricultural outcomes.</p>
        <div class="about-grid">
          <div class="about-stat"><div class="about-stat-num">96.7%</div><div class="about-stat-lbl">Accuracy</div></div>
          <div class="about-stat"><div class="about-stat-num">10</div><div class="about-stat-lbl">Diseases</div></div>
          <div class="about-stat"><div class="about-stat-num">3.6ms</div><div class="about-stat-lbl">Inference</div></div>
        </div>
      </div>

      <div class="about-block">
        <h3>⚙️ Technology Stack</h3>
        <p>YOLOv8 &nbsp;·&nbsp; Python 3.11 &nbsp;·&nbsp; PyTorch &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; OpenCV &nbsp;·&nbsp; Pillow</p>
      </div>

      <div class="about-block">
        <h3>📁 Dataset</h3>
        <p>Trained on 10,853 annotated tomato leaf images spanning 10 disease categories from the PlantVillage dataset. The model was evaluated on 1,960 test images achieving a mAP50 of 96.7%.</p>
      </div>

      <div class="about-block">
        <h3>📖 How to Use</h3>
        <div class="step"><div class="step-num">1</div><div class="step-text">Create a free account or sign in with your existing credentials.</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-text">Navigate to the Detection page from the top navigation bar.</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-text">Upload a clear photo of a tomato leaf (JPG, JPEG, or PNG).</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-text">Receive an instant AI-powered diagnosis with confidence scores.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ============ FOOTER ============
st.markdown("""
<div style="
    text-align:center; padding:24px 20px;
    color:rgba(255,255,255,0.12); font-size:12px;
    border-top:1px solid rgba(255,255,255,0.04);
    margin-top:40px; letter-spacing:0.3px;
">
  🌿 AgroAI &nbsp;·&nbsp; Advanced Tomato Disease Detection &nbsp;·&nbsp; Powered by YOLOv8
</div>
""", unsafe_allow_html=True)
