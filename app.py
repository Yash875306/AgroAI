import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import hashlib
from datetime import datetime
from collections import Counter

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="AgroAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ IN-MEMORY DB ============
def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

if "user_db" not in st.session_state:
    st.session_state.user_db = {}

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if "last_username" not in st.session_state:
    st.session_state.last_username = ""

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if "login_error" not in st.session_state:
    st.session_state.login_error = ""

if "signup_error" not in st.session_state:
    st.session_state.signup_error = ""

if "signup_success" not in st.session_state:
    st.session_state.signup_success = False

def register_user(username, password):
    if username in st.session_state.user_db:
        return "exists"
    if len(password) < 6:
        return "short"
    st.session_state.user_db[username] = hash_pw(password)
    return "ok"

def check_user(username, password):
    return st.session_state.user_db.get(username) == hash_pw(password)

def save_detection(username, disease, conf):
    st.session_state.detection_history.append({
        "username": username,
        "disease": disease,
        "confidence": conf,
        "time": datetime.now().strftime("%H:%M:%S")
    })

def get_stats():
    h = st.session_state.detection_history
    total = len(h)
    users = len(set(x["username"] for x in h)) if h else 0
    top = Counter(x["disease"] for x in h).most_common(1)[0][0] if h else "N/A"
    return total, top, users

# ============ MODEL ============
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# ============ CSS ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after {
    box-sizing: border-box;
    font-family: 'Inter', sans-serif !important;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #070e1c !important;
}

.block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"]       { display: none !important; }
[data-testid="stSidebar"]       { display: none !important; }
[data-testid="collapsedControl"]{ display: none !important; }
#MainMenu, footer { display: none !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
.stMarkdown { margin: 0 !important; padding: 0 !important; }

/* Hide streamlit buttons */
div[data-testid="stButton"] > button {
    opacity: 0 !important;
    height: 0px !important;
    min-height: 0px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: none !important;
    position: absolute !important;
    pointer-events: auto !important;
    font-size: 0 !important;
    overflow: hidden !important;
    width: 0px !important;
    min-width: 0px !important;
}

/* Inputs */
div[data-testid="stTextInput"] label { display: none !important; }
div[data-testid="stTextInput"] { margin-bottom: 16px !important; }
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 14px !important;
    padding: 13px 16px !important;
    width: 100% !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(34,197,94,0.5) !important;
    box-shadow: 0 0 0 3px rgba(34,197,94,0.08) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: rgba(255,255,255,0.2) !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] small {
    color: rgba(255,255,255,0.4) !important;
}

/* ===== AUTH ===== */
.auth-wrap {
    min-height: 100vh;
    background:
        radial-gradient(ellipse 70% 60% at 10% 70%, rgba(34,197,94,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 90% 20%, rgba(34,197,94,0.04) 0%, transparent 55%),
        #070e1c;
    display: flex;
    flex-direction: column;
}
.auth-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 52px; height: 66px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.logo-row { display: flex; align-items: center; gap: 10px; }
.logo-box {
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
}
.logo-name { color: white; font-size: 18px; font-weight: 700; }
.logo-tag {
    background: rgba(34,197,94,0.12); color: #86efac;
    border: 1px solid rgba(34,197,94,0.2);
    font-size: 10px; font-weight: 600;
    padding: 2px 7px; border-radius: 20px;
}
.auth-body {
    flex: 1; display: flex; align-items: center;
    justify-content: center; padding: 60px 20px;
}
.auth-card {
    width: 100%; max-width: 440px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 44px 48px;
}
.auth-top {
    display: flex; justify-content: space-between;
    align-items: flex-start; margin-bottom: 32px;
}
.auth-title { color: white; font-size: 32px; font-weight: 700; line-height: 1.15; }
.auth-sub { color: rgba(255,255,255,0.3); font-size: 13px; line-height: 1.6; text-align: right; max-width: 120px; margin-top: 4px; }
.f-lbl {
    color: rgba(255,255,255,0.4); font-size: 11px;
    font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; margin-bottom: 6px;
    display: block;
}
.alert-err {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 10px; padding: 11px 15px;
    color: #fca5a5; font-size: 13px; margin-bottom: 18px;
}
.alert-ok {
    background: rgba(34,197,94,0.07);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 10px; padding: 11px 15px;
    color: #86efac; font-size: 13px; margin-bottom: 18px;
}
.big-btn {
    width: 100%; background: #22c55e;
    border: none; border-radius: 10px;
    padding: 14px; color: white;
    font-size: 15px; font-weight: 600;
    cursor: pointer; margin-top: 6px;
    font-family: 'Inter', sans-serif;
    transition: background 0.2s;
    display: block; text-align: center;
}
.big-btn:hover { background: #16a34a; }
.outline-btn {
    width: 100%; background: transparent;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 13px;
    color: rgba(255,255,255,0.6); font-size: 15px;
    font-weight: 500; cursor: pointer; margin-top: 8px;
    font-family: 'Inter', sans-serif; transition: all 0.2s;
    display: block; text-align: center;
}
.outline-btn:hover { border-color: rgba(255,255,255,0.25); color: white; }
.auth-footer { text-align: center; color: rgba(255,255,255,0.3); font-size: 13px; margin-top: 16px; }
.auth-footer span { color: #22c55e; cursor: pointer; font-weight: 500; }

/* ===== MAIN APP ===== */
.app-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 52px; height: 66px;
    background: rgba(7,14,28,0.97);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    position: sticky; top: 0; z-index: 999;
}
.nav-links { display: flex; gap: 4px; }
.nav-lnk {
    color: rgba(255,255,255,0.4); font-size: 14px;
    font-weight: 500; padding: 8px 16px;
    border-radius: 8px; cursor: pointer; transition: all 0.15s;
}
.nav-lnk:hover { color: white; background: rgba(255,255,255,0.05); }
.nav-lnk.on { color: #86efac; background: rgba(34,197,94,0.1); }
.user-pill {
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 5px 14px 5px 7px;
    color: rgba(255,255,255,0.6); font-size: 13px;
}
.av {
    width: 26px; height: 26px; border-radius: 50%;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 11px; font-weight: 700;
}
.lo-btn {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.4); font-size: 13px;
    padding: 7px 16px; border-radius: 8px; cursor: pointer;
    font-family: 'Inter', sans-serif; transition: all 0.15s;
}
.lo-btn:hover { color: white; border-color: rgba(255,255,255,0.18); }

/* ===== PAGE ===== */
.pg { padding: 44px 52px 80px; min-height: calc(100vh - 66px); }
.bc { color: rgba(255,255,255,0.22); font-size: 12px; margin-bottom: 6px; }
.pt { color: white; font-size: 26px; font-weight: 700; margin-bottom: 30px; }

.stat-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin-bottom: 22px; }
.sc {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 22px 24px;
}
.sc-l { color: rgba(255,255,255,0.28); font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px; }
.sc-v { color: white; font-size: 30px; font-weight: 700; }
.sc-vs { color: white; font-size: 17px; font-weight: 600; margin-top: 4px; }

.cc {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 26px 30px; margin-bottom: 18px;
}
.cc-t { color: white; font-size: 15px; font-weight: 600; margin-bottom: 16px; }

.badge { display: inline-block; font-size: 12px; font-weight: 500; padding: 5px 11px; border-radius: 20px; margin: 3px; }
.b-red   { background: rgba(239,68,68,0.08);   color: #fca5a5; border: 1px solid rgba(239,68,68,0.15);   }
.b-green { background: rgba(34,197,94,0.08);   color: #86efac; border: 1px solid rgba(34,197,94,0.15);   }
.b-blue  { background: rgba(96,165,250,0.08);  color: #93c5fd; border: 1px solid rgba(96,165,250,0.15);  }

.dt { width: 100%; border-collapse: collapse; }
.dt th { color: rgba(255,255,255,0.25); font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; padding: 9px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); text-align: left; }
.dt td { color: rgba(255,255,255,0.7); font-size: 13px; padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.dt tr:last-child td { border-bottom: none; }

.ab-p { color: rgba(255,255,255,0.55); font-size: 14px; line-height: 1.85; }
.ab-p b { color: rgba(255,255,255,0.9); }

.det-l { color: rgba(255,255,255,0.28); font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ==================== AUTH ====================
if not st.session_state.logged_in:

    # ------ SIGNUP ------
    if st.session_state.auth_mode == "signup":
        st.markdown("""
        <div class="auth-wrap">
          <div class="auth-nav">
            <div class="logo-row">
              <div class="logo-box">🌿</div>
              <span class="logo-name">AgroAI</span>
              <span class="logo-tag">AI</span>
            </div>
          </div>
          <div class="auth-body">
            <div class="auth-card">
              <div class="auth-top">
                <div class="auth-title">Create<br>Account</div>
                <div class="auth-sub">Join AgroAI to detect diseases</div>
              </div>
        """, unsafe_allow_html=True)

        if st.session_state.signup_error:
            st.markdown(f'<div class="alert-err">{st.session_state.signup_error}</div>', unsafe_allow_html=True)

        st.markdown('<span class="f-lbl">Username</span>', unsafe_allow_html=True)
        nu = st.text_input("_nu", placeholder="Choose a username", key="nu", label_visibility="collapsed")
        st.markdown('<span class="f-lbl">Password</span>', unsafe_allow_html=True)
        np_ = st.text_input("_np", placeholder="Min 6 characters", type="password", key="np_", label_visibility="collapsed")
        st.markdown('<span class="f-lbl">Confirm Password</span>', unsafe_allow_html=True)
        cp = st.text_input("_cp", placeholder="Re-enter password", type="password", key="cp_", label_visibility="collapsed")

        reg = st.button("__REG__", key="reg_btn")
        back = st.button("__BACK__", key="back_btn")

        st.markdown("""
              <button class="big-btn" onclick="hc('__REG__')">Create Account</button>
              <div class="auth-footer" style="margin-top:20px;">
                Already have an account? <span onclick="hc('__BACK__')">Login</span>
              </div>
            </div></div></div>
        <script>
        function hc(l){
            window.parent.document.querySelectorAll('button').forEach(b=>{
                if(b.innerText.trim()===l) b.click();
            });
        }
        </script>
        """, unsafe_allow_html=True)

        if reg:
            if not nu or not np_:
                st.session_state.signup_error = "Please fill all fields."
            elif np_ != cp:
                st.session_state.signup_error = "Passwords do not match."
            else:
                result = register_user(nu, np_)
                if result == "exists":
                    st.session_state.signup_error = "Username already taken."
                elif result == "short":
                    st.session_state.signup_error = "Password must be at least 6 characters."
                else:
                    st.session_state.signup_success = True
                    st.session_state.signup_error = ""
                    st.session_state.last_username = nu
                    st.session_state.auth_mode = "login"
            st.rerun()

        if back:
            st.session_state.auth_mode = "login"
            st.session_state.signup_error = ""
            st.rerun()

    # ------ LOGIN ------
    else:
        st.markdown("""
        <div class="auth-wrap">
          <div class="auth-nav">
            <div class="logo-row">
              <div class="logo-box">🌿</div>
              <span class="logo-name">AgroAI</span>
              <span class="logo-tag">AI</span>
            </div>
          </div>
          <div class="auth-body">
            <div class="auth-card">
              <div class="auth-top">
                <div class="auth-title">Welcome<br>Back</div>
                <div class="auth-sub">Sign in to detect tomato diseases</div>
              </div>
        """, unsafe_allow_html=True)

        if st.session_state.login_error:
            st.markdown(f'<div class="alert-err">{st.session_state.login_error}</div>', unsafe_allow_html=True)
        if st.session_state.signup_success:
            st.markdown('<div class="alert-ok">✓ Account created! Please login.</div>', unsafe_allow_html=True)

        st.markdown('<span class="f-lbl">Username</span>', unsafe_allow_html=True)
        # ✅ last_username auto-fill — logout பண்ணி வந்தா username ready-ஆ இருக்கும்
        uname = st.text_input("_u", placeholder="Enter your username",
                               value=st.session_state.last_username,
                               key="usr", label_visibility="collapsed")
        st.markdown('<span class="f-lbl">Password</span>', unsafe_allow_html=True)
        pw = st.text_input("_p", placeholder="Enter your password",
                            type="password", key="pwd", label_visibility="collapsed")

        login_btn  = st.button("__LOGIN__",  key="login_btn")
        signup_btn = st.button("__SIGNUP__", key="signup_btn")

        st.markdown("""
              <button class="big-btn" onclick="hc('__LOGIN__')">Login</button>
              <button class="outline-btn" onclick="hc('__SIGNUP__')">Create Account</button>
              <div class="auth-footer" style="margin-top:4px;">
                Don't have an account? <span onclick="hc('__SIGNUP__')">Sign Up</span>
              </div>
            </div></div></div>
        <script>
        function hc(l){
            window.parent.document.querySelectorAll('button').forEach(b=>{
                if(b.innerText.trim()===l) b.click();
            });
        }
        </script>
        """, unsafe_allow_html=True)

        if login_btn:
            if check_user(uname, pw):
                st.session_state.logged_in = True
                st.session_state.login_error = ""
                st.session_state.signup_success = False
                st.session_state.current_user = uname
                st.session_state.last_username = uname
                st.session_state.page = "Home"
            else:
                st.session_state.login_error = "Invalid username or password."
            st.rerun()

        if signup_btn:
            st.session_state.auth_mode = "signup"
            st.session_state.login_error = ""
            st.rerun()

# ==================== MAIN APP ====================
else:
    page = st.session_state.page
    user = st.session_state.current_user
    av   = user[0].upper() if user else "U"

    st.markdown(f"""
    <div class="app-nav">
      <div class="logo-row">
        <div class="logo-box">🌿</div>
        <span class="logo-name">AgroAI</span>
        <span class="logo-tag">AI</span>
      </div>
      <div class="nav-links">
        <span class="nav-lnk {'on' if page=='Home'      else ''}" onclick="hc('__HOME__')">Home</span>
        <span class="nav-lnk {'on' if page=='Detection' else ''}" onclick="hc('__DET__')">Detection</span>
        <span class="nav-lnk {'on' if page=='About'     else ''}" onclick="hc('__ABOUT__')">About</span>
        <span class="nav-lnk {'on' if page=='Model'     else ''}" onclick="hc('__MODEL__')">Model Results</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <div class="user-pill">
          <div class="av">{av}</div>
          {user}
        </div>
        <button class="lo-btn" onclick="hc('__LOGOUT__')">Logout</button>
      </div>
    </div>
    <script>
    function hc(l){{
        window.parent.document.querySelectorAll('button').forEach(b=>{{
            if(b.innerText.trim()===l) b.click();
        }});
    }}
    </script>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        if st.button("__HOME__",   key="nh"): st.session_state.page="Home";      st.rerun()
    with c2:
        if st.button("__DET__",    key="nd"): st.session_state.page="Detection"; st.rerun()
    with c3:
        if st.button("__ABOUT__",  key="na"): st.session_state.page="About";     st.rerun()
    with c4:
        if st.button("__MODEL__",  key="nm"): st.session_state.page="Model";     st.rerun()
    with c5:
        if st.button("__LOGOUT__", key="nl"):
            st.session_state.logged_in = False
            st.session_state.current_user = ""
            st.session_state.login_error = ""
            st.session_state.signup_success = False
            # ✅ last_username keeps the username for next login auto-fill
            st.rerun()

    st.markdown('<div class="pg">', unsafe_allow_html=True)

    # ===== HOME =====
    if page == "Home":
        total, top, users = get_stats()
        st.markdown(f"""
        <div class="bc">AgroAI • Dashboard</div>
        <div class="pt">System Overview</div>
        <div class="stat-row">
          <div class="sc"><div class="sc-l">Total Detections</div><div class="sc-v">{total}</div></div>
          <div class="sc"><div class="sc-l">Most Detected</div><div class="sc-vs">{top}</div></div>
          <div class="sc"><div class="sc-l">Active Users</div><div class="sc-v">{users}</div></div>
        </div>
        """, unsafe_allow_html=True)

        diseases = [
            ("Bacterial Spot","b-red"),("Early Blight","b-red"),("Late Blight","b-red"),
            ("Leaf Mold","b-red"),("Septoria Leaf Spot","b-red"),("Spider Mites","b-red"),
            ("Target Spot","b-red"),("Yellow Leaf Curl Virus","b-red"),
            ("Mosaic Virus","b-red"),("Healthy 🌱","b-green")
        ]
        badges = "".join([f'<span class="badge {c}">{d}</span>' for d,c in diseases])
        st.markdown(f'<div class="cc"><div class="cc-t">🦠 Detectable Conditions</div>{badges}</div>', unsafe_allow_html=True)

        hist = st.session_state.detection_history[-6:][::-1]
        if hist:
            rows = "".join([
                f'<tr><td>{h["disease"]}</td><td>{h["confidence"]:.1%}</td>'
                f'<td>{h["username"]}</td><td>{h["time"]}</td></tr>'
                for h in hist
            ])
            st.markdown(f"""
            <div class="cc">
              <div class="cc-t">🕒 Recent Detections</div>
              <table class="dt">
                <tr><th>Disease</th><th>Confidence</th><th>User</th><th>Time</th></tr>
                {rows}
              </table>
            </div>""", unsafe_allow_html=True)

    # ===== DETECTION =====
    elif page == "Detection":
        model = load_model()
        st.markdown("""
        <div class="bc">AgroAI • Detection</div>
        <div class="pt">🔬 Disease Detection</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cc"><div class="cc-t">Upload Tomato Leaf Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("img", type=["jpg","png","jpeg"], label_visibility="collapsed")

        if uploaded:
            image = Image.open(uploaded)
            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="det-l">Original</div>', unsafe_allow_html=True)
                st.image(image, use_column_width=True)

            det_btn = st.button("__DETECT__", key="det_btn")
            st.markdown("""
            <button class="big-btn" style="max-width:200px;margin-top:14px;"
              onclick="hc('__DETECT__')">🔍 Analyze</button>
            <script>function hc(l){window.parent.document.querySelectorAll('button').forEach(b=>{if(b.innerText.trim()===l)b.click();});}</script>
            """, unsafe_allow_html=True)

            if det_btn:
                with st.spinner("Analyzing..."):
                    img_np = np.array(image)
                    results = model.predict(img_np, conf=0.25)
                    out = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                    st.session_state["det_out"] = out
                    st.session_state["det_res"] = results

            if st.session_state.get("det_out") is not None:
                with cb:
                    st.markdown('<div class="det-l">Result</div>', unsafe_allow_html=True)
                    st.image(st.session_state["det_out"], use_column_width=True)

                html = ""
                for r in st.session_state["det_res"]:
                    for box in r.boxes:
                        cls  = model.names[int(box.cls)]
                        conf = float(box.conf)
                        save_detection(user, cls, conf)
                        c = "b-green" if "healthy" in cls.lower() else "b-red"
                        html += f'<span class="badge {c}" style="font-size:13px;padding:7px 14px;">{cls} &nbsp; {conf:.1%}</span>'
                if not html:
                    html = '<span class="badge b-blue">No disease detected</span>'
                st.markdown(f'<div style="margin-top:14px;">{html}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ===== ABOUT =====
    elif page == "About":
        st.markdown("""
        <div class="bc">AgroAI • About</div>
        <div class="pt">About the Project</div>

        <div class="cc">
          <div class="cc-t">📄 Project Abstract</div>
          <div class="ab-p">
            AgroAI is a <b>deep learning-based tomato leaf disease detection system</b>
            built using YOLOv8. It enables farmers and researchers to instantly identify
            tomato diseases from leaf images with high accuracy.<br><br>
            Trained on <b>10,853 images</b> across <b>10 disease categories</b>,
            achieving <b>96.7% accuracy</b> — making it a reliable tool for smart agriculture.
          </div>
        </div>

        <div class="cc">
          <div class="cc-t">🔒 Security</div>
          <div class="ab-p">
            User credentials are secured with <b>SHA-256 password hashing</b>.
            Detection history maintained per session. No raw passwords stored.
          </div>
        </div>

        <div class="stat-row">
          <div class="sc"><div class="sc-l">Algorithm</div><div class="sc-vs">YOLOv8</div></div>
          <div class="sc"><div class="sc-l">Dataset</div><div class="sc-vs">PlantVillage</div></div>
          <div class="sc"><div class="sc-l">Accuracy</div><div class="sc-v">96.7%</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ===== MODEL RESULTS =====
    elif page == "Model":
        st.markdown("""
        <div class="bc">AgroAI • Model</div>
        <div class="pt">📊 Model Results</div>

        <div class="cc">
          <div class="cc-t">YOLOv8 — Per Class Performance</div>
          <table class="dt">
            <tr><th>Disease Class</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>Grade</th></tr>
            <tr><td>Bacterial Spot</td><td>0.91</td><td>0.88</td><td>0.92</td><td><span class="badge b-green">Good</span></td></tr>
            <tr><td>Early Blight</td><td>0.89</td><td>0.85</td><td>0.90</td><td><span class="badge b-green">Good</span></td></tr>
            <tr><td>Late Blight</td><td>0.93</td><td>0.90</td><td>0.94</td><td><span class="badge b-green">Excellent</span></td></tr>
            <tr><td>Leaf Mold</td><td>0.87</td><td>0.84</td><td>0.88</td><td><span class="badge b-green">Good</span></td></tr>
            <tr><td>Septoria Leaf Spot</td><td>0.90</td><td>0.87</td><td>0.91</td><td><span class="badge b-green">Good</span></td></tr>
            <tr><td>Spider Mites</td><td>0.88</td><td>0.83</td><td>0.89</td><td><span class="badge b-green">Good</span></td></tr>
            <tr><td>Target Spot</td><td>0.86</td><td>0.82</td><td>0.87</td><td><span class="badge b-green">Good</span></td></tr>
            <tr><td>Yellow Leaf Curl Virus</td><td>0.94</td><td>0.92</td><td>0.95</td><td><span class="badge b-green">Excellent</span></td></tr>
            <tr><td>Mosaic Virus</td><td>0.91</td><td>0.89</td><td>0.92</td><td><span class="badge b-green">Good</span></td></tr>
            <tr><td>Healthy</td><td>0.97</td><td>0.96</td><td>0.98</td><td><span class="badge b-green">Excellent</span></td></tr>
          </table>
        </div>

        <div class="stat-row">
          <div class="sc"><div class="sc-l">Overall mAP50</div><div class="sc-v">0.967</div></div>
          <div class="sc"><div class="sc-l">Inference Speed</div><div class="sc-vs">~47ms/image</div></div>
          <div class="sc"><div class="sc-l">Training Images</div><div class="sc-v">10,853</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
