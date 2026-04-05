import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="LeafScan AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "home"
if "history" not in st.session_state:
    st.session_state.history = []

def nav(p):
    st.session_state.page = p
    st.rerun()

PAGE = st.session_state.page

# ══════════════════════════════════════════════════════
#  DISEASE METADATA
# ══════════════════════════════════════════════════════
DISEASES = {
    "Tomato_Bacterial_Spot":          {"emoji": "🦠", "severity": "High",   "color": "#ef4444"},
    "Tomato_Early_Blight":            {"emoji": "🍂", "severity": "Medium", "color": "#f97316"},
    "Tomato_Late_Blight":             {"emoji": "💧", "severity": "High",   "color": "#ef4444"},
    "Tomato_Leaf_Mold":               {"emoji": "🌫️", "severity": "Medium", "color": "#f97316"},
    "Tomato_Septoria_Leaf_Spot":      {"emoji": "🔵", "severity": "Medium", "color": "#f97316"},
    "Tomato_Spider_Mites":            {"emoji": "🕷️", "severity": "Low",    "color": "#22c55e"},
    "Tomato_Target_Spot":             {"emoji": "🎯", "severity": "Medium", "color": "#f97316"},
    "Tomato_Yellow_Leaf_Curl_Virus":  {"emoji": "🟡", "severity": "High",   "color": "#ef4444"},
    "Tomato_mosaic_virus":            {"emoji": "🧬", "severity": "High",   "color": "#ef4444"},
    "Tomato_healthy":                 {"emoji": "✅", "severity": "None",   "color": "#22c55e"},
}

TREATMENT = {
    "Tomato_Bacterial_Spot":         "Apply copper-based bactericides. Remove infected leaves. Avoid overhead watering.",
    "Tomato_Early_Blight":           "Use chlorothalonil or mancozeb fungicides. Rotate crops annually.",
    "Tomato_Late_Blight":            "Apply metalaxyl fungicide immediately. Remove and destroy all infected plant matter.",
    "Tomato_Leaf_Mold":              "Improve air circulation. Apply fungicide. Reduce humidity in greenhouse settings.",
    "Tomato_Septoria_Leaf_Spot":     "Remove lower infected leaves. Apply fungicide. Avoid working in wet foliage.",
    "Tomato_Spider_Mites":           "Apply miticide or neem oil. Increase humidity. Introduce predatory mites.",
    "Tomato_Target_Spot":            "Apply azoxystrobin fungicide. Improve field drainage. Remove plant debris.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly vectors with insecticide. Remove infected plants. Use resistant varieties.",
    "Tomato_mosaic_virus":           "No cure exists. Remove infected plants. Sanitize tools. Use virus-free seeds.",
    "Tomato_healthy":                "Your plant is healthy! Maintain regular watering, fertilisation, and pest monitoring.",
}

CLASS_PERF = [
    ("Tomato Bacterial Spot",        94.1, 95.3),
    ("Tomato Early Blight",          94.4, 95.7),
    ("Tomato Late Blight",           93.2, 94.1),
    ("Tomato Leaf Mold",             91.5, 92.8),
    ("Tomato Septoria Leaf Spot",    92.7, 93.4),
    ("Tomato Spider Mites",          90.3, 91.6),
    ("Tomato Target Spot",           89.8, 90.5),
    ("Tomato Yellow Leaf Curl Virus",51.2, 96.5),
    ("Tomato Mosaic Virus",          86.2, 96.6),
    ("Tomato Healthy",               99.7, 99.5),
]

# ══════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Epilogue:wght@300;400;500;600&display=swap');

:root {
    --bg:     #050a0e;
    --sf:     #0c1520;
    --sf2:    #111d2b;
    --bdr:    rgba(255,255,255,0.07);
    --acc:    #00e5a0;
    --acc2:   #0ea5e9;
    --txt:    rgba(255,255,255,0.88);
    --muted:  rgba(255,255,255,0.35);
    --danger: #ef4444;
    --warn:   #f97316;
    --ok:     #22c55e;
}

*, *::before, *::after { box-sizing: border-box; font-family: 'Epilogue', sans-serif !important; }

.stApp { background: var(--bg) !important; }

/* hide all default Streamlit chrome */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stDecoration"],
[data-testid="stToolbar"],
#MainMenu, footer, header { display: none !important; visibility: hidden !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }

/* ── TOPBAR ── pure HTML element, no Streamlit buttons inside */
.topbar {
    position: fixed; top: 0; left: 0; right: 0; height: 62px; z-index: 9999;
    background: rgba(5,10,14,0.97);
    border-bottom: 1px solid var(--bdr);
    display: flex; align-items: center; padding: 0 32px; gap: 8px;
    backdrop-filter: blur(24px);
}
.topbar-logo {
    font-family: 'Syne', sans-serif !important;
    font-size: 19px; font-weight: 800; color: white;
    display: flex; align-items: center; gap: 10px;
    white-space: nowrap; flex-shrink: 0; margin-right: 24px;
}
.logo-box {
    width: 34px; height: 34px; flex-shrink: 0;
    background: linear-gradient(135deg, #00e5a0, #0ea5e9);
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 17px;
}
.topbar-logo em { color: var(--acc); font-style: normal; }

.topbar-nav { display: flex; align-items: center; gap: 2px; }
.nav-btn {
    color: var(--muted); font-size: 13px; font-weight: 500;
    padding: 7px 14px; border-radius: 8px; cursor: pointer;
    border: none; background: transparent; white-space: nowrap;
    transition: background 0.15s, color 0.15s; font-family: 'Epilogue', sans-serif;
    line-height: 1;
}
.nav-btn:hover  { background: rgba(255,255,255,0.07); color: white; }
.nav-btn.active { background: rgba(0,229,160,0.1); color: var(--acc); }

.topbar-spacer { flex: 1; min-width: 8px; }
.topbar-badge {
    background: rgba(0,229,160,0.1); color: var(--acc);
    border: 1px solid rgba(0,229,160,0.25); font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; letter-spacing: 0.8px;
    text-transform: uppercase; white-space: nowrap; flex-shrink: 0;
}

/* push all content below the fixed bar */
.main-wrap { margin-top: 62px; }

/* ── hide the 4 hidden nav buttons from view ── */
.hidden-nav-row {
    position: fixed !important; top: -999px !important; left: -999px !important;
    width: 0 !important; height: 0 !important; overflow: hidden !important;
    opacity: 0 !important; pointer-events: none !important;
}

/* ── PAGE CTAs ── */
.stButton > button {
    background: var(--acc) !important; color: #050a0e !important;
    border: none !important; border-radius: 10px !important; padding: 13px !important;
    font-size: 14px !important; font-weight: 700 !important; width: 100% !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #00fbb0 !important;
    box-shadow: 0 0 24px rgba(0,229,160,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── INPUTS ── */
.stTextInput > label {
    color: var(--muted) !important; font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
}
.stTextInput input {
    background: var(--sf2) !important; border: 1px solid var(--bdr) !important;
    border-radius: 10px !important; color: white !important;
    font-size: 15px !important; padding: 12px 16px !important;
}
.stTextInput input:focus {
    border-color: var(--acc) !important;
    box-shadow: 0 0 0 3px rgba(0,229,160,0.1) !important;
}

[data-testid="stFileUploader"] {
    background: rgba(0,229,160,0.02) !important;
    border: 2px dashed rgba(0,229,160,0.2) !important; border-radius: 14px !important;
}
[data-testid="stImage"] img { border-radius: 12px !important; }

[data-testid="stMetric"] {
    background: var(--sf) !important; border: 1px solid var(--bdr) !important;
    border-radius: 14px !important; padding: 20px !important;
}
[data-testid="stMetricValue"] {
    color: var(--acc) !important; font-family: 'Syne', sans-serif !important;
    font-size: 28px !important; font-weight: 700 !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12px !important; }

h1,h2,h3 { color: white !important; font-family: 'Syne', sans-serif !important; }
p { color: var(--txt) !important; }

/* ── HOME ── */
.hero { text-align: center; padding: 88px 24px 60px; max-width: 760px; margin: 0 auto; }
.hero-kicker {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,229,160,0.08); color: var(--acc);
    border: 1px solid rgba(0,229,160,0.2); padding: 6px 20px;
    border-radius: 20px; font-size: 11px; font-weight: 600;
    letter-spacing: 1.8px; text-transform: uppercase; margin-bottom: 32px;
}
.hero-title {
    font-family: 'Syne', sans-serif !important; color: white;
    font-size: 60px; font-weight: 800; line-height: 1.04;
    letter-spacing: -3px; margin-bottom: 22px;
}
.hero-title .hl { color: var(--acc); }
.hero-sub { color: var(--muted); font-size: 18px; line-height: 1.8; max-width: 520px; margin: 0 auto 36px; }

.stats-band {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 12px; padding: 0 40px; margin: 12px 0 64px;
}
.stat-tile {
    background: var(--sf); border: 1px solid var(--bdr);
    border-radius: 16px; padding: 28px 20px; text-align: center;
    position: relative; overflow: hidden;
}
.stat-tile::before {
    content:''; position: absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--acc), var(--acc2));
}
.stat-n {
    font-family: 'Syne', sans-serif !important; color: var(--acc);
    font-size: 36px; font-weight: 800; margin-bottom: 4px; letter-spacing: -1.5px;
}
.stat-l { color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.3px; }

.how-section { padding: 0 40px 72px; }
.sec-eye { color: var(--acc); font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px; }
.sec-title {
    font-family: 'Syne', sans-serif !important; color: white !important;
    font-size: 30px; font-weight: 700; letter-spacing: -0.8px; margin-bottom: 28px;
}
.steps-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }
.step-card {
    background: var(--sf); border: 1px solid var(--bdr);
    border-radius: 16px; padding: 28px; transition: border-color 0.2s;
}
.step-card:hover { border-color: rgba(0,229,160,0.3); }
.step-num {
    width: 40px; height: 40px;
    background: linear-gradient(135deg,rgba(0,229,160,0.12),rgba(14,165,233,0.12));
    border: 1px solid rgba(0,229,160,0.18); color: var(--acc);
    font-family: 'Syne', sans-serif !important; font-size: 16px; font-weight: 800;
    border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;
}
.step-title { color: white; font-size: 16px; font-weight: 600; margin-bottom: 8px; }
.step-desc { color: var(--muted); font-size: 14px; line-height: 1.7; }

.d-section { padding: 0 40px 72px; }
.d-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 10px; }
.d-row {
    background: var(--sf); border: 1px solid var(--bdr); border-radius: 10px;
    padding: 14px 18px; display: flex; align-items: center; gap: 12px; transition: all 0.15s;
}
.d-row:hover { border-color: rgba(0,229,160,0.2); background: var(--sf2); }
.d-icon { font-size: 20px; flex-shrink: 0; }
.d-name { color: rgba(255,255,255,0.65); font-size: 14px; font-weight: 500; flex: 1; }
.sev { font-size: 10px; font-weight: 700; padding: 2px 9px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.sev-High   { background: rgba(239,68,68,0.15);  color: #ef4444; }
.sev-Medium { background: rgba(249,115,22,0.15); color: #f97316; }
.sev-Low    { background: rgba(34,197,94,0.15);  color: #22c55e; }
.sev-None   { background: rgba(34,197,94,0.15);  color: #22c55e; }

/* ── DETECT ── */
.detect-wrap { padding: 40px; }
.pg-eye { color: var(--acc); font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.pg-title {
    font-family: 'Syne', sans-serif !important; color: white;
    font-size: 30px; font-weight: 700; letter-spacing: -0.8px; margin-bottom: 28px;
}
.panel { background: var(--sf); border: 1px solid var(--bdr); border-radius: 18px; padding: 28px; }
.panel-lbl { color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 18px; }
.empty-slot { text-align: center; padding: 64px 20px; color: rgba(255,255,255,0.15); font-size: 14px; }
.empty-ico { font-size: 40px; margin-bottom: 14px; opacity: 0.35; }
.res-card { border-radius: 14px; padding: 20px 22px; margin: 12px 0; border-left: 3px solid; }
.res-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.res-name { color: white; font-size: 16px; font-weight: 600; }
.conf-pill { font-size: 12px; font-weight: 700; padding: 3px 12px; border-radius: 20px; background: rgba(0,229,160,0.1); color: var(--acc); }
.res-sev { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.res-treat { color: var(--muted); font-size: 13px; line-height: 1.7; margin-top: 10px; }

/* ── ANALYTICS ── */
.an-wrap { padding: 40px; }
.an-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 36px; }
.an-card { background: var(--sf); border: 1px solid var(--bdr); border-radius: 16px; padding: 24px; text-align: center; }
.an-num { font-family: 'Syne', sans-serif !important; color: var(--acc); font-size: 34px; font-weight: 800; letter-spacing: -1.5px; margin-bottom: 4px; }
.an-lbl { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }

.pr-label-row { display: flex; justify-content: space-between; margin-bottom: 5px; }
.pr-name { color: rgba(255,255,255,0.6); font-size: 13px; }
.pr-vals { color: var(--muted); font-size: 12px; }
.pr-bar-bg { background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; margin-bottom: 13px; overflow: hidden; }
.pr-bar-fill { height: 6px; border-radius: 4px; background: linear-gradient(90deg, var(--acc), var(--acc2)); }

.hist-row {
    background: var(--sf); border: 1px solid var(--bdr); border-radius: 10px;
    padding: 13px 18px; margin-bottom: 8px;
    display: flex; align-items: center; justify-content: space-between;
}
.hist-left { color: rgba(255,255,255,0.65); font-size: 14px; font-weight: 500; display: flex; align-items: center; gap: 10px; }
.hist-right { display: flex; align-items: center; gap: 12px; }
.hist-conf { color: var(--acc); font-size: 13px; font-weight: 700; }
.hist-time { color: var(--muted); font-size: 12px; }

/* ── ABOUT ── */
.about-wrap { padding: 40px; max-width: 900px; }
.ab-card { background: var(--sf); border: 1px solid var(--bdr); border-radius: 16px; padding: 30px 34px; margin-bottom: 16px; }
.ab-card h3 {
    color: white !important; font-family: 'Syne', sans-serif !important;
    font-size: 17px !important; font-weight: 700 !important; margin-bottom: 14px !important;
}
.ab-card p { color: var(--muted) !important; font-size: 14px !important; line-height: 1.85 !important; }
.chip {
    display: inline-block; background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.18); color: var(--acc);
    font-size: 12px; font-weight: 600; padding: 4px 13px;
    border-radius: 6px; margin: 3px 3px 0 0;
}

.footer {
    text-align: center; padding: 28px 40px;
    color: rgba(255,255,255,0.1); font-size: 12px;
    border-top: 1px solid rgba(255,255,255,0.04); margin-top: 80px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  TOPBAR  — pure HTML nav buttons, no Streamlit columns
# ══════════════════════════════════════════════════════
def _a(p): return "active" if PAGE == p else ""

st.markdown(f"""
<div class="topbar">
  <div class="topbar-logo">
    <div class="logo-box">🌱</div>
    Leaf<em>Scan</em>&nbsp;AI
  </div>
  <div class="topbar-nav">
    <button class="nav-btn {_a('home')}"      onclick="window._nav('home')">🏠 Home</button>
    <button class="nav-btn {_a('detect')}"    onclick="window._nav('detect')">🔬 Detect</button>
    <button class="nav-btn {_a('analytics')}" onclick="window._nav('analytics')">📊 Analytics</button>
    <button class="nav-btn {_a('about')}"     onclick="window._nav('about')">ℹ️ About</button>
  </div>
  <div class="topbar-spacer"></div>
  <span class="topbar-badge">YOLOv8 · 96.7% mAP</span>
</div>
<div class="main-wrap"></div>

<script>
window._nav = function(page) {{
  // find and click the matching hidden Streamlit button
  const btns = window.parent.document.querySelectorAll('button[data-testid="baseButton-secondary"]');
  for (const b of btns) {{
    if (b.innerText.trim() === page) {{ b.click(); return; }}
  }}
  // fallback: look inside stButton containers
  const all = window.parent.document.querySelectorAll('.stButton button');
  for (const b of all) {{
    if (b.innerText.trim() === page) {{ b.click(); return; }}
  }}
}};
</script>
""", unsafe_allow_html=True)

# ── Hidden nav triggers (off-screen, invisible) ──────
# Wrapped in a container we can hide with CSS
st.markdown('<div class="hidden-nav-row" id="hidden-nav">', unsafe_allow_html=True)
_c = st.columns(4)
with _c[0]:
    if st.button("home",      key="_nh"):  nav("home")
with _c[1]:
    if st.button("detect",    key="_nd"):  nav("detect")
with _c[2]:
    if st.button("analytics", key="_na"):  nav("analytics")
with _c[3]:
    if st.button("about",     key="_nb"):  nav("about")
st.markdown('</div>', unsafe_allow_html=True)

# Move that column row off-screen via nth-child targeting
st.markdown("""
<style>
/* The first stHorizontalBlock after our topbar is the hidden nav row */
section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:first-of-type {
    position: fixed !important;
    top: 14px !important;
    left: 0 !important; right: 0 !important;
    z-index: 10001 !important;
    width: 0 !important; height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════
if PAGE == "home":
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">✦ Powered by YOLOv8 Object Detection</div>
      <div class="hero-title">Detect Tomato<br><span class="hl">Diseases</span> Instantly</div>
      <div class="hero-sub">
        Upload a leaf photo and get an AI-powered diagnosis in under 4 ms —
        with treatment recommendations and severity ratings.
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, b1, b2, _ = st.columns([2, 1, 1, 2])
    with b1:
        if st.button("🔬  Start Detection", key="cta1"): nav("detect")
    with b2:
        if st.button("📊  View Analytics",  key="cta2"): nav("analytics")

    st.markdown("""
    <div class="stats-band">
      <div class="stat-tile"><div class="stat-n">96.7%</div><div class="stat-l">mAP50 Accuracy</div></div>
      <div class="stat-tile"><div class="stat-n">10</div><div class="stat-l">Disease Classes</div></div>
      <div class="stat-tile"><div class="stat-n">3.6ms</div><div class="stat-l">Inference Speed</div></div>
      <div class="stat-tile"><div class="stat-n">10.8K</div><div class="stat-l">Training Images</div></div>
    </div>
    <div class="how-section">
      <div class="sec-eye">How it works</div>
      <div class="sec-title">Three steps to a diagnosis</div>
      <div class="steps-row">
        <div class="step-card"><div class="step-num">01</div><div class="step-title">Upload a Leaf Photo</div><div class="step-desc">Take a clear photo of the tomato leaf and upload it in JPG or PNG format.</div></div>
        <div class="step-card"><div class="step-num">02</div><div class="step-title">AI Analyses the Image</div><div class="step-desc">YOLOv8 scans for visual disease markers and bounding-boxes detections in real time.</div></div>
        <div class="step-card"><div class="step-num">03</div><div class="step-title">Get Treatment Advice</div><div class="step-desc">Receive severity rating, confidence score, and specific treatment recommendations.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    rows_html = "".join([
        f'<div class="d-row"><span class="d-icon">{m["emoji"]}</span>'
        f'<span class="d-name">{n.replace("_"," ")}</span>'
        f'<span class="sev sev-{m["severity"]}">{m["severity"]}</span></div>'
        for n, m in DISEASES.items()
    ])
    st.markdown(f"""
    <div class="d-section">
      <div class="sec-eye">Disease Reference</div>
      <div class="sec-title">10 Detectable Conditions</div>
      <div class="d-grid">{rows_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  DETECT
# ══════════════════════════════════════════════════════
elif PAGE == "detect":
    st.markdown("""
    <div class="detect-wrap">
    <div class="pg-eye">AI · Computer Vision · Real-time</div>
    <div class="pg-title">🔬 Disease Detection Engine</div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="panel"><div class="panel-lbl">📷 Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed", key="uploader")
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
        else:
            st.markdown('<div class="empty-slot"><div class="empty-ico">🍃</div><div>Drag & drop or click to upload<br>a tomato leaf image (JPG / PNG)</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-lbl">🤖 Detection Results</div>', unsafe_allow_html=True)
        if uploaded:
            with st.spinner("Running YOLOv8 inference..."):
                try:
                    model = YOLO("best.pt")
                    img_np = np.array(img)
                    results = model.predict(img_np, conf=0.25)
                    annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                    st.image(annotated, use_container_width=True)

                    found = False
                    for r in results:
                        for box in r.boxes:
                            found = True
                            cls_name = model.names[int(box.cls)]
                            conf     = float(box.conf)
                            meta     = DISEASES.get(cls_name, {"emoji":"🌿","severity":"Unknown","color":"#888"})
                            treat    = TREATMENT.get(cls_name, "Consult an agricultural expert.")
                            col      = meta["color"]
                            st.session_state.history.append({
                                "label": cls_name, "conf": conf,
                                "time": datetime.now().strftime("%H:%M:%S"),
                            })
                            st.markdown(f"""
                            <div class="res-card" style="background:rgba(255,255,255,0.03);border-left-color:{col};">
                              <div class="res-sev" style="color:{col};">{meta['emoji']} Severity: {meta['severity']}</div>
                              <div class="res-top">
                                <span class="res-name">{cls_name.replace('_',' ')}</span>
                                <span class="conf-pill">{conf:.1%}</span>
                              </div>
                              <div class="res-treat">💊 <b>Treatment:</b> {treat}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    if not found:
                        st.success("✅ No disease detected — this leaf looks healthy!")
                        st.session_state.history.append({
                            "label":"Tomato_healthy","conf":0.99,
                            "time":datetime.now().strftime("%H:%M:%S"),
                        })
                except Exception as e:
                    st.error(f"Model error: {e}\n\nMake sure best.pt is in the same folder.")
        else:
            st.markdown('<div class="empty-slot"><div class="empty-ico">🤖</div><div>Results appear here<br>after you upload an image</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  ANALYTICS
# ══════════════════════════════════════════════════════
elif PAGE == "analytics":
    st.markdown("""
    <div class="an-wrap">
    <div class="pg-eye">Performance · Metrics · History</div>
    <div class="pg-title">📊 Model Analytics</div>
    <div class="an-grid">
      <div class="an-card"><div class="an-num">96.7%</div><div class="an-lbl">mAP50</div></div>
      <div class="an-card"><div class="an-num">94.2%</div><div class="an-lbl">Avg Precision</div></div>
      <div class="an-card"><div class="an-num">93.8%</div><div class="an-lbl">Avg Recall</div></div>
      <div class="an-card"><div class="an-num">3.6ms</div><div class="an-lbl">Inference Speed</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-eye" style="margin-bottom:10px;">Per-Class Performance</div>', unsafe_allow_html=True)
    for name, prec, rec in CLASS_PERF:
        avg = (prec + rec) / 2
        st.markdown(f"""
        <div class="pr-label-row">
          <span class="pr-name">{name}</span>
          <span class="pr-vals">P: {prec}% &nbsp;·&nbsp; R: {rec}%</span>
        </div>
        <div class="pr-bar-bg"><div class="pr-bar-fill" style="width:{avg}%"></div></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-eye" style="margin:28px 0 10px;">Detection History (This Session)</div>', unsafe_allow_html=True)
    history = st.session_state.history

    if not history:
        st.info("No detections yet. Go to Detect and upload a leaf image!")
    else:
        diseased = [h for h in history if h["label"] != "Tomato_healthy"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Scans", len(history))
        c2.metric("Diseased",    len(diseased))
        c3.metric("Healthy",     len(history) - len(diseased))
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        for h in reversed(history[-15:]):
            meta = DISEASES.get(h["label"], {"emoji":"🌿","severity":"Unknown"})
            sev  = meta["severity"]
            sc   = f"sev-{sev}" if sev in ("High","Medium","Low","None") else "sev-None"
            st.markdown(f"""
            <div class="hist-row">
              <div class="hist-left">{meta['emoji']} {h['label'].replace('_',' ')}
                <span class="sev {sc}">{sev}</span>
              </div>
              <div class="hist-right">
                <span class="hist-conf">{h['conf']:.1%}</span>
                <span class="hist-time">🕐 {h['time']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️  Clear History", key="clr"):
            st.session_state.history = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  ABOUT
# ══════════════════════════════════════════════════════
elif PAGE == "about":
    st.markdown("""
    <div class="about-wrap">
    <div class="pg-eye">Project · Methodology · Stack</div>
    <div class="pg-title">ℹ️ About LeafScan AI</div>

    <div class="ab-card">
      <h3>🌱 Project Overview</h3>
      <p>LeafScan AI is a real-time tomato leaf disease detection system built using the YOLOv8
      object detection architecture. It identifies 10 disease classes from a single leaf photograph,
      providing instant severity ratings and actionable treatment recommendations for farmers and agronomists.</p>
    </div>

    <div class="ab-card">
      <h3>🧠 Model &amp; Training</h3>
      <p>The YOLOv8 model was fine-tuned on 10,853 annotated tomato leaf images from the PlantVillage
      dataset across 5 training epochs. It achieves a mean average precision (mAP50) of 96.7% with an
      inference speed of just 3.6 ms per image — enabling true real-time field diagnostics.</p>
    </div>

    <div class="ab-card">
      <h3>⚙️ Technology Stack</h3>
      <p>
        <span class="chip">YOLOv8</span>
        <span class="chip">Python 3.11</span>
        <span class="chip">PyTorch</span>
        <span class="chip">Streamlit</span>
        <span class="chip">OpenCV</span>
        <span class="chip">Pillow</span>
        <span class="chip">NumPy</span>
        <span class="chip">Ultralytics</span>
      </p>
    </div>

    <div class="ab-card">
      <h3>📋 How to Run Locally</h3>
      <p>
        1. Place your trained <code>best.pt</code> model file in the same directory as this script.<br>
        2. Install: <code>pip install streamlit ultralytics opencv-python pillow</code><br>
        3. Run: <code>streamlit run app.py</code><br>
        4. Open <code>localhost:8501</code> and start scanning leaves!
      </p>
    </div>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
  🌱 LeafScan AI — Tomato Disease Detection &nbsp;·&nbsp; Powered by YOLOv8 &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
