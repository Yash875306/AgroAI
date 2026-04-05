import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgroAI — Tomato Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
_defaults = {
    "page": "home",
    "history": [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def nav(p):
    st.session_state.page = p
    st.rerun()

PAGE = st.session_state.page

# ─────────────────────────────────────────────
#  IN-MEMORY HISTORY
# ─────────────────────────────────────────────
def save_detection(disease, confidence, severity):
    st.session_state.history.append({
        "disease":    disease,
        "confidence": confidence,
        "severity":   severity,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

def get_history():
    return list(reversed(st.session_state.history))

def clear_history():
    st.session_state.history = []

# ─────────────────────────────────────────────
#  DISEASE DATA
# ─────────────────────────────────────────────
DISEASES = {
    "Tomato_Bacterial_spot": {
        "label": "Bacterial Spot", "severity": "High", "icon": "🔴",
        "symptoms":   "Small dark water-soaked lesions on leaves and fruit surfaces.",
        "treatment":  "Apply copper-based bactericides. Remove infected debris. Avoid overhead irrigation.",
        "prevention": "Use certified disease-free seeds. Practice 2-year crop rotation.",
    },
    "Tomato_Early_blight": {
        "label": "Early Blight", "severity": "Medium", "icon": "🟠",
        "symptoms":   "Concentric dark rings forming a target pattern on older leaves.",
        "treatment":  "Apply chlorothalonil or mancozeb fungicide every 7–10 days.",
        "prevention": "Rotate crops annually. Remove lower foliage. Mulch around base.",
    },
    "Tomato_Late_blight": {
        "label": "Late Blight", "severity": "Critical", "icon": "🔴",
        "symptoms":   "Large irregular water-soaked grey-green lesions; white mould on underside.",
        "treatment":  "Apply metalaxyl or cymoxanil fungicide immediately. Destroy infected plants.",
        "prevention": "Avoid overhead watering. Plant resistant varieties. Monitor humidity.",
    },
    "Tomato_Leaf_Mold": {
        "label": "Leaf Mold", "severity": "Medium", "icon": "🟠",
        "symptoms":   "Yellow patches on upper leaf surface; olive-green mould on underside.",
        "treatment":  "Apply mancozeb or chlorothalonil. Improve greenhouse ventilation.",
        "prevention": "Reduce relative humidity below 85%. Space plants adequately.",
    },
    "Tomato_Septoria_leaf_spot": {
        "label": "Septoria Leaf Spot", "severity": "Medium", "icon": "🟠",
        "symptoms":   "Small circular spots with dark borders and pale grey centres.",
        "treatment":  "Apply copper fungicide. Remove heavily infected leaves promptly.",
        "prevention": "Mulch soil. Avoid wetting foliage during irrigation.",
    },
    "Tomato_Spider_mites Two-spotted_spider_mite": {
        "label": "Spider Mites", "severity": "Low", "icon": "🟡",
        "symptoms":   "Fine yellow stippling on leaves; fine webbing on leaf undersides.",
        "treatment":  "Apply miticide or neem oil. Increase ambient humidity.",
        "prevention": "Regular scouting. Introduce predatory mites as biocontrol.",
    },
    "Tomato__Target_Spot": {
        "label": "Target Spot", "severity": "Medium", "icon": "🟠",
        "symptoms":   "Bulls-eye concentric ring lesions on leaves and stems.",
        "treatment":  "Apply azoxystrobin or fluxapyroxad. Improve field drainage.",
        "prevention": "Remove plant debris after harvest. Avoid dense canopy.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "label": "Yellow Leaf Curl Virus", "severity": "Critical", "icon": "🔴",
        "symptoms":   "Upward leaf curling, yellowing margins, stunted plant growth.",
        "treatment":  "No chemical cure. Remove and destroy infected plants immediately.",
        "prevention": "Control whitefly populations. Use insect-proof nets and resistant varieties.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "label": "Mosaic Virus", "severity": "High", "icon": "🔴",
        "symptoms":   "Mosaic light-dark green patterns on leaves; distortion and stunting.",
        "treatment":  "No cure. Remove infected plants. Disinfect all tools with bleach solution.",
        "prevention": "Use virus-free certified seeds. Wash hands before handling plants.",
    },
    "Tomato_healthy": {
        "label": "Healthy", "severity": "None", "icon": "🟢",
        "symptoms":   "No disease symptoms detected. Plant appears healthy.",
        "treatment":  "No treatment required.",
        "prevention": "Continue regular monitoring, balanced fertilisation and irrigation.",
    },
}

CLASS_PERF = [
    ("Bacterial Spot",         94.1, 95.3),
    ("Early Blight",           94.4, 95.7),
    ("Late Blight",            93.2, 94.1),
    ("Leaf Mold",              91.5, 92.8),
    ("Septoria Leaf Spot",     92.7, 93.4),
    ("Spider Mites",           90.3, 91.6),
    ("Target Spot",            89.8, 90.5),
    ("Yellow Leaf Curl Virus", 96.5, 51.2),
    ("Mosaic Virus",           86.2, 96.6),
    ("Healthy",                99.7, 99.5),
]

SEV_STYLE = {
    "None":     ("sev-none",     "#166534", "#dcfce7", "#bbf7d0"),
    "Low":      ("sev-low",      "#713f12", "#fef9c3", "#fde68a"),
    "Medium":   ("sev-medium",   "#7c2d12", "#fff7ed", "#fed7aa"),
    "High":     ("sev-high",     "#991b1b", "#fee2e2", "#fecaca"),
    "Critical": ("sev-critical", "#831843", "#fdf2f8", "#f9a8d4"),
}

SEV_BAR = {
    "None": "#16a34a", "Low": "#ca8a04",
    "Medium": "#ea580c", "High": "#dc2626", "Critical": "#9d174d",
}

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import os
    if not os.path.exists("best.pt"):
        return None, "best.pt not found."
    try:
        return YOLO("best.pt"), None
    except Exception as e:
        return None, str(e)

def run_inference(model, pil_image):
    arr     = np.array(pil_image)
    results = model.predict(arr, conf=0.25, verbose=False)
    r       = results[0]
    if r.probs is not None:
        probs = r.probs.data.cpu().numpy()
        idx   = int(np.argmax(probs))
        return [(model.names[idx], float(probs[idx]))], pil_image
    detections, annotated = [], pil_image
    if r.boxes and len(r.boxes):
        annotated = Image.fromarray(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
        for box in r.boxes:
            detections.append((model.names[int(box.cls.item())], float(box.conf.item())))
    return detections or [("Tomato_healthy", 1.0)], annotated

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --green-900: #14532d;
    --green-800: #166534;
    --green-700: #15803d;
    --green-600: #16a34a;
    --green-500: #22c55e;
    --green-400: #4ade80;
    --green-200: #bbf7d0;
    --green-100: #dcfce7;
    --green-50:  #f0fdf4;
    --slate-900: #0f172a;
    --slate-800: #1e293b;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748b;
    --slate-400: #94a3b8;
    --slate-300: #cbd5e1;
    --slate-200: #e2e8f0;
    --slate-100: #f1f5f9;
    --slate-50:  #f8fafc;
    --white:     #ffffff;
    --radius:    10px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow:    0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
    --shadow-lg: 0 10px 15px rgba(0,0,0,0.07), 0 4px 6px rgba(0,0,0,0.05);
}

*, *::before, *::after {
    box-sizing: border-box;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp { background: var(--slate-50) !important; }

[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stDecoration"], [data-testid="stToolbar"],
#MainMenu, footer, header { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
.stMarkdown { margin: 0 !important; }

/* Hide nav buttons */
section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:first-of-type {
    position: fixed !important; top: -9999px !important;
    left: -9999px !important; width: 0 !important;
    height: 0 !important; overflow: hidden !important;
    opacity: 0 !important; pointer-events: none !important;
}

/* ── TOPBAR ── */
.topbar {
    position: fixed; top: 0; left: 0; right: 0;
    height: 64px; z-index: 9999;
    background: var(--white);
    border-bottom: 1px solid var(--slate-200);
    display: flex; align-items: center;
    padding: 0 48px; gap: 0;
    box-shadow: var(--shadow-sm);
}
.tb-logo {
    display: flex; align-items: center; gap: 10px;
    margin-right: 48px; flex-shrink: 0;
}
.tb-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--green-600), var(--green-800));
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 18px; box-shadow: var(--shadow);
}
.tb-logo-name {
    font-size: 19px; font-weight: 800;
    color: var(--slate-900); letter-spacing: -0.5px;
}
.tb-logo-name span { color: var(--green-600); }
.tb-nav { display: flex; align-items: center; gap: 2px; }
.tb-btn {
    font-size: 14px; font-weight: 500;
    color: var(--slate-500);
    padding: 8px 18px; border-radius: 8px;
    cursor: pointer; border: none;
    background: transparent; white-space: nowrap;
    transition: all 0.15s; line-height: 1;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.tb-btn:hover  { background: var(--green-50);  color: var(--green-700); }
.tb-btn.active {
    background: var(--green-100); color: var(--green-800); font-weight: 600;
}
.tb-spacer { flex: 1; }
.tb-badge {
    display: flex; align-items: center; gap: 6px;
    background: var(--green-50);
    border: 1px solid var(--green-200);
    color: var(--green-700); font-size: 12px; font-weight: 600;
    padding: 6px 14px; border-radius: 20px;
}
.tb-dot {
    width: 7px; height: 7px; background: var(--green-500);
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
.main-wrap { margin-top: 64px; }

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, var(--green-900) 0%, var(--green-700) 60%, var(--green-600) 100%);
    padding: 88px 64px 80px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -80px; right: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: ''; position: absolute; bottom: -60px; left: 200px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(74,222,128,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-inner { max-width: 680px; position: relative; z-index: 1; }
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.9);
    font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
    padding: 6px 16px; border-radius: 20px; margin-bottom: 24px;
}
.hero-title {
    font-size: 54px; font-weight: 800;
    line-height: 1.08; letter-spacing: -2px;
    color: white; margin-bottom: 20px;
}
.hero-title em { color: var(--green-400); font-style: normal; }
.hero-sub {
    font-size: 17px; line-height: 1.75;
    color: rgba(255,255,255,0.65);
    max-width: 520px; margin-bottom: 40px;
}
.hero-cta-wrap { display: flex; gap: 12px; align-items: center; }
.hero-cta {
    background: white; color: var(--green-800);
    font-size: 15px; font-weight: 700;
    padding: 14px 32px; border-radius: 10px;
    border: none; cursor: pointer;
    font-family: 'Plus Jakarta Sans', sans-serif;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    transition: all 0.2s;
}
.hero-cta:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,0,0,0.2); }
.hero-secondary {
    background: rgba(255,255,255,0.1);
    color: white; font-size: 14px; font-weight: 600;
    padding: 14px 28px; border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.25); cursor: pointer;
    font-family: 'Plus Jakarta Sans', sans-serif; transition: all 0.2s;
}
.hero-secondary:hover { background: rgba(255,255,255,0.18); }

/* ── STATS ROW ── */
.stats-strip {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0; border-bottom: 1px solid var(--slate-200);
    background: white;
}
.stat-item {
    padding: 28px 36px; text-align: center;
    border-right: 1px solid var(--slate-200);
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-size: 36px; font-weight: 800;
    color: var(--green-700); letter-spacing: -1px;
    margin-bottom: 4px; line-height: 1;
}
.stat-lbl {
    font-size: 12px; font-weight: 600;
    color: var(--slate-400); text-transform: uppercase; letter-spacing: 0.8px;
}

/* ── PAGE WRAPPER ── */
.page     { padding: 56px 64px 100px; max-width: 1200px; margin: 0 auto; }
.page-sm  { padding: 56px 64px 100px; max-width: 900px;  margin: 0 auto; }
.page-md  { padding: 56px 64px 100px; max-width: 1060px; margin: 0 auto; }

/* ── SECTION HEADING ── */
.sec-label {
    font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--green-600); margin-bottom: 8px;
}
.sec-heading {
    font-size: 32px; font-weight: 800; color: var(--slate-900);
    letter-spacing: -0.8px; margin-bottom: 8px; line-height: 1.2;
}
.sec-sub {
    font-size: 16px; color: var(--slate-500); line-height: 1.7;
    margin-bottom: 40px; max-width: 560px;
}

/* ── CARDS ── */
.card {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: 14px; padding: 28px 32px;
    box-shadow: var(--shadow); margin-bottom: 20px;
}
.card-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 20px; padding-bottom: 16px;
    border-bottom: 1px solid var(--slate-100);
}
.card-icon {
    width: 36px; height: 36px;
    background: var(--green-50); border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; flex-shrink: 0;
}
.card-title { font-size: 15px; font-weight: 700; color: var(--slate-900); }
.card-sub   { font-size: 13px; color: var(--slate-400); margin-top: 1px; }

/* ── HOW IT WORKS ── */
.steps-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 48px; }
.step-card {
    background: var(--white); border: 1px solid var(--slate-200);
    border-radius: 14px; padding: 28px 24px;
    box-shadow: var(--shadow); position: relative; overflow: hidden;
}
.step-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, var(--green-500), var(--green-400));
}
.step-num {
    width: 38px; height: 38px;
    background: var(--green-100); color: var(--green-800);
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 14px; font-weight: 800;
    margin-bottom: 16px;
}
.step-title { font-size: 15px; font-weight: 700; color: var(--slate-900); margin-bottom: 8px; }
.step-desc  { font-size: 13px; color: var(--slate-500); line-height: 1.7; }

/* ── DISEASE GRID ── */
.disease-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 8px; }
.disease-row {
    background: var(--white); border: 1px solid var(--slate-200);
    border-radius: 10px; padding: 12px 16px;
    display: flex; align-items: center; gap: 12px;
    box-shadow: var(--shadow-sm); transition: all 0.15s;
}
.disease-row:hover { border-color: var(--green-300); box-shadow: var(--shadow); }
.disease-icon { font-size: 16px; flex-shrink: 0; }
.disease-name { flex: 1; font-size: 13px; font-weight: 500; color: var(--slate-700); }

/* ── SEVERITY BADGES ── */
.sev {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
    padding: 3px 10px; border-radius: 6px; white-space: nowrap;
}
.sev-none     { background: #dcfce7; color: #166534; }
.sev-low      { background: #fef9c3; color: #713f12; }
.sev-medium   { background: #fff7ed; color: #7c2d12; }
.sev-high     { background: #fee2e2; color: #991b1b; }
.sev-critical { background: #fdf2f8; color: #831843; }

/* ── DETECT PAGE ── */
.detect-panel {
    background: var(--white); border: 1px solid var(--slate-200);
    border-radius: 14px; padding: 28px;
    box-shadow: var(--shadow); height: 100%;
}
.panel-title {
    font-size: 13px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--slate-400);
    margin-bottom: 20px; padding-bottom: 14px;
    border-bottom: 1px solid var(--slate-100);
}
.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 60px 20px;
    color: var(--slate-300); text-align: center; gap: 12px;
}
.empty-icon {
    width: 60px; height: 60px; border-radius: 14px;
    background: var(--slate-100); display: flex;
    align-items: center; justify-content: center;
    font-size: 24px; color: var(--slate-300);
}
.empty-text { font-size: 14px; color: var(--slate-400); line-height: 1.6; }

/* ── RESULT CARD ── */
.result-card {
    border: 1px solid var(--slate-200); border-radius: 12px;
    padding: 22px; margin-bottom: 14px;
    border-left: 4px solid var(--green-500);
    background: var(--white); box-shadow: var(--shadow-sm);
}
.result-top { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 12px; }
.result-disease { font-size: 20px; font-weight: 800; color: var(--slate-900); }
.result-conf    { font-size: 13px; color: var(--slate-400); margin-top: 3px; }
.conf-track { background: var(--slate-100); border-radius: 4px; height: 6px; margin-bottom: 18px; overflow: hidden; }
.conf-fill  { height: 6px; border-radius: 4px; }
.info-block { margin-bottom: 14px; }
.info-lbl   { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--slate-400); margin-bottom: 4px; }
.info-val   { font-size: 14px; color: var(--slate-600); line-height: 1.7; }

/* ── HISTORY TABLE ── */
.htable { width: 100%; border-collapse: collapse; }
.htable th {
    font-size: 11px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--slate-400);
    padding: 12px 18px; text-align: left;
    background: var(--slate-50); border-bottom: 2px solid var(--slate-200);
}
.htable td {
    font-size: 14px; color: var(--slate-700);
    padding: 14px 18px; border-bottom: 1px solid var(--slate-100);
    vertical-align: middle;
}
.htable tr:last-child td { border-bottom: none; }
.htable tr:hover td { background: var(--green-50); }

/* ── METRIC CARDS ── */
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 32px; }
.metric-card {
    background: var(--white); border: 1px solid var(--slate-200);
    border-radius: 12px; padding: 22px 20px;
    box-shadow: var(--shadow-sm); text-align: center;
    border-top: 3px solid var(--green-500);
}
.metric-val { font-size: 32px; font-weight: 800; color: var(--green-700); letter-spacing: -1px; margin-bottom: 4px; }
.metric-lbl { font-size: 12px; font-weight: 600; color: var(--slate-400); text-transform: uppercase; letter-spacing: 0.8px; }

/* ── PERF BARS ── */
.perf-item { margin-bottom: 18px; }
.perf-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
.perf-name { font-size: 13px; font-weight: 600; color: var(--slate-700); }
.perf-nums { font-size: 12px; color: var(--slate-400); }
.perf-track { background: var(--slate-100); border-radius: 4px; height: 8px; overflow: hidden; }
.perf-bar   { height: 8px; border-radius: 4px; background: linear-gradient(90deg, var(--green-600), var(--green-400)); }

/* ── ABOUT ── */
.about-feature {
    display: flex; align-items: flex-start; gap: 16px;
    padding: 18px 0; border-bottom: 1px solid var(--slate-100);
}
.about-feature:last-child { border-bottom: none; }
.af-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: var(--green-100); flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
}
.af-title { font-size: 14px; font-weight: 700; color: var(--slate-900); margin-bottom: 4px; }
.af-desc  { font-size: 13px; color: var(--slate-500); line-height: 1.65; }

.spec-row { display: flex; gap: 8px; padding: 10px 0; border-bottom: 1px solid var(--slate-100); }
.spec-row:last-child { border-bottom: none; }
.spec-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: var(--slate-400); min-width: 150px; }
.spec-val   { font-size: 14px; color: var(--slate-700); font-weight: 500; }

.chip { display: inline-block; font-size: 12px; font-weight: 600; color: var(--green-800); background: var(--green-100); border: 1px solid var(--green-200); padding: 4px 12px; border-radius: 6px; margin: 3px; }

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    background: var(--green-600) !important; color: white !important;
    border: none !important; border-radius: var(--radius) !important;
    padding: 12px 32px !important; font-size: 14px !important;
    font-weight: 700 !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.2s !important; letter-spacing: 0.1px !important;
}
.stButton > button:hover {
    background: var(--green-700) !important;
    box-shadow: 0 4px 14px rgba(22,163,74,0.3) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed var(--green-300) !important;
    border-radius: 12px !important;
    background: var(--green-50) !important;
    padding: 8px !important;
}
[data-testid="stImage"] img { border-radius: 10px; }

/* ── FOOTER ── */
.footer {
    background: var(--slate-900); color: var(--slate-400);
    text-align: center; padding: 32px 40px;
    font-size: 13px; margin-top: 80px;
}
.footer span { color: var(--green-400); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TOPBAR
# ─────────────────────────────────────────────
def _a(p): return "active" if PAGE == p else ""

st.markdown(f"""
<div class="topbar">
  <div class="tb-logo">
    <div class="tb-logo-icon">🌿</div>
    <div class="tb-logo-name">Agro<span>AI</span></div>
  </div>
  <nav class="tb-nav">
    <button class="tb-btn {_a('home')}"    onclick="_go('home')">Home</button>
    <button class="tb-btn {_a('detect')}"  onclick="_go('detect')">Detect</button>
    <button class="tb-btn {_a('results')}" onclick="_go('results')">Results</button>
    <button class="tb-btn {_a('about')}"   onclick="_go('about')">About</button>
  </nav>
  <div class="tb-spacer"></div>
  <div class="tb-badge"><div class="tb-dot"></div>YOLOv8 · 96.7% Accuracy</div>
</div>
<div class="main-wrap"></div>
<script>
function _go(p) {{
    window.parent.document.querySelectorAll('.stButton button').forEach(b => {{
        if (b.innerText.trim() === p) b.click();
    }});
}}
</script>
""", unsafe_allow_html=True)

nc = st.columns(4)
with nc[0]:
    if st.button("home",    key="_nh"): nav("home")
with nc[1]:
    if st.button("detect",  key="_nd"): nav("detect")
with nc[2]:
    if st.button("results", key="_nr"): nav("results")
with nc[3]:
    if st.button("about",   key="_na"): nav("about")

# ═══════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════
if PAGE == "home":

    st.markdown("""
    <div class="hero">
      <div class="hero-inner">
        <div class="hero-tag">🌱 Precision Agriculture · AI-Powered · Real-time</div>
        <div class="hero-title">Detect Tomato Diseases<br><em>Instantly</em> with AI</div>
        <div class="hero-sub">
          Upload a single tomato leaf photo and get an instant diagnosis — complete with
          severity rating, treatment protocol, and prevention advice powered by YOLOv8.
        </div>
        <div class="hero-cta-wrap">
          <button class="hero-cta" onclick="_go('detect')">Start Detection →</button>
          <button class="hero-secondary" onclick="_go('about')">Learn More</button>
        </div>
      </div>
    </div>

    <div class="stats-strip">
      <div class="stat-item"><div class="stat-num">96.7%</div><div class="stat-lbl">mAP50 Accuracy</div></div>
      <div class="stat-item"><div class="stat-num">10</div><div class="stat-lbl">Disease Classes</div></div>
      <div class="stat-item"><div class="stat-num">3.6ms</div><div class="stat-lbl">Inference Speed</div></div>
      <div class="stat-item"><div class="stat-num">10,853</div><div class="stat-lbl">Training Images</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page">', unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="sec-label">How It Works</div>
    <div class="sec-heading">Three steps to a diagnosis</div>
    <div class="sec-sub">From image upload to actionable treatment advice in seconds.</div>
    <div class="steps-grid">
      <div class="step-card">
        <div class="step-num">01</div>
        <div class="step-title">Upload a Leaf Photo</div>
        <div class="step-desc">Take a clear, well-lit photo of a tomato leaf and upload it in JPG or PNG format. No special equipment needed.</div>
      </div>
      <div class="step-card">
        <div class="step-num">02</div>
        <div class="step-title">AI Analyses the Image</div>
        <div class="step-desc">YOLOv8 scans for disease markers and returns bounding-box detections with confidence scores in under 4ms.</div>
      </div>
      <div class="step-card">
        <div class="step-num">03</div>
        <div class="step-title">Receive Expert Advice</div>
        <div class="step-desc">View the severity rating, symptoms, evidence-based treatment steps, and prevention recommendations instantly.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Disease reference
    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown("""
        <div class="sec-label">Disease Reference</div>
        <div class="sec-heading">10 Detectable Conditions</div>
        """, unsafe_allow_html=True)
        rows = "".join(
            f'<div class="disease-row">'
            f'<span class="disease-icon">{v["icon"]}</span>'
            f'<span class="disease-name">{v["label"]}</span>'
            f'<span class="sev {SEV_STYLE[v["severity"]][0]}">{v["severity"]}</span>'
            f'</div>'
            for v in DISEASES.values()
        )
        st.markdown(f'<div class="disease-grid">{rows}</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="sec-label">Why AgroAI</div>
        <div class="sec-heading">Built for Farmers</div>
        <div class="card">
          <div class="about-feature">
            <div class="af-icon">⚡</div>
            <div><div class="af-title">Real-time Detection</div>
            <div class="af-desc">Get results in under 4 milliseconds. No waiting, no delays — instant diagnosis at your fingertips.</div></div>
          </div>
          <div class="about-feature">
            <div class="af-icon">🎯</div>
            <div><div class="af-title">96.7% Accuracy</div>
            <div class="af-desc">Trained on 10,853 professionally annotated leaf images across all 10 disease categories.</div></div>
          </div>
          <div class="about-feature">
            <div class="af-icon">💊</div>
            <div><div class="af-title">Treatment Guidance</div>
            <div class="af-desc">Every diagnosis comes with evidence-based treatment protocols and prevention recommendations.</div></div>
          </div>
          <div class="about-feature">
            <div class="af-icon">📊</div>
            <div><div class="af-title">Detection History</div>
            <div class="af-desc">Track all past detections with severity ratings and timestamps in the Results page.</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  DETECT
# ═══════════════════════════════════════════════════════
elif PAGE == "detect":
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Computer Vision Analysis</div>
    <div class="sec-heading">Disease Detection</div>
    <div class="sec-sub">Upload a clear tomato leaf image to receive an instant AI-powered diagnosis.</div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="detect-panel"><div class="panel-title">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a tomato leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="uploader",
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True)
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📷</div>
              <div class="empty-text">Drop a JPG or PNG leaf image here<br>to start your analysis</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="detect-panel"><div class="panel-title">Detection Results</div>', unsafe_allow_html=True)
        if uploaded:
            model, err = load_model()
            if err:
                st.error(f"⚠️ {err}")
            else:
                with st.spinner("Running YOLOv8 inference..."):
                    detections, annotated = run_inference(model, img)

                if annotated is not img:
                    st.image(annotated, use_column_width=True, caption="Annotated Output")

                for cls_name, conf in detections:
                    info     = DISEASES.get(cls_name, DISEASES["Tomato_healthy"])
                    sev      = info["severity"]
                    cls_s, tc, bg, _ = SEV_STYLE[sev]
                    bar_col  = SEV_BAR[sev]
                    conf_pct = int(conf * 100)

                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{bar_col};">
                      <div class="result-top">
                        <div>
                          <div class="result-disease">{info['icon']} {info['label']}</div>
                          <div class="result-conf">Confidence: <strong>{conf_pct}%</strong></div>
                        </div>
                        <span class="sev {cls_s}">{sev}</span>
                      </div>
                      <div class="conf-track">
                        <div class="conf-fill" style="width:{conf_pct}%;background:{bar_col};"></div>
                      </div>
                      <div class="info-block">
                        <div class="info-lbl">🔬 Symptoms</div>
                        <div class="info-val">{info['symptoms']}</div>
                      </div>
                      <div class="info-block">
                        <div class="info-lbl">💊 Treatment</div>
                        <div class="info-val">{info['treatment']}</div>
                      </div>
                      <div class="info-block" style="margin-bottom:0">
                        <div class="info-lbl">🛡️ Prevention</div>
                        <div class="info-val">{info['prevention']}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    save_detection(info["label"], conf, sev)
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📋</div>
              <div class="empty-text">Results will appear here<br>after uploading a leaf image</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════
elif PAGE == "results":
    st.markdown('<div class="page-md">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Session Records</div>
    <div class="sec-heading">Detection History</div>
    <div class="sec-sub">All detections from this session are logged below with severity and confidence data.</div>
    """, unsafe_allow_html=True)

    rows = get_history()

    if not rows:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px;">
          <div style="font-size:40px;margin-bottom:12px;">📭</div>
          <div style="font-size:16px;font-weight:600;color:#475569;margin-bottom:6px;">No detections yet</div>
          <div style="font-size:14px;color:#94a3b8;">Go to the Detect page and upload a leaf image to begin.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total    = len(rows)
        healthy  = sum(1 for r in rows if r["disease"] == "Healthy")
        diseased = total - healthy
        avg_conf = sum(r["confidence"] for r in rows) / total

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card"><div class="metric-val">{total}</div><div class="metric-lbl">Total Scans</div></div>
          <div class="metric-card"><div class="metric-val">{diseased}</div><div class="metric-lbl">Diseased</div></div>
          <div class="metric-card"><div class="metric-val">{healthy}</div><div class="metric-lbl">Healthy</div></div>
          <div class="metric-card"><div class="metric-val">{avg_conf*100:.1f}%</div><div class="metric-lbl">Avg Confidence</div></div>
        </div>
        """, unsafe_allow_html=True)

        tbody = "".join(
            f"<tr>"
            f"<td><strong>{r['disease']}</strong></td>"
            f"<td>{r['confidence']*100:.1f}%</td>"
            f"<td><span class='sev {SEV_STYLE.get(r['severity'], SEV_STYLE['None'])[0]}'>{r['severity']}</span></td>"
            f"<td style='color:#94a3b8;font-size:13px;'>{r['timestamp']}</td>"
            f"</tr>"
            for r in rows
        )
        st.markdown(f"""
        <div class="card" style="padding:0;overflow:hidden;">
          <table class="htable">
            <thead>
              <tr>
                <th>Disease</th>
                <th>Confidence</th>
                <th>Severity</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>{tbody}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _, c, _ = st.columns([4, 1, 4])
        with c:
            if st.button("Clear History", key="clr"):
                clear_history()
                st.success("History cleared.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  ABOUT
# ═══════════════════════════════════════════════════════
elif PAGE == "about":
    st.markdown('<div class="page-sm">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Project Documentation</div>
    <div class="sec-heading">About AgroAI</div>
    <div class="sec-sub">A professional-grade AI system for real-time tomato leaf disease detection.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon">📄</div>
        <div><div class="card-title">Project Overview</div><div class="card-sub">What AgroAI does and why it matters</div></div>
      </div>
      <p style="font-size:14px;color:#475569;line-height:1.85;margin:0;">
        AgroAI is a real-time tomato leaf disease detection system built on the <strong>YOLOv8</strong>
        object detection architecture. It classifies 10 distinct conditions — from critical blights to
        healthy leaves — from a single photograph and delivers immediate severity ratings,
        evidence-based treatment protocols, and actionable prevention recommendations.
        Built for farmers, agronomists, and agricultural researchers.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <div class="card-icon">🤖</div>
            <div><div class="card-title">Model Specifications</div></div>
          </div>
          <div class="spec-row"><span class="spec-label">Architecture</span><span class="spec-val">YOLOv8 (Ultralytics)</span></div>
          <div class="spec-row"><span class="spec-label">Training Images</span><span class="spec-val">10,853</span></div>
          <div class="spec-row"><span class="spec-label">mAP50 Accuracy</span><span class="spec-val">96.7%</span></div>
          <div class="spec-row"><span class="spec-label">Avg. Precision</span><span class="spec-val">94.2%</span></div>
          <div class="spec-row"><span class="spec-label">Avg. Recall</span><span class="spec-val">93.8%</span></div>
          <div class="spec-row"><span class="spec-label">Inference Speed</span><span class="spec-val">3.6 ms / image</span></div>
          <div class="spec-row"><span class="spec-label">Dataset</span><span class="spec-val">PlantVillage</span></div>
          <div class="spec-row"><span class="spec-label">Classes</span><span class="spec-val">10 (9 diseases + healthy)</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <div class="card-icon">🔧</div>
            <div><div class="card-title">Technology Stack</div></div>
          </div>
          <div style="padding-bottom:16px;border-bottom:1px solid #f1f5f9;margin-bottom:16px;">
            <span class="chip">YOLOv8</span><span class="chip">Python 3.10</span>
            <span class="chip">PyTorch</span><span class="chip">Streamlit</span>
            <span class="chip">OpenCV</span><span class="chip">Ultralytics</span>
            <span class="chip">Pillow</span><span class="chip">NumPy</span>
          </div>
          <div class="spec-row"><span class="spec-label">Frontend</span><span class="spec-val">Streamlit</span></div>
          <div class="spec-row"><span class="spec-label">Model Format</span><span class="spec-val">PyTorch (.pt)</span></div>
          <div class="spec-row"><span class="spec-label">Hosting</span><span class="spec-val">Streamlit Cloud</span></div>
          <div class="spec-row"><span class="spec-label">Data Storage</span><span class="spec-val">In-memory session</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon">📊</div>
        <div><div class="card-title">Per-Class Model Performance</div><div class="card-sub">Precision & Recall across all 10 disease categories</div></div>
      </div>
    """, unsafe_allow_html=True)

    for name, prec, rec in CLASS_PERF:
        avg = (prec + rec) / 2
        st.markdown(f"""
        <div class="perf-item">
          <div class="perf-header">
            <span class="perf-name">{name}</span>
            <span class="perf-nums">P: {prec}% · R: {rec}%</span>
          </div>
          <div class="perf-track"><div class="perf-bar" style="width:{avg}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>AgroAI</span> &nbsp;·&nbsp; Tomato Leaf Disease Detection &nbsp;·&nbsp;
  Powered by YOLOv8 &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
