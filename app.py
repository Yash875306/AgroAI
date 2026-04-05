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
    page_title="AgroAI - Tomato Disease Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for _k, _v in {
    "page": "home",
    "history": [],
    "last_result": None,
    "det_out": None,
    "det_res": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def nav(p):
    st.session_state.page = p
    st.session_state.det_out = None
    st.session_state.det_res = None
    st.rerun()

PAGE = st.session_state.page

# ─────────────────────────────────────────────
#  IN-MEMORY HISTORY (replaces SQLite)
# ─────────────────────────────────────────────
def save_detection(disease, confidence, severity):
    st.session_state.history.append({
        "disease":    disease,
        "confidence": confidence,
        "severity":   severity,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

def get_history(limit=50):
    return st.session_state.history[-limit:][::-1]

def clear_history_db():
    st.session_state.history = []

# ─────────────────────────────────────────────
#  DISEASE REFERENCE
# ─────────────────────────────────────────────
DISEASES = {
    "Tomato_Bacterial_spot": {
        "label": "Bacterial Spot", "severity": "High",
        "symptoms":   "Small dark water-soaked lesions on leaves and fruit surfaces.",
        "treatment":  "Apply copper-based bactericides. Remove infected debris. Avoid overhead irrigation.",
        "prevention": "Use certified disease-free seeds. Practice 2-year crop rotation.",
    },
    "Tomato_Early_blight": {
        "label": "Early Blight", "severity": "Medium",
        "symptoms":   "Concentric dark rings forming a target pattern on older leaves.",
        "treatment":  "Apply chlorothalonil or mancozeb fungicide every 7 to 10 days.",
        "prevention": "Rotate crops annually. Remove lower foliage. Mulch around base.",
    },
    "Tomato_Late_blight": {
        "label": "Late Blight", "severity": "Critical",
        "symptoms":   "Large irregular water-soaked grey-green lesions; white mould on underside.",
        "treatment":  "Apply metalaxyl or cymoxanil fungicide immediately. Destroy infected plants.",
        "prevention": "Avoid overhead watering. Plant resistant varieties. Monitor humidity.",
    },
    "Tomato_Leaf_Mold": {
        "label": "Leaf Mold", "severity": "Medium",
        "symptoms":   "Yellow patches on upper leaf surface; olive-green mould on underside.",
        "treatment":  "Apply mancozeb or chlorothalonil. Improve greenhouse ventilation.",
        "prevention": "Reduce relative humidity below 85%. Space plants adequately.",
    },
    "Tomato_Septoria_leaf_spot": {
        "label": "Septoria Leaf Spot", "severity": "Medium",
        "symptoms":   "Small circular spots with dark borders and pale grey centres.",
        "treatment":  "Apply copper fungicide. Remove heavily infected leaves promptly.",
        "prevention": "Mulch soil. Avoid wetting foliage during irrigation.",
    },
    "Tomato_Spider_mites Two-spotted_spider_mite": {
        "label": "Spider Mites", "severity": "Low",
        "symptoms":   "Fine yellow stippling on leaves; fine webbing on leaf undersides.",
        "treatment":  "Apply miticide or neem oil. Increase ambient humidity.",
        "prevention": "Regular scouting. Introduce predatory mites as biocontrol.",
    },
    "Tomato__Target_Spot": {
        "label": "Target Spot", "severity": "Medium",
        "symptoms":   "Bulls-eye concentric ring lesions on leaves and stems.",
        "treatment":  "Apply azoxystrobin or fluxapyroxad. Improve field drainage.",
        "prevention": "Remove plant debris after harvest. Avoid dense canopy.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "label": "Yellow Leaf Curl Virus", "severity": "Critical",
        "symptoms":   "Upward leaf curling, yellowing margins, stunted plant growth.",
        "treatment":  "No chemical cure. Remove and destroy infected plants immediately.",
        "prevention": "Control whitefly populations. Use insect-proof nets and resistant varieties.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "label": "Tomato Mosaic Virus", "severity": "High",
        "symptoms":   "Mosaic light-dark green patterns on leaves; distortion and stunting.",
        "treatment":  "No cure. Remove infected plants. Disinfect all tools with bleach solution.",
        "prevention": "Use virus-free certified seeds. Wash hands before handling plants.",
    },
    "Tomato_healthy": {
        "label": "Healthy", "severity": "None",
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

SEV_BADGE = {
    "None":     "badge-none",
    "Low":      "badge-low",
    "Medium":   "badge-medium",
    "High":     "badge-high",
    "Critical": "badge-critical",
}
SEV_COLOR = {
    "None":     "#15803d",
    "Low":      "#ca8a04",
    "Medium":   "#ea580c",
    "High":     "#dc2626",
    "Critical": "#9d174d",
}

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import os
    if not os.path.exists("best.pt"):
        return None, "Model file best.pt not found."
    try:
        return YOLO("best.pt"), None
    except Exception as exc:
        return None, str(exc)

def run_inference(model, pil_image):
    arr     = np.array(pil_image)
    results = model.predict(arr, conf=0.25, verbose=False)
    r       = results[0]
    if r.probs is not None:
        probs = r.probs.data.cpu().numpy()
        idx   = int(np.argmax(probs))
        return [(model.names[idx], float(probs[idx]))], pil_image
    detections = []
    annotated  = pil_image
    if r.boxes and len(r.boxes):
        annotated = Image.fromarray(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
        for box in r.boxes:
            detections.append((
                model.names[int(box.cls.item())],
                float(box.conf.item())
            ))
    return detections or [("Tomato_healthy", 1.0)], annotated

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --g900:#0f3d1f; --g800:#145228; --g700:#1a6b34;
    --g600:#218a42; --g500:#28a745; --g400:#48c267;
    --g300:#7dd99a; --g100:#e8f5ed; --g50:#f2fbf5;
    --white:#ffffff;
    --grey900:#111827; --grey700:#374151; --grey500:#6b7280;
    --grey300:#d1d5db; --grey100:#f3f4f6; --grey50:#fafafa;
    --red:#b91c1c; --orange:#c2410c; --yellow:#854d0e; --pink:#9d174d;
    --r:10px;
    --sh:0 1px 3px rgba(0,0,0,0.07),0 4px 14px rgba(0,0,0,0.05);
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; font-family:'DM Sans',sans-serif; }
.stApp { background:var(--grey100) !important; }
[data-testid="stSidebar"],[data-testid="collapsedControl"],
[data-testid="stDecoration"],[data-testid="stToolbar"],
#MainMenu,footer,header { display:none !important; }
.block-container { padding:0 !important; max-width:100% !important; }
section[data-testid="stMain"]>div { padding:0 !important; }

/* Hide nav button row */
section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:first-of-type {
    position:fixed !important; top:-9999px !important; left:-9999px !important;
    width:0 !important; height:0 !important; overflow:hidden !important;
    opacity:0 !important; pointer-events:none !important;
}

.topbar {
    position:fixed; top:0; left:0; right:0; height:60px; z-index:9999;
    background:var(--white); border-bottom:1px solid var(--grey300);
    display:flex; align-items:center; padding:0 44px; gap:0;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
}
.tb-brand { font-family:'DM Serif Display',serif; font-size:21px; color:var(--g800); white-space:nowrap; flex-shrink:0; margin-right:44px; letter-spacing:-0.3px; }
.tb-brand span { color:var(--g500); }
.tb-nav { display:flex; align-items:center; gap:2px; }
.tb-btn { font-size:14px; font-weight:500; color:var(--grey500); padding:7px 18px; border-radius:7px; cursor:pointer; border:none; background:transparent; white-space:nowrap; transition:background .15s,color .15s; line-height:1; font-family:'DM Sans',sans-serif; letter-spacing:0.1px; }
.tb-btn:hover  { background:var(--g50);  color:var(--g700); }
.tb-btn.active { background:var(--g100); color:var(--g800); font-weight:600; }
.tb-spacer { flex:1; }
.tb-pill { font-size:11px; font-weight:700; letter-spacing:0.6px; color:var(--g700); background:var(--g100); border:1px solid var(--g300); padding:4px 14px; border-radius:20px; white-space:nowrap; }
.main-wrap { margin-top:60px; }

.page    { padding:48px 56px 80px; max-width:1180px; margin:0 auto; }
.page-sm { padding:48px 56px 80px; max-width:860px;  margin:0 auto; }

.section-label { font-size:11px; font-weight:700; letter-spacing:1.6px; text-transform:uppercase; color:var(--g600); margin-bottom:8px; }
.page-heading  { font-family:'DM Serif Display',serif; font-size:30px; color:var(--g900); margin-bottom:28px; line-height:1.15; font-weight:400; }
.page-heading em { color:var(--g500); font-style:italic; }

.card { background:var(--white); border:1px solid var(--grey300); border-radius:var(--r); padding:28px 32px; box-shadow:var(--sh); margin-bottom:20px; }
.card-title { font-size:14px; font-weight:700; color:var(--grey900); letter-spacing:0.2px; margin-bottom:18px; padding-bottom:14px; border-bottom:1px solid var(--grey100); }

.badge { display:inline-block; font-size:11px; font-weight:700; letter-spacing:0.4px; text-transform:uppercase; padding:3px 10px; border-radius:5px; white-space:nowrap; }
.badge-none     { background:#dcfce7; color:#15803d; }
.badge-low      { background:#fef9c3; color:#854d0e; }
.badge-medium   { background:#fff7ed; color:#c2410c; }
.badge-high     { background:#fee2e2; color:#b91c1c; }
.badge-critical { background:#fce7f3; color:#9d174d; }

.conf-wrap { background:var(--grey100); border-radius:4px; height:8px; margin:8px 0 16px; overflow:hidden; }
.conf-fill { height:8px; border-radius:4px; }

.hero { background:linear-gradient(135deg, var(--g900) 0%, var(--g700) 100%); padding:80px 56px 72px; color:white; text-align:center; }
.hero-tag { display:inline-block; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; background:rgba(255,255,255,0.12); color:rgba(255,255,255,0.88); border:1px solid rgba(255,255,255,0.22); padding:5px 18px; border-radius:20px; margin-bottom:28px; }
.hero-title { font-family:'DM Serif Display',serif; font-size:52px; font-weight:400; line-height:1.1; letter-spacing:-1.5px; margin-bottom:20px; color:white; }
.hero-title em { color:var(--g300); font-style:italic; }
.hero-sub { font-size:17px; line-height:1.78; color:rgba(255,255,255,0.65); max-width:520px; margin:0 auto 36px; }

.stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:40px; }
.stat-tile { background:var(--white); border:1px solid var(--grey300); border-radius:var(--r); padding:24px 20px; text-align:center; box-shadow:var(--sh); border-top:3px solid var(--g500); }
.stat-n { font-family:'DM Serif Display',serif; font-size:34px; color:var(--g700); margin-bottom:5px; }
.stat-l { font-size:11px; font-weight:700; color:var(--grey500); text-transform:uppercase; letter-spacing:1px; }

.steps-col { display:flex; flex-direction:column; gap:14px; }
.step-card { background:var(--white); border:1px solid var(--grey300); border-radius:var(--r); padding:22px 24px; box-shadow:var(--sh); }
.step-num { width:34px; height:34px; background:var(--g100); color:var(--g700); border-radius:8px; display:inline-flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; margin-bottom:12px; }
.step-title { font-size:14px; font-weight:600; color:var(--grey900); margin-bottom:6px; }
.step-desc  { font-size:13px; color:var(--grey500); line-height:1.65; }

.dtable { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.drow { background:var(--white); border:1px solid var(--grey300); border-radius:8px; padding:11px 14px; display:flex; align-items:center; gap:10px; box-shadow:var(--sh); }
.drow-name { flex:1; font-size:13px; font-weight:500; color:var(--grey700); }

.panel { background:var(--white); border:1px solid var(--grey300); border-radius:var(--r); padding:28px; box-shadow:var(--sh); }
.panel-title { font-size:12px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:var(--grey500); margin-bottom:18px; padding-bottom:14px; border-bottom:1px solid var(--grey100); }
.empty-box { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:56px 20px; color:var(--grey300); font-size:13px; text-align:center; gap:10px; line-height:1.6; }
.empty-icon { width:52px; height:52px; border-radius:50%; background:var(--grey100); display:flex; align-items:center; justify-content:center; font-size:20px; color:var(--grey300); margin-bottom:4px; }

.result-block { border:1px solid var(--grey300); border-radius:8px; padding:20px 22px; margin-bottom:12px; border-left:4px solid var(--g500); }
.result-name  { font-size:18px; font-weight:700; color:var(--grey900); margin-bottom:4px; }
.result-conf  { font-size:13px; color:var(--grey500); margin-bottom:10px; }
.result-label { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:var(--grey500); margin-bottom:4px; }
.result-value { font-size:14px; color:var(--grey700); line-height:1.65; margin-bottom:14px; }

.htable { width:100%; border-collapse:collapse; font-size:14px; }
.htable th { font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:var(--grey500); padding:10px 16px; text-align:left; background:var(--grey50); border-bottom:2px solid var(--grey100); }
.htable td { color:var(--grey700); padding:12px 16px; border-bottom:1px solid var(--grey100); vertical-align:middle; }
.htable tr:last-child td { border-bottom:none; }
.htable tr:hover td { background:var(--g50); }

.perf-row { margin-bottom:15px; }
.perf-head { display:flex; justify-content:space-between; margin-bottom:5px; }
.perf-name { font-size:13px; font-weight:500; color:var(--grey700); }
.perf-vals { font-size:12px; color:var(--grey500); }
.perf-bg   { background:var(--grey100); border-radius:4px; height:7px; overflow:hidden; }
.perf-fill { height:7px; border-radius:4px; background:linear-gradient(90deg,var(--g600),var(--g400)); }

.info-row { display:flex; gap:12px; padding:8px 0; border-bottom:1px solid var(--grey100); }
.info-row:last-child { border-bottom:none; }
.info-label { font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:var(--grey500); min-width:140px; }
.info-value { font-size:14px; color:var(--grey700); }
.chip { display:inline-block; font-size:12px; font-weight:600; color:var(--g700); background:var(--g100); border:1px solid var(--g300); padding:4px 12px; border-radius:6px; margin:3px 4px 0 0; }
code { background:var(--grey100); padding:2px 7px; border-radius:4px; font-size:13px; color:var(--grey700); font-family:'DM Mono','Courier New',monospace; }

.stButton>button { background:var(--g600) !important; color:white !important; border:none !important; border-radius:var(--r) !important; padding:11px 30px !important; font-size:14px !important; font-weight:600 !important; font-family:'DM Sans',sans-serif !important; transition:background .15s,box-shadow .15s !important; }
.stButton>button:hover { background:var(--g700) !important; box-shadow:0 4px 14px rgba(33,138,66,0.28) !important; }

[data-testid="stFileUploader"] { border:2px dashed var(--g300) !important; border-radius:var(--r) !important; background:var(--g50) !important; }
[data-testid="stImage"] img { border-radius:8px; }
[data-testid="stMetric"] { background:var(--white) !important; border:1px solid var(--grey300) !important; border-radius:var(--r) !important; padding:18px 20px !important; }
[data-testid="stMetricValue"] { color:var(--g700) !important; font-size:26px !important; font-family:'DM Serif Display',serif !important; font-weight:400 !important; }
[data-testid="stMetricLabel"] { color:var(--grey500) !important; font-size:11px !important; }

.footer { text-align:center; padding:22px 40px; font-size:12px; color:var(--grey300); border-top:1px solid var(--grey100); background:var(--white); margin-top:60px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TOPBAR
# ─────────────────────────────────────────────
def _a(p): return "active" if PAGE == p else ""

st.markdown(f"""
<div class="topbar">
  <div class="tb-brand">Agro<span>AI</span></div>
  <nav class="tb-nav">
    <button class="tb-btn {_a('home')}"    onclick="window._go('home')">Home</button>
    <button class="tb-btn {_a('detect')}"  onclick="window._go('detect')">Detect</button>
    <button class="tb-btn {_a('results')}" onclick="window._go('results')">Results</button>
    <button class="tb-btn {_a('about')}"   onclick="window._go('about')">About</button>
  </nav>
  <div class="tb-spacer"></div>
  <span class="tb-pill">YOLOv8 &nbsp;&middot;&nbsp; 96.7% mAP</span>
</div>
<div class="main-wrap"></div>
<script>
window._go = function(p) {{
  var all = window.parent.document.querySelectorAll('.stButton button');
  for (var i = 0; i < all.length; i++) {{
    if (all[i].innerText.trim() === p) {{ all[i].click(); return; }}
  }}
}};
</script>
""", unsafe_allow_html=True)

_nc = st.columns(4)
with _nc[0]:
    if st.button("home",    key="_nh"): nav("home")
with _nc[1]:
    if st.button("detect",  key="_nd"): nav("detect")
with _nc[2]:
    if st.button("results", key="_nr"): nav("results")
with _nc[3]:
    if st.button("about",   key="_na"): nav("about")

# ═══════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════
if PAGE == "home":
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">YOLOv8 &nbsp;&middot;&nbsp; Precision Agriculture &nbsp;&middot;&nbsp; Real-time Detection</div>
      <div class="hero-title">AI-Powered Tomato<br><em>Disease Detection</em></div>
      <div class="hero-sub">Upload a single leaf photograph and receive an instant AI diagnosis complete with severity rating, treatment protocol, and prevention recommendations.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="stats-row">
      <div class="stat-tile"><div class="stat-n">96.7%</div><div class="stat-l">mAP50 Accuracy</div></div>
      <div class="stat-tile"><div class="stat-n">10</div><div class="stat-l">Disease Classes</div></div>
      <div class="stat-tile"><div class="stat-n">3.6 ms</div><div class="stat-l">Inference Speed</div></div>
      <div class="stat-tile"><div class="stat-n">10,853</div><div class="stat-l">Training Images</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("""
        <div class="section-label">How It Works</div>
        <div class="page-heading">Three steps<br>to a diagnosis</div>
        <div class="steps-col">
          <div class="step-card"><div class="step-num">01</div><div class="step-title">Upload a Leaf Photograph</div><div class="step-desc">Take a clear, well-lit photo of a tomato leaf and upload it in JPG or PNG format.</div></div>
          <div class="step-card"><div class="step-num">02</div><div class="step-title">AI Analyses the Image</div><div class="step-desc">YOLOv8 scans for visual disease markers and returns bounding-box detections in under 4 ms.</div></div>
          <div class="step-card"><div class="step-num">03</div><div class="step-title">Receive Treatment Advice</div><div class="step-desc">View the severity rating, confidence score, symptoms, treatment steps, and prevention guidance.</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="section-label">Disease Reference</div>
        <div class="page-heading">10 Detectable<br>Conditions</div>
        """, unsafe_allow_html=True)
        rows_html = "".join(
            f'<div class="drow"><span class="drow-name">{v["label"]}</span>'
            f'<span class="badge {SEV_BADGE[v["severity"]]}">{v["severity"]}</span></div>'
            for v in DISEASES.values()
        )
        st.markdown(f'<div class="dtable">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    _, cta, _ = st.columns([3, 1, 3])
    with cta:
        if st.button("Start Detection", key="cta"): nav("detect")
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  DETECT
# ═══════════════════════════════════════════════════════
elif PAGE == "detect":
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">Computer Vision Analysis</div>
    <div class="page-heading">Disease Detection</div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="panel"><div class="panel-title">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Select a tomato leaf image (JPG or PNG)", type=["jpg","jpeg","png"], key="uploader")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True)
        else:
            st.markdown('<div class="empty-box"><div class="empty-icon">+</div><div>Select a JPG or PNG image<br>to begin analysis</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Detection Results</div>', unsafe_allow_html=True)
        if uploaded:
            model, err = load_model()
            if err:
                st.error(f"Model error: {err}")
            else:
                with st.spinner("Running inference..."):
                    detections, annotated = run_inference(model, img)

                if annotated is not img:
                    st.image(annotated, use_column_width=True, caption="Annotated output")

                for cls_name, conf in detections:
                    info     = DISEASES.get(cls_name, DISEASES["Tomato_healthy"])
                    sev      = info["severity"]
                    bc       = SEV_COLOR.get(sev, "#218a42")
                    conf_pct = int(conf * 100)
                    st.markdown(f"""
                    <div class="result-block" style="border-left-color:{bc};">
                      <div class="result-name">{info['label']}</div>
                      <div class="result-conf">Confidence: <strong>{conf_pct}%</strong> &nbsp;&nbsp; <span class="badge {SEV_BADGE[sev]}">{sev}</span></div>
                      <div class="conf-wrap"><div class="conf-fill" style="width:{conf_pct}%;background:{bc};"></div></div>
                      <div class="result-label">Symptoms</div><div class="result-value">{info['symptoms']}</div>
                      <div class="result-label">Treatment</div><div class="result-value">{info['treatment']}</div>
                      <div class="result-label">Prevention</div><div class="result-value" style="margin-bottom:0">{info['prevention']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    save_detection(info["label"], conf, sev)
        else:
            st.markdown('<div class="empty-box"><div class="empty-icon">-</div><div>Results will appear here<br>after uploading an image</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  RESULTS / HISTORY
# ═══════════════════════════════════════════════════════
elif PAGE == "results":
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">Session Records</div>
    <div class="page-heading">Detection History</div>
    """, unsafe_allow_html=True)

    rows = get_history()

    if not rows:
        st.info("No detections recorded yet. Navigate to Detect and upload a leaf image to begin.")
    else:
        total    = len(rows)
        healthy  = sum(1 for r in rows if r["disease"] == "Healthy")
        diseased = total - healthy
        avg_conf = sum(r["confidence"] for r in rows) / total

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Scans",     total)
        m2.metric("Diseased",        diseased)
        m3.metric("Healthy",         healthy)
        m4.metric("Avg. Confidence", f"{avg_conf*100:.1f}%")

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        tbody = "".join(
            f"<tr>"
            f"<td>{r['disease']}</td>"
            f"<td>{r['confidence']*100:.1f}%</td>"
            f"<td><span class='badge {SEV_BADGE.get(r['severity'],'badge-none')}'>{r['severity']}</span></td>"
            f"<td style='color:#9ca3af;font-size:13px'>{r['timestamp']}</td>"
            f"</tr>"
            for r in rows
        )
        st.markdown(f"""
        <div class="card" style="padding:0;overflow:hidden;">
          <table class="htable">
            <thead><tr><th>Disease</th><th>Confidence</th><th>Severity</th><th>Timestamp</th></tr></thead>
            <tbody>{tbody}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([3, 1, 3])
        with btn_col:
            if st.button("Clear All History", key="clr"):
                clear_history_db()
                st.success("All history records cleared.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  ABOUT
# ═══════════════════════════════════════════════════════
elif PAGE == "about":
    st.markdown('<div class="page-sm">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">Project Documentation</div>
    <div class="page-heading">About AgroAI</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">Project Overview</div>
      <p style="font-size:14px;color:#374151;line-height:1.85;margin:0;">
        AgroAI is a real-time tomato leaf disease detection system built on the YOLOv8 object
        detection architecture. It classifies 10 distinct disease conditions from a single leaf
        photograph and delivers immediate severity ratings, evidence-based treatment protocols,
        and actionable prevention recommendations.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Model Specifications</div>
          <div class="info-row"><span class="info-label">Architecture</span><span class="info-value">YOLOv8</span></div>
          <div class="info-row"><span class="info-label">Training Images</span><span class="info-value">10,853</span></div>
          <div class="info-row"><span class="info-label">mAP50 Accuracy</span><span class="info-value">96.7%</span></div>
          <div class="info-row"><span class="info-label">Avg. Precision</span><span class="info-value">94.2%</span></div>
          <div class="info-row"><span class="info-label">Avg. Recall</span><span class="info-value">93.8%</span></div>
          <div class="info-row"><span class="info-label">Inference Speed</span><span class="info-value">3.6 ms / image</span></div>
          <div class="info-row"><span class="info-label">Dataset</span><span class="info-value">PlantVillage</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Technology Stack</div>
          <div>
            <span class="chip">YOLOv8</span><span class="chip">Python 3.10</span>
            <span class="chip">PyTorch</span><span class="chip">Streamlit</span>
            <span class="chip">OpenCV</span><span class="chip">Ultralytics</span>
            <span class="chip">Pillow</span><span class="chip">NumPy</span>
          </div>
        </div>
        <div class="card">
          <div class="card-title">Run Locally</div>
          <div style="font-size:14px;color:#374151;line-height:2.2;">
            1. Place <code>best.pt</code> in the same directory as <code>app.py</code><br>
            2. Run <code>pip install streamlit ultralytics opencv-python-headless pillow</code><br>
            3. Run <code>streamlit run app.py</code><br>
            4. Open <code>localhost:8501</code> in your browser
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">Per-Class Model Performance</div>
    """, unsafe_allow_html=True)
    for name, prec, rec in CLASS_PERF:
        avg = (prec + rec) / 2
        st.markdown(f"""
        <div class="perf-row">
          <div class="perf-head">
            <span class="perf-name">{name}</span>
            <span class="perf-vals">Precision {prec}%&nbsp;&nbsp;&middot;&nbsp;&nbsp;Recall {rec}%</span>
          </div>
          <div class="perf-bg"><div class="perf-fill" style="width:{avg}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  AgroAI &nbsp;&middot;&nbsp; Tomato Leaf Disease Detection &nbsp;&middot;&nbsp; Powered by YOLOv8
</div>
""", unsafe_allow_html=True)
