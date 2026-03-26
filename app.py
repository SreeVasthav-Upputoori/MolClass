"""
app.py — Classification Model Builder  (Premium UI Edition)
=============================================================
Dark-theme, top-tab navigation, SVG icons, glassmorphism cards,
animated pipeline diagram, About section.
"""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MolClass — Binary Classification Builder",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⬡</text></svg>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design Tokens & CSS ───────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── CSS Variables ─── */
:root {
  --bg:          #0b0f19; /* Deep neutral dark slate */
  --surface:     #111827; /* Dark grayish blue */
  --surface2:    #1f2937; /* Elevated grayish blue */
  --card:        #1e293b; /* Light slate blue for cards/tiles */
  --border:      rgba(255, 255, 255, 0.08);
  --border-glow: rgba(52, 211, 153, 0.35);
  --teal:        #10b981; /* Vibrant Emerald/Teal green accent */
  --teal-dim:    rgba(16, 185, 129, 0.15);
  --teal-dark:   #059669;
  --amber:       #fbbf24;
  --amber-dim:   rgba(251, 191, 36, 0.15);
  --purple:      #8b5cf6;
  --purple-dim:  rgba(139, 92, 246, 0.15);
  --red:         #ef4444;
  --text:        #f8fafc; /* Crisp white text for readabilty */
  --text-muted:  #94a3b8; /* Slate text muted */
  --text-faint:  #475569;
  --radius:      12px;
  --radius-sm:   8px;
  --shadow:      0 8px 32px rgba(0,0,0,0.4);
  --shadow-glow: 0 0 24px rgba(16, 185, 129, 0.15);
}

/* ─── Global Reset ─── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text);
}

.main .block-container {
  padding: 0 2rem 4rem 2rem;
  max-width: 1280px;
}

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none !important; }

/* ─── Top app bar ─── */
.app-bar {
  position: sticky;
  top: 0;
  z-index: 999;
  background: rgba(8,13,20,0.88);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  padding: 0 2rem;
  display: flex;
  align-items: center;
  height: 62px;
  margin-bottom: 2rem;
}
.app-bar-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-right: auto;
}
.app-bar-logo-text {
  font-size: 1.8rem;
  font-weight: 800;
  color: var(--teal);
  letter-spacing: -0.02em;
}
.app-bar-logo-sub {
  font-size: 0.95rem;
  color: var(--text-muted);
  font-weight: 400;
}

/* ─── Tab Navigation ─── */
[data-testid="stTabs"] {
  background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 50px !important;
  padding: 5px !important;
  gap: 4px !important;
  width: fit-content !important;
  margin: 0 auto 2.5rem auto !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-radius: 40px !important;
  padding: 8px 22px !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  border: none !important;
  outline: none !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.02em !important;
  white-space: nowrap !important;
}
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {
  background: var(--teal) !important;
  color: #080d14 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  display: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
  display: none !important;
}

/* ─── Glass Card ─── */
.glass-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.glass-card:hover {
  border-color: var(--border-glow);
  box-shadow: var(--shadow-glow);
}
.glass-card-teal  { border-left: 3px solid var(--teal);   }
.glass-card-amber { border-left: 3px solid var(--amber);  }
.glass-card-purple{ border-left: 3px solid var(--purple); }

/* ─── Section Headers ─── */
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 2.5rem 0 1.2rem 0;
}
.section-header-icon {
  width: 36px; height: 36px;
  border-radius: 9px;
  background: var(--teal-dim);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.section-header h2 {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.01em;
}
.section-header p {
  margin: 0;
  font-size: 0.78rem;
  color: var(--text-muted);
}

/* ─── Metric Cards ─── */
.metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-card {
  flex: 1;
  min-width: 130px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.1rem 1.2rem;
}
.metric-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text-muted);
  margin-bottom: 0.4rem;
}
.metric-value {
  font-size: 1.85rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: var(--text);
}
.metric-value.teal   { color: var(--teal);   }
.metric-value.amber  { color: var(--amber);  }
.metric-value.red    { color: var(--red);    }
.metric-sub { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.2rem; }

/* ─── Pipeline Progress Bar ─── */
.pipeline-steps {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  margin: 0.5rem 0 2.5rem 0;
  position: relative;
}
.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  flex: 1;
  max-width: 140px;
}
.step-circle {
  width: 42px; height: 42px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem;
  font-weight: 700;
  border: 2px solid var(--border);
  background: var(--surface2);
  color: var(--text-muted);
  position: relative;
  z-index: 2;
  transition: all 0.3s ease;
}
.step-circle.done {
  background: var(--teal);
  border-color: var(--teal);
  color: #080d14;
  box-shadow: 0 0 16px rgba(6,214,160,0.4);
}
.step-circle.active {
  background: var(--card);
  border-color: var(--teal);
  color: var(--teal);
  box-shadow: 0 0 12px rgba(6,214,160,0.2);
}
.step-label {
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--text-muted);
  margin-top: 0.45rem;
  text-align: center;
  letter-spacing: 0.02em;
  line-height: 1.3;
}
.step-label.done   { color: var(--teal);  }
.step-label.active { color: var(--text);  }
.step-connector {
  flex: 1; height: 2px; background: var(--border); position: relative; top: -22px; margin: 0 -2px;
}
.step-connector.done { background: var(--teal); }

/* ─── Buttons ─── */
.stButton > button {
  border-radius: 9px !important;
  font-size: 0.87rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  padding: 0.55rem 1.4rem !important;
  transition: all 0.2s ease !important;
  border: none !important;
}
.stButton > button[kind="primary"] {
  background: var(--teal) !important;
  color: #080d14 !important;
}
.stButton > button[kind="primary"] * {
  color: #080d14 !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
  background: #08f5b8 !important;
  box-shadow: 0 0 22px rgba(6,214,160,0.35) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--teal) !important;
  color: var(--teal) !important;
}

/* ─── Download Buttons ─── */
[data-testid="stDownloadButton"] > button {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--teal) !important;
  border-radius: 9px !important;
  font-weight: 600 !important;
  transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--teal) !important;
  background: var(--teal-dim) !important;
  transform: translateY(-1px) !important;
}

/* ─── File Uploader ─── */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  background: var(--surface2) !important;
  border-radius: var(--radius) !important;
  padding: 1rem !important;
  transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--teal) !important;
}
[data-testid="stFileUploader"] * { color: var(--text) !important; }

/* ─── Inputs ─── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
  background: var(--surface2) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
}
[data-baseweb="select"] span { color: var(--text) !important; }
[data-baseweb="popover"] { background: var(--card) !important; }
[data-baseweb="menu-item"] { background: var(--card) !important; color: var(--text) !important; }

/* ─── Checkboxes ─── */
[data-testid="stCheckbox"] label { color: var(--text) !important; font-size: 0.88rem !important; }

/* ─── Sliders ─── */
[data-testid="stSlider"] [data-baseweb="slider"] > div:last-child > div {
  background: var(--teal) !important;
}

/* ─── Radio ─── */
[data-testid="stRadio"] label { color: var(--text) !important; font-size: 0.87rem !important; }

/* ─── Expanders ─── */
[data-testid="stExpander"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; font-weight: 600 !important; }
[data-testid="stExpander"] svg { fill: var(--text-muted) !important; }

/* ─── DataFrames ─── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface2) !important;
}
[data-testid="stDataFrame"] * { color: var(--text) !important;}
.dvn-scroller { background: var(--surface2) !important; }

/* ─── Alerts ─── */
[data-testid="stAlert"] {
  border-radius: var(--radius-sm) !important;
  font-size: 0.87rem !important;
  border-left-width: 3px !important;
}
div[data-testid="stAlert"][data-type="info"] {
  background: rgba(6,214,160,0.08) !important;
  border-color: var(--teal) !important;
  color: #b4e8db !important;
}
div[data-testid="stAlert"][data-type="warning"] {
  background: rgba(255,183,3,0.08) !important;
  border-color: var(--amber) !important;
  color: #f0d48a !important;
}
div[data-testid="stAlert"][data-type="error"] {
  background: rgba(239,68,68,0.08) !important;
  border-color: var(--red) !important;
  color: #fca5a5 !important;
}
div[data-testid="stAlert"][data-type="success"] {
  background: rgba(6,214,160,0.1) !important;
  border-color: var(--teal) !important;
  color: #a7f3d0 !important;
}

/* ─── Progress bar ─── */
[data-testid="stProgress"] > div > div { background: var(--teal) !important; }
[data-testid="stProgress"] > div { background: var(--surface2) !important; border-radius: 99px !important; }

/* ─── Metrics ─── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label { color: var(--text-muted) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.7rem !important; font-weight: 800 !important; }

/* ─── Divider ─── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ─── Spinner ─── */
[data-testid="stSpinner"] { color: var(--teal) !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--card); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal-dim); }

/* ─── Typography ─── */
h1 { font-size: 1.9rem !important; font-weight: 800 !important; color: var(--text) !important; letter-spacing: -0.03em !important; }
h2 { font-size: 1.15rem !important; font-weight: 700 !important; color: var(--text) !important; }
h3 { font-size: 0.97rem !important; font-weight: 700 !important; color: var(--text) !important; }
p  { font-size: 0.9rem; color: var(--text-muted); line-height: 1.7; }
small { color: var(--text-muted); font-size: 0.78rem; }

/* ─── Code ─── */
code, .stCode {
  font-family: 'JetBrains Mono', monospace !important;
  background: var(--surface2) !important;
  color: var(--teal) !important;
  border-radius: 4px !important;
  padding: 0.1em 0.4em !important;
}

/* ─── Status tag ─── */
.tag {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 99px;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.tag-teal   { background: var(--teal-dim);   color: var(--teal);   }
.tag-amber  { background: var(--amber-dim);  color: var(--amber);  }
.tag-purple { background: var(--purple-dim); color: var(--purple); }

/* ─── Hero gradient text ─── */
.gradient-text {
  background: linear-gradient(135deg, var(--teal) 0%, #60efff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* ─── About feature grid ─── */
.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 1.5rem 0;
}
.feature-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.3rem;
  transition: border-color 0.2s, transform 0.2s;
}
.feature-card:hover {
  border-color: var(--border-glow);
  transform: translateY(-3px);
}
.feature-icon {
  width: 38px; height: 38px;
  border-radius: 9px;
  margin-bottom: 0.8rem;
  display: flex; align-items: center; justify-content: center;
}
.feature-title {
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.35rem;
}
.feature-desc {
  font-size: 0.76rem;
  color: var(--text-muted);
  line-height: 1.55;
}

/* ─── Step info card inside pages ─── */
.step-info-bar {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.85rem 1.2rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.8rem;
}
.step-info-bar-num {
  background: var(--teal-dim);
  color: var(--teal);
  border-radius: 8px;
  padding: 0.35rem 0.7rem;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  white-space: nowrap;
}
.step-info-bar-text { font-size: 0.84rem; color: var(--text-muted); }
.step-info-bar-text strong { color: var(--text); }

/* ─── Table styles ─── */
table { width: 100% !important; border-collapse: separate !important; border-spacing: 0 !important; }
thead tr th { background: var(--surface2) !important; color: var(--text-muted) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.07em; padding: 0.7rem 1rem !important; }
tbody tr td { background: var(--card) !important; color: var(--text) !important; padding: 0.65rem 1rem !important; font-size: 0.86rem !important; }
tbody tr:hover td { background: var(--surface2) !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── Shared UI Imports ────────────────────────────────────────────────────────
from utils.ui_components import icon, section_header, step_info_bar

# ── Top App Bar ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-bar">
  <div class="app-bar-logo">
    {icon("molecule", 36, "#06d6a0")}
    <div>
      <div class="app-bar-logo-text">MolClass</div>
      <div class="app-bar-logo-sub">Binary Classification Builder</div>
    </div>
  </div>
  <span class="tag tag-teal">v1.0</span>
</div>
""", unsafe_allow_html=True)

# ── Session State Defaults ────────────────────────────────────────────────────
STATE_DEFAULTS = {
    "upload_done": False, "validation_done": False, "step0_complete": False,
    "step1_complete": False, "step2_complete": False, "step3_complete": False,
    "step4_complete": False, "raw_df": None, "valid_df": None, "invalid_df": None,
    "validation_report": None, "train_df": None, "test_df": None, "val_df": None,
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "feature_columns": None, "fs_report": None, "benchmark_results": None,
    "trained_models": None, "chosen_model_name": None, "chosen_model": None,
    "final_model": None, "final_model_name": None, "apply_smote": False,
    "prediction_results": None,
}
for k, v in STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def done(k): return bool(st.session_state.get(k))


# ── Pipeline Progress Strip ───────────────────────────────────────────────────
steps_meta = [
    ("0", "Upload",     "step0_complete"),
    ("1", "Preprocess", "step1_complete"),
    ("2", "Features",   "step2_complete"),
    ("3", "Benchmark",  "step3_complete"),
    ("4", "Final Model","step4_complete"),
]

def _progress_strip():
    circles = ""
    for i, (num, label, gate) in enumerate(steps_meta):
        is_done = done(gate)
        prev_done = done(steps_meta[i-1][2]) if i > 0 else True
        is_active = not is_done and prev_done
        c_cls = "done" if is_done else ("active" if is_active else "")
        l_cls = "done" if is_done else ("active" if is_active else "")
        inner = icon("check", 16, "#080d14") if is_done else f'<span style="font-size:0.78rem;font-weight:800;">{num}</span>'
        connector = "" if i == 0 else f'<div class="step-connector {"done" if done(steps_meta[i-1][2]) else ""}"></div>'
        circles += f'{connector}<div class="pipeline-step"><div class="step-circle {c_cls}">{inner}</div><div class="step-label {l_cls}">{label}</div></div>'

    st.markdown(f'<div class="pipeline-steps">{circles}</div>', unsafe_allow_html=True)

_progress_strip()


# ── Top Navigation Tabs ───────────────────────────────────────────────────────
from pages import step0_upload, step1_preprocessing, step2_features
from pages import step3_benchmarking, step4_final_model, prediction

tab_labels = ["About", "Upload", "Preprocess", "Features", "Benchmark", "Final Model", "Predict"]
tabs = st.tabs(tab_labels)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — About
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem 1.5rem 2rem;">
      <div style="font-size:0.78rem; font-weight:700; letter-spacing:0.12em; color:var(--teal); text-transform:uppercase; margin-bottom:1rem;">
        Binary Classification for SMILES Data
      </div>
      <h1 style="font-size:3.5rem; margin-bottom:1rem; color:#f8fafc !important;">
        Build. Benchmark.<br>
        <span class="gradient-text">Deploy.</span>
      </h1>
      <p style="max-width:620px; margin:0 auto 2.5rem auto; font-size:1rem; line-height:1.8; color:var(--text-muted);">
        MolClass is an end-to-end interactive ML pipeline for predicting the
        biological activity of molecules — from raw SMILES input to a
        production-ready exported model, with zero code required.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    features = [
        ("molecule",  "var(--teal-dim)",   "#06d6a0", "Molecular Descriptors",      "RDKit physicochemical (~200), Morgan ECFP fingerprints, and MACCS structural keys — all computed on the fly."),
        ("trophy",    "var(--amber-dim)",  "#ffb703", "5-Model Benchmarking",       "Random Forest, LightGBM, XGBoost, SVM, and Logistic Regression — tuned via RandomizedSearchCV with 5-fold CV."),
        ("chart",     "var(--purple-dim)", "#7c6fe0", "Rich Evaluation Metrics",   "Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1, and ROC-AUC — with confusion matrix and ROC curve."),
        ("target",    "var(--teal-dim)",   "#06d6a0", "Leakage-Free Design",        "Validation set is held out completely and never touched during training or feature selection."),
        ("predict",   "var(--amber-dim)",  "#ffb703", "One-Click Predictions",      "Upload any trained model with a new SMILES CSV and download predictions as Excel in seconds."),
        ("pkg",       "var(--purple-dim)", "#7c6fe0", "Full Export Package",        "Download a ZIP with your model, feature pipeline, datasets, metrics report, and reproducibility README."),
    ]

    grid_html = '<div class="feature-grid">'
    for ic, bg, clr, title, desc in features:
        grid_html += f"""
        <div class="feature-card">
          <div class="feature-icon" style="background:{bg};">{icon(ic, 20, clr)}</div>
          <div class="feature-title">{title}</div>
          <div class="feature-desc">{desc}</div>
        </div>"""
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown("<hr/>", unsafe_allow_html=True)
    section_header("info", "How It Works", "A guided 5-step pipeline — each step unlocks the next")

    pipeline_html = '''<div style="display:flex; align-items:flex-start; gap:0; margin: 2rem 0; overflow-x:auto; padding-bottom:1rem;">'''
    steps_info = [
        ("#10b981", "01", "Upload & Validate",       "CSV with SMILES + Activity. Validate with RDKit."),
        ("#fbbf24", "02", "Preprocess",               "Clean, split (train/test/val), optional SMOTE."),
        ("#8b5cf6", "03", "Feature Engineering",       "Descriptors and apply feature selection."),
        ("#f87171", "04", "Model Benchmarking",        "Train models, tune, compare leaderboard."),
        ("#10b981", "05", "Final Model + Export",      "Train winner, SHAP, ZIP package export."),
    ]
    for i, (color, num, title, desc) in enumerate(steps_info):
        arrow = "" if i == 0 else f'<div style="font-size:1.5rem; color:#64748b; padding: 28px 12px 0 12px; flex-shrink:0;">&#10142;</div>'
        pipeline_html += f'''
{arrow}
<div style="flex:1; min-width:160px; background:var(--card); border:1px solid var(--border); border-top:3px solid {color}; border-radius:var(--radius); padding:1.2rem 1rem;">
<div style="font-size:1.4rem; font-weight:900; color:{color}; opacity:0.6; font-family:'JetBrains Mono',monospace; margin-bottom:0.5rem;">{num}</div>
<div style="font-size:0.82rem; font-weight:700; color:var(--text); margin-bottom:0.4rem;">{title}</div>
<div style="font-size:0.73rem; color:var(--text-muted); line-height:1.5;">{desc}</div>
</div>'''
    
    pipeline_html += "</div>"
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # Quick start
    section_header("book", "Quick Start", "Get up and running in under 5 minutes")
    st.markdown("""
    <div style="background:var(--surface2); border:1px solid var(--border); border-radius:var(--radius); padding:1.5rem 2rem;">
      <ol style="margin:0; padding-left:1.3rem; color:var(--text-muted); font-size:0.88rem; line-height:2.2;">
        <li>Navigate to <strong style="color:var(--teal);">Upload</strong> tab — upload your CSV or use the sample dataset</li>
        <li>Click <strong style="color:var(--text);">Run Validation</strong> to inspect your data quality</li>
        <li>Go to <strong style="color:var(--teal);">Preprocess</strong> — configure splits and click Run</li>
        <li>In <strong style="color:var(--teal);">Features</strong> — choose descriptors and feature selection strategy</li>
        <li>In <strong style="color:var(--teal);">Benchmark</strong> — select models and run comparison</li>
        <li>In <strong style="color:var(--teal);">Final Model</strong> — train, evaluate, and export your ZIP package</li>
        <li>Use <strong style="color:var(--teal);">Predict</strong> tab to run inference on new molecules anytime</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack chips
    st.markdown("<br/>", unsafe_allow_html=True)
    chips = ["RDKit", "scikit-learn", "LightGBM", "XGBoost", "SHAP", "SMOTE", "Plotly", "Streamlit"]
    chip_html = '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;">' + "".join(
        f'<span style="background:var(--surface2);border:1px solid var(--border);border-radius:99px;padding:4px 14px;font-size:0.72rem;font-weight:600;color:var(--text-muted);">{c}</span>'
        for c in chips
    ) + "</div>"
    st.markdown(chip_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    step_info_bar("0", "Upload & Validate", "Upload your SMILES + Activity CSV and inspect data quality")
    step0_upload.render()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Preprocess
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    step_info_bar("1", "Data Preprocessing", "Clean the data, configure your train/test/validation split")
    if not done("step0_complete"):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;background:var(--surface2);
             border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.5rem;margin:1rem 0;">
          {icon("lock", 20, "#7a8fa6")}
          <span style="color:var(--text-muted);font-size:0.88rem;">
            Complete <strong style="color:var(--text);">Step 0 — Upload & Validate</strong> first to unlock preprocessing.
          </span>
        </div>""", unsafe_allow_html=True)
    else:
        step1_preprocessing.render()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Features
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    step_info_bar("2", "Feature Engineering", "Generate molecular descriptors and apply feature selection")
    if not done("step1_complete"):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;background:var(--surface2);
             border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.5rem;margin:1rem 0;">
          {icon("lock", 20, "#7a8fa6")}
          <span style="color:var(--text-muted);font-size:0.88rem;">
            Complete <strong style="color:var(--text);">Step 1 — Preprocessing</strong> first.
          </span>
        </div>""", unsafe_allow_html=True)
    else:
        step2_features.render()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Benchmark
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    step_info_bar("3", "Model Benchmarking", "Train and compare 5 classifiers, select the best")
    if not done("step2_complete"):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;background:var(--surface2);
             border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.5rem;margin:1rem 0;">
          {icon("lock", 20, "#7a8fa6")}
          <span style="color:var(--text-muted);font-size:0.88rem;">
            Complete <strong style="color:var(--text);">Step 2 — Feature Engineering</strong> first.
          </span>
        </div>""", unsafe_allow_html=True)
    else:
        step3_benchmarking.render()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Final Model
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    step_info_bar("4", "Final Model Training", "Train chosen model, evaluate, generate SHAP plots, export ZIP")
    if not done("step3_complete"):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;background:var(--surface2);
             border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.5rem;margin:1rem 0;">
          {icon("lock", 20, "#7a8fa6")}
          <span style="color:var(--text-muted);font-size:0.88rem;">
            Complete <strong style="color:var(--text);">Step 3 — Benchmarking</strong> first.
          </span>
        </div>""", unsafe_allow_html=True)
    else:
        step4_final_model.render()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Predict
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    step_info_bar("—", "Prediction Module", "Upload a saved model and new SMILES data to run inference")
    prediction.render()
