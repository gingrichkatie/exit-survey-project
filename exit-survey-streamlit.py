import os, io, traceback
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import altair as alt

# ---- sklearn shim for old pickles that reference private class
import sklearn.compose._column_transformer as ct
class _RemainderColsList(list): pass
ct._RemainderColsList = _RemainderColsList

BASE_DIR = os.path.dirname(__file__)
REQ = ["best_model.pkl", "preprocess.pkl", "text_vectorizer.pkl", "column_config.pkl"]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Exit Survey Classifier", layout="wide")

# ---------------- GLOBAL STYLES ----------------
st.markdown("""
<style>
:root{
  --ink:#111827;          /* main text */
  --muted:#6B7280;        /* secondary text */
  --line:#E5E7EB;         /* borders */
  --bg:#F8FAFC;           /* page bg */
  --card:#FFFFFF;         /* card bg */
  --accent:#C62828;       /* business red for accents/charts */
  --btn:#E5E7EB;          /* neutral gray button */
  --btn-h:#D1D5DB;        /* hover gray */
}

html, body, [class^="css"]{ background:var(--bg) !important; color:var(--ink); }
section.main > div{ padding-top:0.75rem; padding-bottom:2rem; }

/* Full-width red header bar */
.full-bleed-banner{
  width:100vw; position:relative; left:50%; right:50%;
  margin-left:-50vw; margin-right:-50vw;
  background:var(--accent); border:none;
  margin-bottom: 1.5rem;  /* breathing room before tabs */
}
.full-bleed-inner{
  max-width:1200px; margin:0 auto; padding:16px 24px;
  display:flex; align-items:center; gap:14px;
}
.full-bleed-title{ color:#fff; margin:0; font-weight:700; font-size:1.6rem; letter-spacing:.2px; }
.full-bleed-sub{ color:#FFEAEA; margin:2px 0 0 0; font-size:.95rem; }

/* Panels */
.panel{
  background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; box-shadow:0 1px 0 rgba(0,0,0,0.02);
}
.panel.tight{ padding:10px 12px; }

/* Flat Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:14px; border-bottom:2px solid var(--line); }
.stTabs [role="tab"]{ background:transparent; border:none; padding:8px 2px; margin-bottom:-2px; color:var(--muted); font-weight:600; }
.stTabs [aria-selected="true"]{ color:var(--ink); border-bottom:3px solid var(--ink); }

/* KPI badges */
.kpi{
  display:grid; gap:4px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px;
}
.kpi .label{ color:var(--muted); font-size:.85rem; }
.kpi .value{ font-weight:700; font-size:1.05rem; }

/* Exit options chips — clean, no hollow circle */
.chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
.chip{
  background: var(--btn);
  color: var(--ink);
  border: none;
  border-radius: 6px;
  padding: 6px 12px;
  font-weight: 600;
  font-size: .88rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* Neutral gray buttons (download & primary) */
div[data-testid="stDownloadButton"] button,
div[data-testid="stDownloadButton"] button[kind="primary"],
div[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"],
div[data-testid="stDownloadButton"] button[data-testid="baseButton-secondary"],
.stButton > button,
.stButton > button[kind="primary"]{
  background: var(--btn) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 6px !important;
  font-weight: 600 !important; box-shadow: none !important;
}
div[data-testid="stDownloadButton"] button:hover,
.stButton > button:hover{ background: var(--btn-h) !important; }

/* Small helpers */
.small{ color:var(--muted); font-size:.9rem; }
hr.div{ border:none; height:2px; background:var(--line); margin:12px 0 18px; }
</style>
""", unsafe_allow_html=True)

# ---------------- BANNER ----------------
st.markdown("""
<div class="full-bleed-banner">
  <div class="full-bleed-inner">
    <div>
      <div class="full-bleed-title">Exit Survey Classifier</div>
      <div class="full-bleed-sub">Predict the primary reason for leaving — manual entry or CSV batch.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def _csv_template_bytes(cols) -> bytes:
    tpl = pd.DataFrame([{c: "" for c in cols}])
    buf = io.StringIO(); tpl.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [f for f in REQ if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing files next to this script: {', '.join(missing)}")
    model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    preprocess = joblib.load(os.path.join(BASE_DIR, "preprocess.pkl"))
    text_vectorizer = joblib.load(os.path.join(BASE_DIR, "text_vectorizer.pkl"))
    config = joblib.load(os.path.join(BASE_DIR, "column_config.pkl"))
    return model, preprocess, text_vectorizer, config

def to_matrix(df_in: pd.DataFrame, preprocess, text_vectorizer, text_col):
    Xs = preprocess.transform(df_in)
    if text_col and (text_vectorizer is not None) and (text_col in df_in.columns):
        Xt = text_vectorizer.transform(df_in[text_col].astype(str).fillna(""))
        return hstack([Xs, Xt]).tocsr()
    return Xs

def predict_df(df_in, model, preprocess, text_vectorizer, text_col):
    X = to_matrix(df_in, preprocess, text_vectorizer, text_col)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        preds = model.classes_[np.argmax(proba, axis=1)]
        conf = np.max(proba, axis=1)
        return preds, conf, proba, model.classes_
    preds = model.predict(X)
    return preds, None, None, None

# Consistent red charts
def bar_chart(df, x, y, title=""):
    base = alt.Chart(df).encode(
        x=alt.X(x, sort='-y', axis=alt.Axis(labelColor='#111827', titleColor='#111827', labelLimit=220)),
        y=alt.Y(y, axis=alt.Axis(labelColor='#111827', titleColor='#111827')),
        tooltip=[x, y]
    )
    bars = base.mark_bar(cornerRadius=2, stroke='#111827', strokeWidth=1, opacity=0.95).encode(
        color=alt.value("#C62828")  # consistent business red
    )
    return (bars.properties(title=title, height=300)
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0))

# ---------------- LOAD ARTIFACTS ----------------
try:
    model, preprocess, text_vectorizer, cfg = load_artifacts()
except Exception as e:
    st.error("The app failed to start — model files missing or invalid.")
    st.exception(e); st.code(traceback.format_exc()); st.stop()

NUMERIC_COLS = cfg.get("NUMERIC_COLS", [])
CATEGORICAL_COLS = cfg.get("CATEGORICAL_COLS", [])
TEXT_COL = cfg.get("TEXT_COL", None)
CLASS_ORDER = cfg.get("CLASS_ORDER", None)
CAT_CHOICES = cfg.get("CATEGORICAL_CHOICES", {})
CLASS_LABEL_DESCRIPTIONS = cfg.get("CLASS_LABEL_DESCRIPTIONS", {})  # optional {class: description}
EXPECTED_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if TEXT_COL else [])
INT_COLS = cfg.get("INTEGER_COLS", [])
if "Age" in NUMERIC_COLS and "Age" not in INT_COLS:
    INT_COLS = list(set(INT_COLS + ["Age"]))

# Class list (for exit options)
try:
    CLASS_LIST = list(CLASS_ORDER) if CLASS_ORDER else list(getattr(model, "classes_", []))
except Exception:
    CLASS_LIST = list(getattr(model, "classes_", []))

# ---------------- STATUS ROW ----------------
k1, k2, k3, k4 = st.columns([1,1,1,1])
with k1:
    st.markdown('<div class="kpi"><div class="label">Artifacts</div><div class="value">Loaded</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div class="label">Features</div><div class="value">{len(EXPECTED_COLS)}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi"><div class="label">Classes</div><div class="value">{len(CLASS_LIST) if CLASS_LIST else "–"}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi"><div class="label">Text Feature</div><div class="value">{"Enabled" if TEXT_COL else "–"}</div></div>', unsafe_allow_html=True)
st.markdown('<hr class="div" />', unsafe_allow_html=True)

# ---------------- ABOUT THIS MODEL ----------------
with st.expander("About this model", expanded=False):
    algo = type(model).__name__
    target = cfg.get("TARGET_COL", "Predicted class")
    n_num = len(NUMERIC_COLS)
    n_cat = len(CATEGORICAL_COLS)
    text_feat = "Yes" if TEXT_COL else "No"
    n_classes = len(CLASS_LIST) if CLASS_LIST else 0

    train_start = cfg.get("TRAIN_START") or cfg.get("TRAIN_START_DATE") or cfg.get("train_start")
    train_end   = cfg.get("TRAIN_END")   or cfg.get("TRAIN_END_DATE")   or cfg.get("train_end")
    metrics = cfg.get("VALIDATION_METRICS", {})

    st.markdown(
        f"""
**Algorithm:** {algo}  
**Target:** {target}  
**Classes:** {n_classes if n_classes else "—"}  
**Features:** {n_num} numeric, {n_cat} categorical, Text feature: {text_feat}
""".strip()
    )

    if train_start or train_end:
        st.markdown(f"**Training window:** {train_start or 'unknown'} → {train_end or 'unknown'}")

    if isinstance(metrics, dict) and metrics:
        cols = st.columns(min(4, max(1, len(metrics))))
        i = 0
        for k, v in metrics.items():
            val = f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
            with cols[i % len(cols)]:
                st.metric(k.replace("_", " ").title(), val)
            i += 1

    with st.expander("Feature details", expanded=False):
        st.markdown("**Numeric features**")
        st.code(", ".join(NUMERIC_COLS) if NUMERIC_COLS else "—", language="text")
        st.markdown("**Categorical features**")
        st.code(", ".join(CATEGORICAL_COLS) if CATEGORICAL_COLS else "—", language="text")
        if TEXT_COL:
            st.markdown("**Text feature**")
            st.code(TEXT_COL, language="text")

    notes = cfg.get("NOTES") or cfg.get("MODEL_NOTES")
    if notes:
        st.markdown("**Notes**")
        st.markdown(notes)

# ---------------- TABS ----------------
tab_manual, tab_csv, tab_insights = st.tabs(["Manual Prediction", "CSV Upload", "Insights"])

# Session history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])

# =============== MANUAL PREDICTION =================
with tab_manual:
    st.subheader("Manual Prediction")

    left, right = st.columns([2,1], gap="large")

    # Left: form + results
    with left:
        st.markdown("<div class='panel small'>Complete the form and generate a prediction. All required dropdowns must be selected to enable the button.</div>", unsafe_allow_html=True)
        vals = {}
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Numeric Inputs**")
            for col in NUMERIC_COLS:
                if col in INT_COLS:
                    vals[col] = st.number_input(col, value=0, step=1, format="%d")
                else:
                    vals[col] = st.number_input(col, value=0.00, step=0.01, format="%.2f")
        with cB:
            st.markdown("**Categorical Inputs**")
            for col in CATEGORICAL_COLS:
                choices = [str(x) for x in CAT_CHOICES.get(col, [])]
                if "Other" in choices:
                    choices = [c for c in choices if c != "Other"] + ["Other"]
                vals[col] = st.selectbox(col, options=choices, index=None, placeholder=f"Select {col}") if choices else st.text_input(col, value="")

        if TEXT_COL:
            st.markdown("**Optional Text**")
            vals[TEXT_COL] = st.text_area(TEXT_COL, value="", height=110, placeholder="Short note (optional)")

        ready = all(v not in (None, "") for k, v in vals.items() if k in CATEGORICAL_COLS)
        c_btn, _ = st.columns([1,4])
        with c_btn:
            predict_clicked = st.button("Predict", use_container_width=True, disabled=not ready)

        if not ready:
            st.caption("Select all dropdown values to enable prediction.")

        if predict_clicked:
            row = pd.DataFrame([vals])
            row = row[EXPECTED_COLS] if EXPECTED_COLS else row
            for c in CATEGORICAL_COLS:
                row[c] = row[c].astype(str).str.strip()
            for c in NUMERIC_COLS:
                if c in INT_COLS:
                    row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0).astype(int)
                else:
                    row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0.0)

            preds, conf, proba, classes_ = predict_df(row, model, preprocess, text_vectorizer, TEXT_COL)

            st.markdown('<br/>', unsafe_allow_html=True)
            rc1, rc2 = st.columns([1.2, 2])
            with rc1:
                if conf is not None:
                    st.markdown(
                        f"""
                        <div class="panel">
                          <div style="font-weight:700; color:var(--accent);">Prediction</div>
                          <h3 style="margin:6px 0 4px 0;">{preds[0]}</h3>
                          <div class="small">Confidence: <strong>{conf[0]:.0%}</strong></div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="panel">
                          <div style="font-weight:700; color:var(--accent);">Prediction</div>
                          <h3 style="margin:6px 0 4px 0;">{preds[0]}</h3>
                          <div class="small">Model does not expose probabilities</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
            with rc2:
                if proba is not None:
                    p = pd.DataFrame({"class": classes_, "prob": proba.flatten()}).sort_values("prob", ascending=False)
                    st.altair_chart(bar_chart(p, "class:N", "prob:Q", title="Class Probabilities"), use_container_width=True)

            # Update history
            hist_row = row.copy()
            hist_row["prediction"] = preds[0]
            hist_row["confidence"] = None if conf is None else float(conf[0])
            st.session_state.history = pd.concat([st.session_state.history, hist_row], ignore_index=True)

    # Right: Exit Options (chips + dropdown with description)
    with right:
        # Clean section header (no icon/circle)
        st.markdown("<h4 style='margin:0 0 6px 0;'>Exit Options</h4>", unsafe_allow_html=True)
        st.markdown("<div class='panel tight'>", unsafe_allow_html=True)

        if CLASS_LIST:
            # Chips view (flat, borderless—no empty circles)
            st.markdown('<div class="chips">', unsafe_allow_html=True)
            for c in CLASS_LIST:
                st.markdown(f'<div class="chip">{c}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Focused browser for descriptions (no extra circle or hr above)
            sel = st.selectbox("Browse option details", options=CLASS_LIST, index=0 if CLASS_LIST else None)
            desc = CLASS_LABEL_DESCRIPTIONS.get(sel, "") if isinstance(CLASS_LABEL_DESCRIPTIONS, dict) else ""
            if desc:
                st.markdown(f"<div class='small' style='margin-top:6px;'>{desc}</div>", unsafe_allow_html=True)
            else:
                st.caption("No description available for this option.")
        else:
            st.caption("No class list available from model/config.")
        st.markdown("</div>", unsafe_allow_html=True)

# =============== CSV UPLOAD =================
with tab_csv:
    st.subheader("CSV Upload")
    st.markdown("<div class='panel small'>Batch predictions. Start with the template to match the expected columns.</div>", unsafe_allow_html=True)

    st.download_button(
        "Download CSV Template",
        data=_csv_template_bytes(EXPECTED_COLS),
        file_name="exit_survey_template.csv",
        mime="text/csv",
        type="secondary",  # neutral; CSS forces gray
        help="Includes all required columns in order."
    )

    if EXPECTED_COLS:
        preview = pd.DataFrame(columns=EXPECTED_COLS).head(3)
        st.caption("Template preview (columns only):")
        st.dataframe(preview, use_container_width=True, height=120)

    f = st.file_uploader("Upload CSV matching the expected schema", type=["csv"])
    if f:
        try:
            data = pd.read_csv(f)
        except Exception as e:
            st.error("Could not read your CSV."); st.exception(e); st.stop()

        missing = [c for c in EXPECTED_COLS if c not in data.columns]
        extra = [c for c in data.columns if c not in EXPECTED_COLS]
        if missing:
            st.error("Missing columns: " + ", ".join(missing)); st.stop()
        if extra:
            st.info("Note: Extra columns ignored — " + ", ".join(extra))

        data = data[EXPECTED_COLS]
        for c in CATEGORICAL_COLS:
            data[c] = data[c].astype(str).str.strip()
        for c in NUMERIC_COLS:
            if c in INT_COLS:
                data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0).astype(int)
            else:
                data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)

        preds, conf, proba, classes_ = predict_df(data, model, preprocess, text_vectorizer, TEXT_COL)

        out = data.copy()
        out["prediction"] = preds
        if conf is not None:
            out["confidence"] = conf
        st.markdown("**Results**")
        st.dataframe(out, use_container_width=True)

        # Distribution (red)
        dist = pd.Series(preds).value_counts().reset_index()
        dist.columns = ["class", "count"]
        st.subheader("Predicted Class Distribution")
        st.altair_chart(bar_chart(dist, "class:N", "count:Q"), use_container_width=True)

        # Confidence histogram (red bars too)
        if conf is not None:
            bins = np.linspace(0, 1, 11)
            hist, edges = np.histogram(conf, bins=bins)
            conf_df = pd.DataFrame({
                "bucket": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))],
                "count": hist
            })
            st.subheader("Confidence Histogram")
            st.altair_chart(bar_chart(conf_df, "bucket:N", "count:Q"), use_container_width=True)

# =============== INSIGHTS =================
with tab_insights:
    st.subheader("Recent Predictions")
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history.tail(15), use_container_width=True)
        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])
                st.success("History cleared.")
        with c2:
            counts = st.session_state.history["prediction"].value_counts().reset_index()
            counts.columns = ["class","count"]
            if not counts.empty:
                st.altair_chart(bar_chart(counts, "class:N", "count:Q", title="History — Class Counts"), use_container_width=True)
    else:
        st.caption("No predictions yet. Use Manual Prediction or CSV Upload.")



