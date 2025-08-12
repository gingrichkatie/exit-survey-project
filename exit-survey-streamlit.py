# -*- coding: utf-8 -*-
# Exit Survey Classifier — Pro+ Theme (with Footer)
# Business-forward UI with banner, class options panel, blue buttons, and slim footer.

import os, io, traceback
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import altair as alt

# ===== Footer metadata (edit these) =====
AUTHOR_NAME  = "Kathleen Gingrich"
COURSE_INFO  = "CIS 9660"  
# ========================================

# ---- sklearn shim for old pickles that reference private class
import sklearn.compose._column_transformer as ct
class _RemainderColsList(list): pass
ct._RemainderColsList = _RemainderColsList

BASE_DIR = os.path.dirname(__file__)
REQ = ["best_model.pkl", "preprocess.pkl", "text_vectorizer.pkl", "column_config.pkl"]

# ───────────────────────────
# Page config
# ───────────────────────────
st.set_page_config(
    page_title="Exit Survey Classifier",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────
# Professional theme (navy / blue / slate)
# ───────────────────────────
st.markdown("""
<style>
:root{
  --brand:#0B3A75;        /* navy */
  --brand-2:#1F6FEB;      /* blue */
  --ink:#111827;          /* text */
  --muted:#6B7280;        /* secondary text */
  --line:#D1D5DB;         /* border */
  --bg:#F8FAFC;           /* page bg */
  --card:#FFFFFF;         /* card bg */
}

html, body, [class^="css"]{ background:var(--bg) !important; color:var(--ink); }
section.main > div{ padding-top:1rem; padding-bottom:2.25rem; }

/* Banner */
.pro-banner{
  display:grid; grid-template-columns:auto 1fr; gap:14px; align-items:center;
  background:linear-gradient(90deg, rgba(11,58,117,.05), rgba(31,111,235,.05));
  border:2px solid var(--brand); border-radius:10px; padding:14px 16px;
}
.pro-banner .title{ font-size:1.45rem; font-weight:700; letter-spacing:.2px; }
.pro-banner .subtitle{ color:var(--muted); margin-top:4px; }

/* KPI cards */
.kpi{
  background:var(--card); border:2px solid var(--line); border-radius:10px;
  padding:12px 14px; display:grid; gap:4px;
}
.kpi .label{ color:var(--muted); font-size:.85rem; }
.kpi .value{ font-weight:700; font-size:1.05rem; }

/* Panels */
.panel{
  background:var(--card); border:2px solid var(--line); border-radius:10px; padding:14px 16px;
}
.panel.tight{ padding:10px 12px; }

/* FLAT TABS */
.stTabs [data-baseweb="tab-list"]{ gap:18px; border-bottom:2px solid var(--line); }
.stTabs [role="tab"]{
  background:transparent; border:none; padding:8px 2px; margin-bottom:-2px;
  color:var(--muted); font-weight:600;
}
.stTabs [aria-selected="true"]{
  color:var(--brand); border-bottom:3px solid var(--brand);
}

/* Chips for classes */
.chips{ display:flex; flex-wrap:wrap; gap:8px; }
.chip{
  border:2px solid var(--line); border-radius:999px; padding:4px 10px; background:#fff;
  font-weight:600; color:var(--ink); font-size:.88rem;
}
.chip.note{ color:var(--muted); font-weight:500; border-style:dashed; }

/* Download button style — neutral gray */
div[data-testid="stDownloadButton"] > button{
  background: #E5E7EB !important;  /* light gray */
  color: #111827 !important;        /* near-black text */
  border: 1px solid #D1D5DB !important; 
  border-radius: 6px !important;
  font-weight: 600 !important;
}
div[data-testid="stDownloadButton"] > button:hover{
  background: #D1D5DB !important;   /* slightly darker on hover */
}

/* Primary buttons */
.stButton > button[kind="primary"], .stButton > button{
  background:#0B3A75; color:#fff; border:2px solid #082B56; border-radius:10px;
  font-weight:700;
}
.stButton > button:hover{ filter:brightness(1.03); }

/* Footer */
.footer{
  text-align:center; color:var(--muted); font-size:.85rem; margin-top:22px;
  padding-top:10px; border-top:2px solid var(--line);
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────
# Optional logo (logo.png/.jpg/.jpeg alongside the script)
# ───────────────────────────
logo_path = None
for candidate in ("logo.png","logo.jpg","logo.jpeg"):
    p = os.path.join(BASE_DIR, candidate)
    if os.path.exists(p):
        logo_path = p; break

# Banner
c_logo, c_text = st.columns([1,5])
with c_logo:
    if logo_path:
        st.image(logo_path, width=56)
with c_text:
    st.markdown(f"""
<div class="pro-banner">
  <div style="width:14px;height:14px;border-radius:2px;background:var(--brand);"></div>
  <div>
    <div class="title">Exit Survey Classifier</div>
    <div class="subtitle">Predict the primary reason for leaving from survey inputs — manual entry or CSV batch.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr style="border:none;height:2px;background:var(--line);margin:12px 0 18px;" />', unsafe_allow_html=True)

# ───────────────────────────
# Helpers
# ───────────────────────────
def _csv_template_bytes(cols) -> bytes:
    tpl = pd.DataFrame([{c: "" for c in cols}])
    buf = io.StringIO(); tpl.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [f for f in REQ if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing files next to this script: {', '.join(missing)}")
    try:
        model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
        preprocess = joblib.load(os.path.join(BASE_DIR, "preprocess.pkl"))
        text_vectorizer = joblib.load(os.path.join(BASE_DIR, "text_vectorizer.pkl"))
        config = joblib.load(os.path.join(BASE_DIR, "column_config.pkl"))
        return model, preprocess, text_vectorizer, config
    except Exception as e:
        raise RuntimeError("Failed while loading model artifacts.") from e

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

def bar_chart_pro(df, x, y, title=""):
    base = alt.Chart(df).encode(
        x=alt.X(x, sort='-y', axis=alt.Axis(labelColor='#111827', titleColor='#111827', labelLimit=220)),
        y=alt.Y(y, axis=alt.Axis(labelColor='#111827', titleColor='#111827')),
        tooltip=[x, y]
    )
    bars = base.mark_bar(cornerRadius=2, stroke='#111827', strokeWidth=1, opacity=0.95).encode(
        color=alt.value("#1F6FEB")
    )
    return bars.properties(height=260, title=title).configure_title(color='#111827')

# ───────────────────────────
# Load artifacts
# ───────────────────────────
try:
    model, preprocess, text_vectorizer, cfg = load_artifacts()
except Exception as e:
    st.error("The app failed to start because model files could not be loaded.")
    with st.expander("Error details"):
        st.exception(e)
        st.code(traceback.format_exc())
    st.stop()

NUMERIC_COLS = cfg.get("NUMERIC_COLS", [])
CATEGORICAL_COLS = cfg.get("CATEGORICAL_COLS", [])
TEXT_COL = cfg.get("TEXT_COL", None)
CLASS_ORDER = cfg.get("CLASS_ORDER", None)
CAT_CHOICES = cfg.get("CATEGORICAL_CHOICES", {})
CLASS_LABEL_DESCRIPTIONS = cfg.get("CLASS_LABEL_DESCRIPTIONS", {})  # optional mapping
EXPECTED_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if TEXT_COL else [])
INT_COLS = cfg.get("INTEGER_COLS", [])
if "Age" in NUMERIC_COLS and "Age" not in INT_COLS:
    INT_COLS = list(set(INT_COLS + ["Age"]))

# Derive class list
try:
    CLASS_LIST = list(CLASS_ORDER) if CLASS_ORDER else list(getattr(model, "classes_", []))
except Exception:
    CLASS_LIST = list(getattr(model, "classes_", []))

# ───────────────────────────
# Status row
# ───────────────────────────
k1, k2, k3, k4 = st.columns([1,1,1,1])
with k1:
    st.markdown('<div class="kpi"><div class="label">Artifacts</div><div class="value">Loaded</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div class="label">Features</div><div class="value">{len(EXPECTED_COLS)}</div></div>', unsafe_allow_html=True)
with k3:
    n_classes = len(CLASS_LIST) if CLASS_LIST else "–"
    st.markdown(f'<div class="kpi"><div class="label">Classes</div><div class="value">{n_classes}</div></div>', unsafe_allow_html=True)
with k4:
    txt = "Enabled" if TEXT_COL else "–"
    st.markdown(f'<div class="kpi"><div class="label">Text Feature</div><div class="value">{txt}</div></div>', unsafe_allow_html=True)

st.markdown('<hr style="border:none;height:2px;background:var(--line);margin:12px 0 18px;" />', unsafe_allow_html=True)

# ───────────────────────────
# Tabs
# ───────────────────────────
tab_manual, tab_csv, tab_insights = st.tabs(["Manual Prediction", "CSV Upload", "Insights"])

# Session history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])

# =========================================================
# Manual Prediction
# =========================================================
with tab_manual:
    st.subheader("Manual Prediction")

    left, right = st.columns([2,1])

    with left:
        st.markdown("<div class='panel' style='margin-bottom:10px;'>Complete the form and generate a prediction. All required dropdowns must be selected to enable the button.</div>", unsafe_allow_html=True)
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
                if choices:
                    vals[col] = st.selectbox(col, options=choices, index=None, placeholder=f"Select {col}")
                else:
                    vals[col] = st.text_input(col, value="")

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

            # normalize types
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
                          <div class="badge">Prediction</div>
                          <h3 style="margin:6px 0 4px 0;">{preds[0]}</h3>
                          <div class="small">Confidence: <strong>{conf[0]:.0%}</strong></div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="panel">
                          <div class="badge">Prediction</div>
                          <h3 style="margin:6px 0 4px 0;">{preds[0]}</h3>
                          <div class="small">Model does not expose probabilities</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
            with rc2:
                if proba is not None:
                    p = pd.DataFrame({"class": classes_, "prob": proba.flatten()}).sort_values("prob", ascending=False)
                    st.altair_chart(
                        bar_chart_pro(p, "class:N", "prob:Q", title="Class Probabilities"),
                        use_container_width=True
                    )

            # Update history
            hist_row = row.copy()
            hist_row["prediction"] = preds[0]
            hist_row["confidence"] = None if conf is None else float(conf[0])
            st.session_state.history = pd.concat([st.session_state.history, hist_row], ignore_index=True)

    with right:
        st.markdown("**Options for Exiting**")
        st.markdown("<div class='panel tight'>", unsafe_allow_html=True)
        if CLASS_LIST:
            st.markdown('<div class="chips">', unsafe_allow_html=True)
            for c in CLASS_LIST:
                st.markdown(f'<div class="chip">{c}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Optional descriptions
            if isinstance(CLASS_LABEL_DESCRIPTIONS, dict) and len(CLASS_LABEL_DESCRIPTIONS) > 0:
                st.markdown('<div style="margin-top:8px;"><strong>Definitions</strong></div>', unsafe_allow_html=True)
                for cls in CLASS_LIST:
                    desc = CLASS_LABEL_DESCRIPTIONS.get(cls, "")
                    if desc:
                        st.markdown(f"- **{cls}** — {desc}")
                    else:
                        st.markdown(f"- **{cls}** — <span class='chip note'>No description provided</span>", unsafe_allow_html=True)
        else:
            st.caption("No class list available from model/config.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CSV Upload
# =========================================================
with tab_csv:
    st.subheader("CSV Upload")
    st.markdown("<div class='panel' style='margin-bottom:10px;'>Batch predictions. Start with the template to match the expected schema.</div>", unsafe_allow_html=True)

    st.download_button(
        "Download CSV Template",
        data=_csv_template_bytes(EXPECTED_COLS),
        file_name="exit_survey_template.csv",
        mime="text/csv",
        type="primary",
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
            st.error("Could not read your CSV.")
            with st.expander("Details"):
                st.exception(e)
            st.stop()

        # Schema check
        missing = [c for c in EXPECTED_COLS if c not in data.columns]
        extra = [c for c in data.columns if c not in EXPECTED_COLS]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
            st.stop()
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

        # Predicted class distribution
        dist = pd.Series(preds).value_counts().reset_index()
        dist.columns = ["class", "count"]
        st.markdown("**Predicted Class Distribution**")
        st.altair_chart(bar_chart_pro(dist, "class:N", "count:Q"), use_container_width=True)

        # Confidence histogram (if available)
        if conf is not None:
            bins = np.linspace(0, 1, 11)
            hist, edges = np.histogram(conf, bins=bins)
            conf_df = pd.DataFrame({
                "bucket":[f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))],
                "count":hist
            })
            st.markdown("**Confidence Histogram**")
            st.altair_chart(bar_chart_pro(conf_df, "bucket:N", "count:Q"), use_container_width=True)

# =========================================================
# Insights
# =========================================================
with tab_insights:
    st.subheader("Insights")
    if not st.session_state.history.empty:
        st.markdown("<span class='badge'>Newest entries appear last</span>", unsafe_allow_html=True)
        st.dataframe(st.session_state.history.tail(15), use_container_width=True)

        s1, s2 = st.columns([1,3])
        with s1:
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])
                st.success("History cleared.")
        with s2:
            counts = st.session_state.history["prediction"].value_counts().reset_index()
            counts.columns = ["class","count"]
            if not counts.empty:
                st.altair_chart(
                    bar_chart_pro(counts, "class:N", "count:Q", title="History — Class Counts"),
                    use_container_width=True
                )
    else:
        st.caption("No predictions yet. Use Manual Prediction or CSV Upload.")

# ───────────────────────────
# Footer (slim, professional)
# ───────────────────────────
last_updated = datetime.now().strftime("%b %d, %Y")
st.markdown(
    f"""
<div class="footer">
  <strong>Kathleen</strong> · CIS 9660 · Last updated {last_updated}
</div>
""",
    unsafe_allow_html=True
)


