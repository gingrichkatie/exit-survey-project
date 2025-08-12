# -*- coding: utf-8 -*-
# Exit Survey Classifier — Business Theme 🧭
# Polished, professional UI with subtle emojis, banner, and optional logo.

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

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exit Survey Classifier",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Clean Business Theme (light)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --accent: #1F6FEB;       /* primary blue */
  --accent-2: #0EA5E9;     /* cyan accent */
  --ink: #111827;          /* text */
  --muted: #6B7280;        /* secondary text */
  --line: #E5E7EB;         /* borders */
  --card: #FFFFFF;         /* cards */
  --bg: #F8FAFC;           /* page bg */
}

html, body, [class^="css"]  { background: var(--bg) !important; color: var(--ink); }

section.main > div { padding-top: 1.25rem; padding-bottom: 2rem; }

/* Banner */
.banner {
  background: linear-gradient(90deg, rgba(31,111,235,0.08), rgba(14,165,233,0.08));
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px 20px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 16px;
  align-items: center;
}
.banner .title { font-size: 1.6rem; font-weight: 700; letter-spacing: .2px; }
.banner .subtitle { color: var(--muted); margin-top: 4px; }

/* KPI cards */
.kpi {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 14px;
  display: grid; gap: 4px;
}
.kpi .label { color: var(--muted); font-size: 0.82rem; }
.kpi .value { font-weight: 700; font-size: 1.1rem; }

/* Panels */
.panel {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; flex-wrap: wrap; }
.stTabs [role="tab"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-bottom: 2px solid transparent;
  border-radius: 12px 12px 0 0;
  color: var(--ink);
}
.stTabs [aria-selected="true"] {
  border-color: var(--accent);
}

/* Badges & helpers */
.badge {
  display:inline-block; padding: 2px 10px;
  border-radius: 999px; border: 1px solid var(--line);
  background: #F3F4F6; color: var(--ink);
  font-weight: 600; font-size: 0.78rem; letter-spacing: 0.2px;
}
.small { color: var(--muted); font-size: 0.9rem; }
hr.divider { border: none; height: 1px; background: var(--line); margin: 10px 0 18px; }
</style>
""", unsafe_allow_html=True)

# Optional logo if present next to script (logo.png or logo.jpg)
logo_path = None
for candidate in ("logo.png", "logo.jpg", "logo.jpeg"):
    p = os.path.join(BASE_DIR, candidate)
    if os.path.exists(p):
        logo_path = p
        break

# Banner
col_logo, col_text = st.columns([1,5])
with col_logo:
    if logo_path:
        st.image(logo_path, width=72)
    else:
        st.markdown("### 💼")
with col_text:
    st.markdown("""
<div class="banner">
  <div>📊</div>
  <div>
    <div class="title">Exit Survey Classifier</div>
    <div class="subtitle">Predict the primary reason for leaving from survey inputs — manually or in bulk via CSV.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider" />', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
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

def bar_chart_business(df, x, y, title=""):
    base = alt.Chart(df).encode(
        x=alt.X(x, sort='-y', axis=alt.Axis(labelColor='#111827', titleColor='#111827', labelLimit=200)),
        y=alt.Y(y, axis=alt.Axis(labelColor='#111827', titleColor='#111827')),
        tooltip=[x, y]
    )
    bars = base.mark_bar(cornerRadius=4, stroke='#ffffff', strokeWidth=0.2, opacity=0.95).encode(
        color=alt.value("#1F6FEB")
    )
    return bars.properties(height=280, title=title).configure_title(color='#111827')

# ─────────────────────────────────────────────────────────────
# Load Model Artifacts
# ─────────────────────────────────────────────────────────────
try:
    model, preprocess, text_vectorizer, cfg = load_artifacts()
except Exception as e:
    st.error("⚠️ The app failed to start because model files could not be loaded.")
    with st.expander("Show error details"):
        st.exception(e)
        st.code(traceback.format_exc())
    st.stop()

NUMERIC_COLS = cfg.get("NUMERIC_COLS", [])
CATEGORICAL_COLS = cfg.get("CATEGORICAL_COLS", [])
TEXT_COL = cfg.get("TEXT_COL", None)
CLASS_ORDER = cfg.get("CLASS_ORDER", None)
CAT_CHOICES = cfg.get("CATEGORICAL_CHOICES", {})
EXPECTED_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if TEXT_COL else [])
INT_COLS = cfg.get("INTEGER_COLS", [])
if "Age" in NUMERIC_COLS and "Age" not in INT_COLS:
    INT_COLS = list(set(INT_COLS + ["Age"]))

# ─────────────────────────────────────────────────────────────
# Status Row (minimal + business)
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1,1,1,1.2])
with c1:
    st.markdown('<div class="kpi"><div class="label">📦 Artifacts</div><div class="value">Loaded</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><div class="label">🧮 Features</div><div class="value">{len(EXPECTED_COLS)} cols</div></div>', unsafe_allow_html=True)
with c3:
    n_classes = len(CLASS_ORDER) if CLASS_ORDER else "—"
    st.markdown(f'<div class="kpi"><div class="label">🏷️ Classes</div><div class="value">{n_classes}</div></div>', unsafe_allow_html=True)
with c4:
    txt = "Enabled" if TEXT_COL else "—"
    st.markdown(f'<div class="kpi"><div class="label">💬 Text Feature</div><div class="value">{txt}</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider" />', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_manual, tab_csv, tab_insights = st.tabs(["✨ Manual Prediction", "📁 CSV Upload", "🔎 Insights"])

# Session History
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])

# =========================================================
# Manual Prediction
# =========================================================
with tab_manual:
    st.subheader("✨ Manual Prediction")
    st.markdown("<div class='panel small'>Complete the form and get a prediction. Select all dropdowns to enable the button.</div>", unsafe_allow_html=True)

    vals = {}
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Numeric Inputs**")
        for col in NUMERIC_COLS:
            if col in INT_COLS:
                vals[col] = st.number_input(f"{col}", value=0, step=1, format="%d")
            else:
                vals[col] = st.number_input(f"{col}", value=0.00, step=0.01, format="%.2f")

    with col2:
        st.markdown("**Categorical Inputs**")
        for col in CATEGORICAL_COLS:
            choices = [str(x) for x in CAT_CHOICES.get(col, [])]
            if "Other" in choices:
                choices = [c for c in choices if c != "Other"] + ["Other"]
            if choices:
                vals[col] = st.selectbox(f"{col}", options=choices, index=None, placeholder=f"Select {col}")
            else:
                vals[col] = st.text_input(f"{col}", value="")

    if TEXT_COL:
        st.markdown("**Optional Text**")
        vals[TEXT_COL] = st.text_area(TEXT_COL, value="", height=110, placeholder="Short note (optional)")

    ready = all(v not in (None, "") for k, v in vals.items() if k in CATEGORICAL_COLS)
    cA, cB = st.columns([1,4])
    with cA:
        predict_clicked = st.button("🔮 Predict", use_container_width=True, type="primary", disabled=not ready)
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

        # Result + Probability
        st.markdown('<br/>', unsafe_allow_html=True)
        rc1, rc2 = st.columns([1.2, 2])
        with rc1:
            if conf is not None:
                st.markdown(
                    f"""
                    <div class="panel">
                      <div class="badge">Prediction</div>
                      <h3 style="margin:6px 0 4px 0;">🎯 {preds[0]}</h3>
                      <div class="small">Confidence: <strong>{conf[0]:.0%}</strong></div>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="panel">
                      <div class="badge">Prediction</div>
                      <h3 style="margin:6px 0 4px 0;">🎯 {preds[0]}</h3>
                      <div class="small">Model does not expose probabilities</div>
                    </div>
                    """, unsafe_allow_html=True
                )
        with rc2:
            if proba is not None:
                p = pd.DataFrame({"class": classes_, "prob": proba.flatten()}).sort_values("prob", ascending=False)
                st.altair_chart(
                    bar_chart_business(p, "class:N", "prob:Q", title="Class Probabilities"),
                    use_container_width=True
                )

        # Update history
        hist_row = row.copy()
        hist_row["prediction"] = preds[0]
        hist_row["confidence"] = None if conf is None else float(conf[0])
        st.session_state.history = pd.concat([st.session_state.history, hist_row], ignore_index=True)

# =========================================================
# CSV Upload
# =========================================================
with tab_csv:
    st.subheader("📁 CSV Upload")
    st.markdown("<div class='panel small'>Upload multiple rows. Start from the template to match expected columns.</div>", unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download CSV Template",
        data=_csv_template_bytes(EXPECTED_COLS),
        file_name="exit_survey_template.csv",
        mime="text/csv",
        type="primary",
        help="Includes all required columns in order."
    )

    f = st.file_uploader("Upload CSV matching the expected schema", type=["csv"])

    if f:
        try:
            data = pd.read_csv(f)
        except Exception as e:
            st.error("❌ Could not read your CSV.")
            with st.expander("Details"):
                st.exception(e)
            st.stop()

        missing = [c for c in EXPECTED_COLS if c not in data.columns]
        if missing:
            st.error("🚫 Missing columns: " + ", ".join(missing))
            st.stop()

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

        # Distribution
        dist = pd.Series(preds).value_counts().reset_index()
        dist.columns = ["class", "count"]
        st.markdown("**Predicted Class Distribution**")
        st.altair_chart(bar_chart_business(dist, "class:N", "count:Q"), use_container_width=True)

        # Confidence histogram
        if conf is not None:
            bins = np.linspace(0, 1, 11)
            hist, edges = np.histogram(conf, bins=bins)
            conf_df = pd.DataFrame({
                "bucket": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))],
                "count": hist
            })
            st.markdown("**Confidence Histogram**")
            st.altair_chart(bar_chart_business(conf_df, "bucket:N", "count:Q"), use_container_width=True)

# =========================================================
# Insights / Recent predictions
# =========================================================
with tab_insights:
    st.subheader("🔎 Insights")
    if not st.session_state.history.empty:
        st.markdown("<span class='badge'>Newest is last</span>", unsafe_allow_html=True)
        st.dataframe(st.session_state.history.tail(15), use_container_width=True)

        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("🧹 Clear history", use_container_width=True):
                st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])
                st.toast("History cleared.", icon="✅")
        with c2:
            pred_counts = st.session_state.history["prediction"].value_counts().reset_index()
            pred_counts.columns = ["class", "count"]
            if not pred_counts.empty:
                st.altair_chart(
                    bar_chart_business(pred_counts, "class:N", "count:Q", title="History — Class Counts"),
                    use_container_width=True
                )
    else:
        st.caption("No predictions yet. Try **Manual Prediction** or **CSV Upload** above.")



