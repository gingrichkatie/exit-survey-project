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

# ---------------- STYLES ----------------
st.markdown("""
<style>
/* Full-width red header bar */
.full-bleed-banner{
  width:100vw;
  position:relative;
  left:50%; right:50%;
  margin-left:-50vw; margin-right:-50vw;
  background:#C62828;
  border:none;
}
.full-bleed-inner{
  max-width:1200px;
  margin:0 auto;
  padding:16px 24px;
  display:flex; align-items:center; gap:14px;
}
.full-bleed-title{
  color:#fff; margin:0;
  font-weight:700; font-size:1.6rem; letter-spacing:.2px;
}
.full-bleed-sub{
  color:#FFEAEA; margin:2px 0 0 0; font-size:.95rem;
}

/* Tabs: clean style */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #E5E7EB; }
.stTabs [data-baseweb="tab"] { background: #f9fafb; border-radius: 6px 6px 0 0; padding: 6px 12px; }
.stTabs [aria-selected="true"] { background: white; border: 1px solid #E5E7EB; border-bottom: none; }

/* Panels */
.panel {
  background: #fff; border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px 16px; margin: 6px 0 12px 0;
}

/* CSV Download Button - Neutral Gray */
div[data-testid="stDownloadButton"] button{
  background: #E5E7EB !important;
  color: #111827 !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}
div[data-testid="stDownloadButton"] button:hover{
  background: #D1D5DB !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- BANNER ----------------
st.markdown(f"""
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

@st.cache_resource
def load_artifacts():
    missing = [f for f in REQ if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing files: {', '.join(missing)}")
    model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    preprocess = joblib.load(os.path.join(BASE_DIR, "preprocess.pkl"))
    text_vectorizer = joblib.load(os.path.join(BASE_DIR, "text_vectorizer.pkl"))
    config = joblib.load(os.path.join(BASE_DIR, "column_config.pkl"))
    return model, preprocess, text_vectorizer, config

def to_matrix(df_in, preprocess, text_vectorizer, text_col):
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
        color=alt.value("#C62828")  # business red
    )
    return (bars.properties(title=title, height=300)
                 .configure_axis(grid=False)
                 .configure_view(strokeWidth=0))

# ---------------- LOAD ARTIFACTS ----------------
try:
    model, preprocess, text_vectorizer, cfg = load_artifacts()
except Exception as e:
    st.error("The app failed to start — model files missing or invalid.")
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
    INT_COLS.append("Age")

# ---------------- TABS ----------------
tab_manual, tab_csv, tab_insights = st.tabs(["Manual Prediction", "CSV Upload", "Insights"])
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])

# MANUAL PREDICTION
with tab_manual:
    st.header("Manual Prediction")
    st.markdown("<div class='panel'>Fill out the form and predict the primary reason for leaving.</div>", unsafe_allow_html=True)
    vals = {}
    col1, col2 = st.columns(2)
    with col1:
        for col in NUMERIC_COLS:
            if col in INT_COLS:
                vals[col] = st.number_input(col, value=0, step=1, format="%d")
            else:
                vals[col] = st.number_input(col, value=0.00, step=0.01, format="%.2f")
    with col2:
        for col in CATEGORICAL_COLS:
            choices = [str(x) for x in CAT_CHOICES.get(col, [])]
            if "Other" in choices:
                choices = [c for c in choices if c != "Other"] + ["Other"]
            vals[col] = st.selectbox(col, options=choices, index=None, placeholder=f"Select {col}") if choices else st.text_input(col, value="")
    if TEXT_COL:
        vals[TEXT_COL] = st.text_area(TEXT_COL, value="", height=100, placeholder="Optional notes")
    ready = all(v not in (None, "") for k, v in vals.items() if k in CATEGORICAL_COLS)
    if st.button("Predict", disabled=not ready):
        row = pd.DataFrame([vals])[EXPECTED_COLS]
        for c in CATEGORICAL_COLS:
            row[c] = row[c].astype(str).str.strip()
        for c in NUMERIC_COLS:
            if c in INT_COLS:
                row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0).astype(int)
            else:
                row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0.0)
        preds, conf, proba, classes_ = predict_df(row, model, preprocess, text_vectorizer, TEXT_COL)
        st.success(f"Prediction: **{preds[0]}**" + (f" — Confidence: **{conf[0]:.0%}**" if conf is not None else ""))
        if proba is not None:
            p = pd.DataFrame({"class": classes_, "prob": proba.flatten()}).sort_values("prob", ascending=False)
            st.altair_chart(bar_chart_pro(p, "class:N", "prob:Q"), use_container_width=True)
        hist_row = row.copy()
        hist_row["prediction"] = preds[0]
        hist_row["confidence"] = None if conf is None else float(conf[0])
        st.session_state.history = pd.concat([st.session_state.history, hist_row], ignore_index=True)

# CSV UPLOAD
with tab_csv:
    st.header("CSV Upload")
    st.markdown("<div class='panel'>Upload multiple rows at once. Use the template for correct columns.</div>", unsafe_allow_html=True)
    st.download_button(
        "Download CSV Template",
        data=_csv_template_bytes(EXPECTED_COLS),
        file_name="exit_survey_template.csv",
        mime="text/csv"
    )
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f:
        data = pd.read_csv(f)
        missing = [c for c in EXPECTED_COLS if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
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
            st.dataframe(out, use_container_width=True)
            dist = pd.Series(preds).value_counts().reset_index()
            dist.columns = ["class", "count"]
            st.subheader("Predicted Class Distribution")
            st.altair_chart(bar_chart_pro(dist, "class:N", "count:Q"), use_container_width=True)
            if conf is not None:
                bins = np.linspace(0, 1, 11)
                hist, edges = np.histogram(conf, bins=bins)
                conf_df = pd.DataFrame({
                    "bucket": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))],
                    "count": hist
                })
                st.subheader("Confidence Histogram")
                st.altair_chart(bar_chart_pro(conf_df, "bucket:N", "count:Q"), use_container_width=True)

# INSIGHTS
with tab_insights:
    st.header("Recent Predictions")
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history.tail(15), use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])
    else:
        st.caption("No predictions yet.")


