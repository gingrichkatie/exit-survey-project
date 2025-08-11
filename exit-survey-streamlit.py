import os, io, traceback
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

# Optional: red charts via Altair (bundled with Streamlit)
import altair as alt

# ---- sklearn shim for old pickles that reference private class
import sklearn.compose._column_transformer as ct
class _RemainderColsList(list): pass
ct._RemainderColsList = _RemainderColsList

BASE_DIR = os.path.dirname(__file__)
REQ = ["best_model.pkl", "preprocess.pkl", "text_vectorizer.pkl", "column_config.pkl"]

st.set_page_config(page_title="Exit Survey Classifier", layout="wide")
st.markdown("""
<style>
/* Wrap tabs so they don't get cut off */
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
}
/* Remove gray borders/outlines on tabs */
.stTabs [role="tab"] {
    border: none !important;
    box-shadow: none !important;
}
/* Optional: adjust padding */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- red & white styling ----------
st.markdown("""
<style>
:root {
  --accent: #C62828;   /* red */
  --accent-soft: #FBE9E7; /* very light red */
}
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
h1, h2, h3 { color: var(--accent); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #fff; border: 1px solid #eee; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { border-color: var(--accent); }
.badge {
  display:inline-block; padding: 2px 8px; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent); font-weight: 600; font-size: 0.8rem;
  border: 1px solid #f4c4c1;
}
.panel {
  background: #fff; border: 1px solid #eee; border-radius: 14px; padding: 14px 16px; margin: 6px 0 12px 0;
}
.table-note { color: #6b6b6b; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def _csv_template_bytes(cols) -> bytes:
    tpl = pd.DataFrame([{c: "" for c in cols}])
    buf = io.StringIO(); tpl.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

@st.cache_resource
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

def bar_chart_red(df, x, y):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X(x, sort='-y'),
        y=y,
        color=alt.value("#C62828")
    ).properties(height=250)

# ---------- load artifacts ----------
try:
    model, preprocess, text_vectorizer, cfg = load_artifacts()
except Exception as e:
    st.title("Exit Survey – Reason for Leaving")
    st.error("The app failed to start because model files could not be loaded.")
    st.exception(e)
    st.code(traceback.format_exc())
    st.stop()

NUMERIC_COLS = cfg.get("NUMERIC_COLS", [])
CATEGORICAL_COLS = cfg.get("CATEGORICAL_COLS", [])
TEXT_COL = cfg.get("TEXT_COL", None)
CLASS_ORDER = cfg.get("CLASS_ORDER", None)
CAT_CHOICES = cfg.get("CATEGORICAL_CHOICES", {})  # drives dropdowns if present
EXPECTED_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if TEXT_COL else [])

# Integer-only numeric fields (Age by default if present)
INT_COLS = cfg.get("INTEGER_COLS", [])
if "Age" in NUMERIC_COLS and "Age" not in INT_COLS:
    INT_COLS = list(set(INT_COLS + ["Age"]))

# ---------- tabs ----------
tab_manual, tab_csv, tab_insights = st.tabs(["Manual Prediction", "CSV Upload", "Insights"])

# session history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])

# =========================================================
# Manual Prediction
# =========================================================
with tab_manual:
    st.header("Manual Prediction")
    st.markdown("<div class='panel'>Fill out the form and we’ll predict the primary reason for leaving.</div>", unsafe_allow_html=True)

    vals = {}
    col1, col2 = st.columns(2)

    # Numerics: Age (int), others 2-decimals
    with col1:
        for col in NUMERIC_COLS:
            if col in INT_COLS:
                vals[col] = st.number_input(col, value=0, step=1, format="%d")
            else:
                vals[col] = st.number_input(col, value=0.00, step=0.01, format="%.2f")

    # Categorical dropdowns: require choice, put "Other" last
    with col2:
        for col in CATEGORICAL_COLS:
            choices = [str(x) for x in CAT_CHOICES.get(col, [])]
            if "Other" in choices:
                choices = [c for c in choices if c != "Other"] + ["Other"]
            if choices:
                vals[col] = st.selectbox(col, options=choices, index=None, placeholder=f"Select {col}")
            else:
                vals[col] = st.text_input(col, value="")

    if TEXT_COL:
        vals[TEXT_COL] = st.text_area(TEXT_COL, value="", height=100, placeholder="Optional short text")

    # Require all dropdowns chosen
    ready = all(v not in (None, "") for k, v in vals.items() if k in CATEGORICAL_COLS)
    c1, c2 = st.columns([1, 3])
    with c1:
        predict_clicked = st.button("Predict", disabled=not ready)
    if not ready:
        st.caption("Select all dropdown values to enable prediction.")

    if predict_clicked:
        row = pd.DataFrame([vals])
        row = row[EXPECTED_COLS] if EXPECTED_COLS else row

        # normalize types
        for c in CATEGORICAL_COLS:
            row[c] = row[c].astype(str).str.strip()
        if NUMERIC_COLS:
            for c in NUMERIC_COLS:
                if c in INT_COLS:
                    row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0).astype(int)
                else:
                    row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0.0)

        preds, conf, proba, classes_ = predict_df(row, model, preprocess, text_vectorizer, TEXT_COL)

        # result card
        if conf is not None:
            st.markdown(f"<span class='badge'>Prediction</span> **{preds[0]}** &nbsp;&nbsp; "
                        f"<span class='badge'>Confidence</span> **{conf[0]:.0%}**",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge'>Prediction</span> **{preds[0]}**", unsafe_allow_html=True)

        # probability bar (red) if available
        if proba is not None:
            p = pd.DataFrame({"class": classes_, "prob": proba.flatten()}).sort_values("prob", ascending=False)
            st.altair_chart(bar_chart_red(p, "class:N", "prob:Q"), use_container_width=True)

        # update history (latest at bottom)
        hist_row = row.copy()
        hist_row["prediction"] = preds[0]
        hist_row["confidence"] = None if conf is None else float(conf[0])
        st.session_state.history = pd.concat([st.session_state.history, hist_row], ignore_index=True)

# =========================================================
# CSV Upload
# =========================================================
with tab_csv:
    st.header("CSV Upload")
    st.markdown("<div class='panel'>Upload many rows at once. Use the template to match the expected columns.</div>", unsafe_allow_html=True)

    st.download_button(
        "Download CSV template",
        data=_csv_template_bytes(EXPECTED_COLS),
        file_name="exit_survey_template.csv",
        mime="text/csv",
    )

    f = st.file_uploader("Upload CSV matching the expected schema", type=["csv"])

    if f:
        try:
            data = pd.read_csv(f)
        except Exception as e:
            st.error("Could not read your CSV."); st.exception(e); st.stop()

        missing = [c for c in EXPECTED_COLS if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing)); st.stop()

        data = data[EXPECTED_COLS]
        for c in CATEGORICAL_COLS:
            data[c] = data[c].astype(str).str.strip()
        if NUMERIC_COLS:
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

        # quick distribution chart
        dist = pd.Series(preds).value_counts().reset_index()
        dist.columns = ["class", "count"]
        st.subheader("Predicted class distribution")
        st.altair_chart(bar_chart_red(dist, "class:N", "count:Q"), use_container_width=True)

        # confidence histogram if available
        if conf is not None:
            bins = np.linspace(0, 1, 11)
            hist, edges = np.histogram(conf, bins=bins)
            conf_df = pd.DataFrame({
                "bucket": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))],
                "count": hist
            })
            st.subheader("Confidence histogram")
            st.altair_chart(bar_chart_red(conf_df, "bucket:N", "count:Q"), use_container_width=True)

# =========================================================
# Insights / Recent predictions
# =========================================================
with tab_insights:
    st.header("Recent Predictions")
    if not st.session_state.history.empty:
        # highlight latest row textually (styling limitations in st.dataframe)
        st.markdown("<span class='badge'>Newest</span> is the last row below.", unsafe_allow_html=True)
        st.dataframe(st.session_state.history.tail(15), use_container_width=True)
        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("Clear history"):
                st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])
    else:
        st.caption("No predictions yet.")

# (No debug panels shown; errors surface only if the app cannot start)
