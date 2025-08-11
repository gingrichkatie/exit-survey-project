import os, io, traceback
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

# ---- sklearn shim for old pickles that reference private class
import sklearn.compose._column_transformer as ct
class _RemainderColsList(list): pass
ct._RemainderColsList = _RemainderColsList

BASE_DIR = os.path.dirname(__file__)
REQ = ["best_model.pkl", "preprocess.pkl", "text_vectorizer.pkl", "column_config.pkl"]

st.set_page_config(page_title="Exit Survey Classifier", layout="wide")

# ---------- light style polish ----------
st.markdown("""
<style>
/* tone down default gray and add spacing */
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
/* nicer big title */
h1 {letter-spacing: .2px;}
/* cardy panels */
.stCard {background: #ffffff; border: 1px solid #eee; padding: 1rem 1.2rem; border-radius: 14px;}
/* tighter select inputs */
.css-1d391kg, .stSelectbox {margin-bottom: 0.15rem;}
</style>
""", unsafe_allow_html=True)

# ---------- utilities ----------
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

# ---------- load artifacts (no debug panel shown) ----------
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

# ---------- sidebar ----------
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose", ["Manual Prediction", "CSV Upload"], index=0)
st.sidebar.caption("Use the CSV template from the Overview section in the main area.")

# ---------- header ----------
st.title("Exit Survey – Reason for Leaving")

with st.container():
    st.markdown(
        "<div class='stCard'>Use the manual form for a single person, or upload a CSV for many. "
        "We’ll return a predicted reason and confidence with a couple of quick charts.</div>",
        unsafe_allow_html=True
    )

# session state to keep a tiny prediction history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])

# =========================================================
# ================ Manual Prediction tab ==================
# =========================================================
if mode == "Manual Prediction":
    vals = {}
    col1, col2 = st.columns(2)

    # numerics: 2-decimal, no NaNs
    with col1:
        for col in NUMERIC_COLS:
            vals[col] = st.number_input(col, value=0.00, step=0.01, format="%.2f")

    # categoricals: dropdowns if choices provided; require explicit choice
    with col2:
        for col in CATEGORICAL_COLS:
            choices = [str(x) for x in CAT_CHOICES.get(col, [])]
            # put 'Other' last when present
            if "Other" in choices:
                choices = [c for c in choices if c != "Other"] + ["Other"]
            if choices:
                vals[col] = st.selectbox(col, options=choices, index=None, placeholder=f"Select {col}")
            else:
                vals[col] = st.text_input(col, value="")

    if TEXT_COL:
        vals[TEXT_COL] = st.text_area(TEXT_COL, value="", height=100, placeholder="Enter short text")

    # require all non-text categorical fields to be chosen
    ready = all(v not in (None, "") for k, v in vals.items() if (k in CATEGORICAL_COLS))
    disabled_msg = None if ready else "Select all dropdown values to enable prediction."
    c1, c2 = st.columns([1, 3])
    with c1:
        predict_clicked = st.button("Predict", disabled=not ready)
    if disabled_msg and not ready:
        st.caption(disabled_msg)

    if predict_clicked:
        row = pd.DataFrame([vals])
        row = row[EXPECTED_COLS] if EXPECTED_COLS else row

        # basic normalization
        for c in CATEGORICAL_COLS:
            row[c] = row[c].astype(str).str.strip()
        if NUMERIC_COLS:
            row[NUMERIC_COLS] = row[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        preds, conf, proba, classes_ = predict_df(row, model, preprocess, text_vectorizer, TEXT_COL)

        # show result
        if conf is not None:
            st.success(f"Predicted reason: **{preds[0]}**  (confidence: {conf[0]:.2%})")
        else:
            st.success(f"Predicted reason: **{preds[0]}**")

        # tiny probability bar if available
        if proba is not None:
            p = pd.DataFrame({"class": classes_, "prob": proba.flatten()})
            p = p.sort_values("prob", ascending=False)
            st.bar_chart(p.set_index("class"))

        # update history
        hist_row = row.copy()
        hist_row["prediction"] = preds[0]
        hist_row["confidence"] = None if conf is None else float(conf[0])
        st.session_state.history = pd.concat([st.session_state.history, hist_row], ignore_index=True)

# =========================================================
# =================== CSV Upload tab ======================
# =========================================================
else:
    st.subheader("Batch predictions from CSV")
    st.download_button(
        "Download CSV template",
        data=_csv_template_bytes(EXPECTED_COLS),
        file_name="exit_survey_template.csv",
        mime="text/csv",
    )

    f = st.file_uploader("Upload CSV matching the expected schema", type=["csv"])
    st.caption("Tip: headers must exactly match the template.")

    if f:
        try:
            data = pd.read_csv(f)
        except Exception as e:
            st.error("Could not read your CSV."); st.exception(e); st.stop()

        missing = [c for c in EXPECTED_COLS if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
            st.stop()

        # order + light cleanup
        data = data[EXPECTED_COLS]
        for c in CATEGORICAL_COLS:
            data[c] = data[c].astype(str).str.strip()
        if NUMERIC_COLS:
            data[NUMERIC_COLS] = data[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        preds, conf, proba, classes_ = predict_df(data, model, preprocess, text_vectorizer, TEXT_COL)

        out = data.copy()
        out["prediction"] = preds
        if conf is not None:
            out["confidence"] = conf
        st.dataframe(out, use_container_width=True)

        # quick insights
        st.markdown("### Insights")
        # class distribution
        dist = pd.Series(preds).value_counts().sort_values(ascending=False)
        st.write("Predicted class distribution")
        st.bar_chart(dist)

        # confidence histogram (if available)
        if conf is not None:
            bins = np.linspace(0, 1, 11)
            hist, edges = np.histogram(conf, bins=bins)
            conf_df = pd.DataFrame({"bucket": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))],
                                    "count": hist})
            st.write("Confidence histogram")
            st.bar_chart(conf_df.set_index("bucket"))

# =========================================================
# ====================== History ==========================
# =========================================================
st.markdown("---")
st.subheader("Recent predictions (this session)")
if not st.session_state.history.empty:
    st.dataframe(st.session_state.history.tail(10), use_container_width=True)
    if st.button("Clear history"):
        st.session_state.history = pd.DataFrame(columns=EXPECTED_COLS + ["prediction", "confidence"])
else:
    st.caption("No predictions yet.")
