mport os, io, traceback
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack

# ---- sklearn shim for old pickles that reference private class
import sklearn.compose._column_transformer as ct
class _RemainderColsList(list): pass
ct._RemainderColsList = _RemainderColsList

BASE_DIR = os.path.dirname(__file__)
REQUIRED_FILES = ["best_model.pkl", "preprocess.pkl", "text_vectorizer.pkl", "column_config.pkl"]

st.set_page_config(page_title="Exit Survey Classifier", layout="wide")
st.title("Exit Survey – Reason for Leaving")

# ---- Quick environment + file sanity
with st.expander("Debug (env & files)"):
    import sys, sklearn, numpy as np
    st.write("Python:", sys.version.split()[0])
    st.write("sklearn:", sklearn.__version__)
    st.write("numpy:", np.__version__)
    st.write("Files present:", {f: os.path.exists(os.path.join(BASE_DIR, f)) for f in REQUIRED_FILES})

@st.cache_resource
def load_artifacts():
    # Existence check first
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing files next to this script: {', '.join(missing)}")

    # Try to load and show the real traceback if anything fails
    try:
        model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
        preprocess = joblib.load(os.path.join(BASE_DIR, "preprocess.pkl"))
        text_vectorizer = joblib.load(os.path.join(BASE_DIR, "text_vectorizer.pkl"))
        config = joblib.load(os.path.join(BASE_DIR, "column_config.pkl"))
        return model, preprocess, text_vectorizer, config
    except Exception as e:
        # Re-raise with more context so Streamlit shows full traceback
        raise RuntimeError("Failed while loading model artifacts.") from e

# Load artifacts with inline error surface
try:
    model, preprocess, text_vectorizer, cfg = load_artifacts()
except Exception as e:
    st.error("🚨 App failed to start because artifacts could not be loaded.")
    st.exception(e)
    st.code(traceback.format_exc())
    st.stop()

NUMERIC_COLS = cfg.get('NUMERIC_COLS', [])
CATEGORICAL_COLS = cfg.get('CATEGORICAL_COLS', [])
TEXT_COL = cfg.get('TEXT_COL', None)

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose", ["Upload CSV", "Manual Form"])

def to_matrix(df_in: pd.DataFrame):
    Xs = preprocess.transform(df_in)
    if TEXT_COL:
        # Only try text if both a column is specified and we actually have a vectorizer object
        if text_vectorizer is not None and TEXT_COL in df_in.columns:
            Xt = text_vectorizer.transform(df_in[TEXT_COL].astype(str).fillna(""))
            return hstack([Xs, Xt]).tocsr()
    return Xs

# Helper: template CSV download so uploads match schema
def csv_template_bytes() -> bytes:
    cols = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if TEXT_COL else [])
    tpl = pd.DataFrame([{c: "" for c in cols}])
    buf = io.StringIO()
    tpl.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

st.download_button(
    "Download CSV template",
    data=csv_template_bytes(),
    file_name="exit_survey_template.csv",
    mime="text/csv",
    help="Use this so your upload has the exact columns the model expects.",
)

if mode == "Upload CSV":
    f = st.file_uploader("Upload CSV matching training schema", type=["csv"])
    if f:
        try:
            data = pd.read_csv(f)
        except Exception as e:
            st.error("Could not read your CSV.")
            st.exception(e)
            st.stop()

        expected = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if TEXT_COL else [])
        missing = [c for c in expected if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            X = to_matrix(data)
            try:
                preds = model.predict(X)
            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)
                st.stop()
            out = data.copy()
            out['prediction'] = preds
            st.dataframe(out, use_container_width=True)
else:
    vals = {}
    col1, col2 = st.columns(2)
    with col1:
        for col in NUMERIC_COLS:
            vals[col] = st.number_input(col, value=0.0)
    with col2:
        for col in CATEGORICAL_COLS:
            vals[col] = st.text_input(col, value="")
    if TEXT_COL:
        vals[TEXT_COL] = st.text_area(TEXT_COL, value="", height=100)

    if st.button("Predict"):
        row = pd.DataFrame([vals])
        X = to_matrix(row)
        try:
            pred = model.predict(X)[0]
            st.success(f"Predicted reason: **{pred}**")
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
