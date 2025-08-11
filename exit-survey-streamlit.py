import os, io, traceback
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack

# ---- sklearn shim for old pickles that reference private class
import sklearn.compose._column_transformer as ct
class _RemainderColsList(list): pass
ct._RemainderColsList = _RemainderColsList

BASE_DIR = os.path.dirname(__file__)
REQUIRED_FILES = ["best_model.pkl", "preprocess.pkl", "text_vectorizer.pkl", "column_config.pkl"]

st.set_page_config(page_title="Exit Survey Classifier", layout="wide")

# ---------- utils ----------
def csv_template_bytes(cols) -> bytes:
    tpl = pd.DataFrame([{c: "" for c in cols}])
    buf = io.StringIO()
    tpl.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

@st.cache_resource
def load_artifacts():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(BASE_DIR, f))]
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
        return preds, conf
    return model.predict(X), None

# ---------- load artifacts ----------
try:
    model, preprocess, text_vectorizer, cfg = load_artifacts()
except Exception as e:
    st.title("Exit Survey – Reason for Leaving")
    st.error("🚨 App failed to start because artifacts could not be loaded.")
    st.exception(e)
    st.code(traceback.format_exc())
    st.stop()

NUMERIC_COLS = cfg.get("NUMERIC_COLS", [])
CATEGORICAL_COLS = cfg.get("CATEGORICAL_COLS", [])
TEXT_COL = cfg.get("TEXT_COL", None)
CLASS_ORDER = cfg.get("CLASS_ORDER", None)
CAT_CHOICES = cfg.get("CATEGORICAL_CHOICES", {})  # <-- drives dropdowns

EXPECTED_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if TEXT_COL else [])

# ---------- tabs ----------
tab_overview, tab_manual, tab_batch, tab_model = st.tabs(
    ["Overview", "Manual Prediction", "CSV Upload", "Model Details"]
)

with tab_overview:
    st.title("Exit Survey – Reason for Leaving")
    st.write(
        "Enter a single record in **Manual Prediction** or upload a CSV in **CSV Upload**. "
        "The app uses your trained model to predict the reason for leaving."
    )
    st.subheader("Expected Input Columns")
    st.write(EXPECTED_COLS if EXPECTED_COLS else "No schema found in config.")
    st.download_button(
        "Download CSV template",
        data=csv_template_bytes(EXPECTED_COLS),
        file_name="exit_survey_template.csv",
        mime="text/csv",
    )
    with st.expander("Debug (env & files)"):
        import sys, sklearn
        st.write("Python:", sys.version.split()[0])
        st.write("sklearn:", sklearn.__version__)
        st.write("Files present:", {f: os.path.exists(os.path.join(BASE_DIR, f)) for f in REQUIRED_FILES})

with tab_manual:
    st.subheader("Manual Prediction")
    vals = {}
    col1, col2 = st.columns(2)

    # numeric inputs -> two decimals
    with col1:
        for col in NUMERIC_COLS:
            vals[col] = st.number_input(col, value=0.00, step=0.01, format="%.2f")

    # categoricals -> dropdowns if choices provided, else text input
    with col2:
        for col in CATEGORICAL_COLS:
            choices = CAT_CHOICES.get(col, [])
            if choices:
                # ensure str type in UI; model handles encoding
                choices = [str(x) for x in choices]
                vals[col] = st.selectbox(col, choices, index=0)
            else:
                vals[col] = st.text_input(col, value="")

    if TEXT_COL:
        vals[TEXT_COL] = st.text_area(TEXT_COL, value="", height=100)

    if st.button("Predict"):
        row = pd.DataFrame([vals])
        # keep column order
        row = row[EXPECTED_COLS] if EXPECTED_COLS else row
        preds, conf = predict_df(row, model, preprocess, text_vectorizer, TEXT_COL)
        if conf is not None:
            st.success(f"Predicted reason: **{preds[0]}**  (confidence: {conf[0]:.2%})")
        else:
            st.success(f"Predicted reason: **{preds[0]}**")

with tab_batch:
    st.subheader("Batch Predictions via CSV")
    f = st.file_uploader("Upload CSV matching the expected schema", type=["csv"])
    st.caption("Tip: use the template from the Overview tab.")
    if f:
        try:
            data = pd.read_csv(f)
        except Exception as e:
            st.error("Could not read your CSV.")
            st.exception(e)
            st.stop()
        missing = [c for c in EXPECTED_COLS if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            preds, conf = predict_df(data[EXPECTED_COLS], model, preprocess, text_vectorizer, TEXT_COL)
            out = data.copy()
            out["prediction"] = preds
            if conf is not None:
                out["confidence"] = conf
            st.dataframe(out, use_container_width=True)

with tab_model:
    st.subheader("Model Details")
    if CLASS_ORDER is not None:
        st.write("Class labels / order:", CLASS_ORDER)
    st.write("Numeric features:", NUMERIC_COLS)
    st.write("Categorical features:", CATEGORICAL_COLS)
    st.write("Text feature:", TEXT_COL if TEXT_COL else "None")
    if CAT_CHOICES:
        st.write("Dropdown choices (from config):", CAT_CHOICES)
    else:
        st.info("Add `CATEGORICAL_CHOICES` to `column_config.pkl` to enable dropdowns.")

