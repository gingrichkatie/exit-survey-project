import pandas as pd
import joblib
from scipy.sparse import hstack

st.set_page_config(page_title="Exit Survey Classifier", layout="wide")
st.title("Exit Survey – Reason for Leaving")

@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/best_model.pkl")
    preprocess = joblib.load("artifacts/preprocess.pkl")
    text_vectorizer = joblib.load("artifacts/text_vectorizer.pkl")
    config = joblib.load("artifacts/column_config.pkl")
    return model, preprocess, text_vectorizer, config

model, preprocess, text_vectorizer, cfg = load_artifacts()
NUMERIC_COLS = cfg['NUMERIC_COLS']
CATEGORICAL_COLS = cfg['CATEGORICAL_COLS']
TEXT_COL = cfg['TEXT_COL']

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose", ["Upload CSV", "Manual Form"])

def to_matrix(df_in):
    Xs = preprocess.transform(df_in)
    if TEXT_COL and text_vectorizer:
        Xt = text_vectorizer.transform(df_in[TEXT_COL].astype(str).fillna(""))
        return hstack([Xs, Xt]).tocsr()
    return Xs

if mode == "Upload CSV":
    f = st.file_uploader("Upload CSV matching training schema", type=["csv"])
    if f:
        data = pd.read_csv(f)
        missing = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if TEXT_COL else [])) if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            X = to_matrix(data)
            preds = model.predict(X)
            out = data.copy()
            out['prediction'] = preds
            st.dataframe(out)
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
        pred = model.predict(X)[0]
        st.success(f"Predicted reason: **{pred}**")
