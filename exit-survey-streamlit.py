# -*- coding: utf-8 -*-
# Exit Survey Classifier — Glow Theme Edition 🌟
# (Drop-in replacement for your existing script)

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
    page_title="Exit Survey Classifier 🌟",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Global Style (Glow + Pastel)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg: #0f172a;         /* deep slate */
  --card: #0b1223;       /* darker slate */
  --ink: #e5e7eb;        /* light text */
  --muted: #9ca3af;      /* muted text */
  --accent: #f472b6;     /* pink */
  --accent2: #60a5fa;    /* blue */
  --accent3: #34d399;    /* green */
  --glow: 0 0 30px rgba(244,114,182,0.2), 0 0 60px rgba(96,165,250,0.15);
}

html, body, [class^="css"]  {
  background: radial-gradient(1200px 700px at 10% 10%, rgba(96,165,250,0.10), transparent 60%),
              radial-gradient(1200px 700px at 90% 20%, rgba(244,114,182,0.10), transparent 60%),
              #0a0f1f !important;
  color: var(--ink);
}
section.main > div { padding-top: 1.4rem; padding-bottom: 2.2rem; }

h1, h2, h3, h4 { color: var(--ink); letter-spacing: 0.2px; }

.glass {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: var(--glow);
}

.header-hero {
  border-radius: 18px;
  padding: 18px 22px;
  background: linear-gradient(135deg, rgba(244,114,182,0.35), rgba(96,165,250,0.35));
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: var(--glow);
}

.kpi {
  display: grid; gap: 8px;
  background: var(--card);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 12px 14px;
}
.kpi .label { color: var(--muted); font-size: 0.85rem; }
.kpi .value { font-weight: 700; font-size: 1.15rem; color: var(--ink); }

.badge {
  display:inline-block; padding: 2px 10px;
  border-radius: 999px; border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06); color: var(--ink);
  font-weight: 600; font-size: 0.78rem; letter-spacing: 0.3px;
}

.cta {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(96,165,250,0.15)) !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; flex-wrap: wrap; }
.stTabs [role="tab"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-bottom: 2px solid transparent;
  border-radius: 12px 12px 0 0;
  color: var(--ink);
}
.stTabs [aria-selected="true"] {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(244,114,182,0.4) inset;
}

.dataframe { background: var(--card) !important; }
.small { color: var(--muted); font-size: 0.85rem; }

hr.divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent); margin: 12px 0 18px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-hero">
  <h1>💼 Exit Survey Classifier <span class="badge">AI-Powered</span></h1>
  <div class="small">Predict the primary reason for leaving from survey inputs — manually or in bulk via CSV.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _csv_template_bytes(cols) -> bytes:
    tpl = pd.DataFrame([{c: "" for c in cols}])
    buf = io.StringIO(); tpl.to_csv(_

