import streamlit as st
import pandas as pd
import joblib
import os

# --- Path Configuration ---
# This ensures the app finds the files regardless of where you run it from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'epl_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_matches.csv')

# --- Load System Files ---
if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    st.sidebar.success("✅ System Loaded Successfully")
else:
    st.error("⚠️ System Files Missing!")
    st.info(f"Looking for Model at: {MODEL_PATH}")
    st.info(f"Looking for Data at: {DATA_PATH}")
    st.stop()