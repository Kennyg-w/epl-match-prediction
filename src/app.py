import streamlit as st
import pandas as pd
import joblib

st.title("⚽ EPL Predictor")

try:
    model = joblib.load('outputs/epl_model.pkl')
    df = pd.read_csv('data/processed_matches.csv')

    teams = sorted(df['team'].unique())
    home_team = st.selectbox("Select Home Team", teams)
    
    opponents = sorted(df['opponent'].unique())
    away_team = st.selectbox("Select Away Team", opponents)

    if st.button("Predict"):
        # Get latest stats for selected team
        latest_data = df[df['team'] == home_team].iloc[-1:]
        predictors = ["venue_code", "opp_code", "day_code", "gf_rolling", "ga_rolling"]
        
        # Force to Home perspective
        input_features = latest_data[predictors].copy()
        input_features["venue_code"] = 1 
        
        prob = model.predict_proba(input_features)[0][1]
        st.metric("Win Probability", f"{prob*100:.1f}%")

except Exception as e:
    st.error(f"Waiting for data: {e}")