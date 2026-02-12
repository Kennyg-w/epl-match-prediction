import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="EPL Predictor", page_icon="⚽")
st.title("⚽ EPL Match Outcome Predictor")
st.markdown("Predict outcomes based on 7 seasons of Premier League data.")

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'epl_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_matches.csv')

# --- Load System Files ---
if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    st.sidebar.success("✅ System Loaded Successfully")
else:
    st.error("⚠️ System Files Missing! Run your Notebook Master Cell.")
    st.stop()

# --- SIDEBAR: User Input ---
st.sidebar.header("Match Selection")

# Get unique list of teams
teams = sorted(df['team'].unique())

home_team = st.sidebar.selectbox("Select Home Team", teams)
away_team = st.sidebar.selectbox("Select Away Team", teams)
day_of_week = st.sidebar.select_slider("Day of Match", 
                                      options=[0,1,2,3,4,5,6], 
                                      format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])

# --- MAIN PAGE: Prediction Logic ---
if st.button("Predict Match Result"):
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        # 1. Fetch recent form for the Home Team (rolling averages)
        # We look at the most recent entry for this team in our data
        home_data = df[df['team'] == home_team].sort_values('date').iloc[-1]
        
        # 2. Fetch the Opponent code for the Away Team
        opp_code = df[df['opponent'] == away_team]['opp_code'].iloc[0]
        
        # 3. Create the input features for the model
        # Order must match: venue_code, opp_code, day_code, gf_rolling, ga_rolling
        features = pd.DataFrame([[
            1, # Venue = Home
            opp_code,
            day_of_week,
            home_data['gf_rolling'],
            home_data['ga_rolling']
        ]], columns=["venue_code", "opp_code", "day_code", "gf_rolling", "ga_rolling"])
        
        # 4. Run Prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] # Probability of Win
        
        # 5. Display Result
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", "HOME WIN" if prediction == 1 else "DRAW or AWAY WIN")
        
        with col2:
            st.metric("Win Confidence", f"{probability*100:.1f}%")
            
        if prediction == 1:
            st.success(f"The model favors **{home_team}** to win against {away_team}!")
        else:
            st.info(f"The model suggests {away_team} might hold {home_team} to a draw or win.")

st.sidebar.divider()
st.sidebar.info("Model: Random Forest Regressor\nLast Trained: 2026 Season Update")