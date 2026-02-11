import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="EPL Win Predictor", page_icon="⚽", layout="centered")

st.title("⚽ EPL Match Outcome Predictor")
st.markdown("""
This app uses a **Random Forest Machine Learning model** trained on years of Premier League data. 
It analyzes the **Home Team's recent form** (rolling averages) against the **Away Team's difficulty** to predict a winner.
""")

# 2. File Path Definitions
model_path = 'outputs/epl_model.pkl'
data_path = 'data/processed_matches.csv'

# 3. Load Model and Data
if not os.path.exists(model_path) or not os.path.exists(data_path):
    st.error("⚠️ System Files Missing! Please run the Master Cell in your Jupyter Notebook to generate the model and data.")
else:
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # 4. Sidebar Selection
    st.sidebar.header("Match Selection")
    
    # Select Home Team
    all_teams = sorted(df['team'].unique())
    home_team = st.sidebar.selectbox("Select Home Team", all_teams, index=all_teams.index("Arsenal") if "Arsenal" in all_teams else 0)
    
    # Select Away Team
    all_opponents = sorted(df['opponent'].unique())
    away_team = st.sidebar.selectbox("Select Away Team", all_opponents, index=all_opponents.index("Chelsea") if "Chelsea" in all_opponents else 1)

    # 5. Prediction Logic
    if st.button("Analyze Matchup & Predict"):
        if home_team == away_team:
            st.warning("Please select two different teams.")
        else:
            # Step A: Get the latest 'Form' for the selected Home Team
            # This pulls their most recent gf_rolling and ga_rolling
            home_form = df[df['team'] == home_team].iloc[-1:]
            
            # Step B: Get the numeric 'Opponent Code' for the selected Away Team
            # This ensures the model knows exactly who the opponent is
            try:
                opp_code = df[df['opponent'] == away_team]['opp_code'].iloc[0]
            except IndexError:
                # Fallback if names are slightly different
                opp_code = 0 

            # Step C: Build the feature vector for the model
            # Order must match training: [venue_code, opp_code, day_code, gf_rolling, ga_rolling]
            # venue_code is 1 because we are predicting from the Home Team perspective
            input_data = pd.DataFrame({
                'venue_code': [1],
                'opp_code': [opp_code],
                'day_code': [home_form['day_code'].values[0]],
                'gf_rolling': [home_form['gf_rolling'].values[0]],
                'ga_rolling': [home_form['ga_rolling'].values[0]]
            })

            # Step D: Execute Prediction
            prediction_proba = model.predict_proba(input_data)
            win_prob = prediction_proba[0][1]

            # 6. Display the UI Results
            st.divider()
            st.subheader(f"Projected Outcome: {home_team} vs {away_team}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label=f"{home_team} Win Probability", value=f"{win_prob*100:.1f}%")
            
            with col2:
                if win_prob > 0.65:
                    st.success(f"**Verdict:** Strong favorite. {home_team} is highly likely to win.")
                elif win_prob > 0.50:
                    st.info(f"**Verdict:** Marginal favorite. The model leans towards {home_team}.")
                else:
                    st.warning(f"**Verdict:** High risk. The model suggests a Draw or {away_team} win.")

            # Step E: Context for the user (Hiring Managers love this detail)
            st.expander("View Model Analysis Data").write(f"""
            - **{home_team}'s current offensive form:** {home_form['gf_rolling'].values[0]:.2f} goals/game (last 3)
            - **{home_team}'s current defensive form:** {home_form['ga_rolling'].values[0]:.2f} goals conceded/game (last 3)
            - **Opponent Weight (opp_code):** {opp_code}
            """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Data Source: Football-Data.co.uk | Model: Random Forest Classifier")