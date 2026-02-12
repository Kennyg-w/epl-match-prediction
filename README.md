# ⚽ EPL Match Outcome Predictor (2026 Update)

## 📊 Project Summary
Developed a machine learning web application to predict English Premier League match outcomes. The system utilises 7 seasons of historical data (2019-2026) to assess team strength and recent form.

## 🛠️ Key Achievements
- **Real-Time Data Pipeline:** Automated fetching of ongoing 2025/2026 season data directly from football-data.co.uk.
- **Advanced Feature Engineering:** Engineered "Rolling Averages" (GF/GA) to capture short-term team momentum (Form).
- **Modeling:** Trained a Random Forest Classifier with 200 estimators, prioritising Precision to ensure reliable "Win" predictions.
- **Web UI:** Built an interactive dashboard using **Streamlit**, allowing users to select matchups and see real-time win probabilities.

## 🐍 Tech Stack
- **AI/ML:** Scikit-Learn, Random Forest
- **Data:** Pandas, Joblib
- **Web Framework:** Streamlit