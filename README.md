# ⚽ English Premier League (EPL) Match Predictor & Web App

## 📊 Project Overview
Developed a comprehensive machine learning system to predict English Premier League match outcomes. This project integrates automated data ingestion, advanced time-series feature engineering, and a live web deployment to provide real-time win probabilities.

## 🛠️ Technical Achievements
- **Big Data Scaling:** Trained the model on **5 seasons** of historical match data (>3,500 match perspectives) to improve predictive robustness and capture long-term team performance trends.
- **Advanced Feature Engineering:** Engineered a **3-game rolling average** pipeline for offensive and defensive metrics (GF, GA) to dynamically capture "Team Form."
- **Model Optimization:** Deployed a **Random Forest Classifier** achieving a **52.34% precision rate** for win predictions, significantly outperforming the 33.3% random baseline.
- **Time-Series Integrity:** Implemented a chronological training split to prevent data leakage and ensure the model learns only from historical precedents.
- **Full-Stack Deployment:** Built and launched a live **Streamlit Web Application** allowing users to select any matchup and retrieve model-driven win probabilities.

## 📈 Business & Operational Impact
- Automated the transition from raw historical archives to a production-ready model.
- Provided a scalable framework for sports analytics that can be extended to include player-level stats or betting market odds.

## 🐍 Tech Stack
- **Languages:** Python 3.11 (Conda)
- **ML & Data:** Scikit-Learn, Pandas, NumPy, Joblib
- **Web:** Streamlit
- **Version Control:** Git & GitHub