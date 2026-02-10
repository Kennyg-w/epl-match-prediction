# ⚽ English Premier League (EPL) Match Predictor

## 📊 Project Overview
Developed a machine learning system to predict the winners of English Premier League matches. The project focuses on handling time-series dependencies and engineering "form-based" features to improve prediction precision.

## 🛠️ Technical Achievements
- **Data Engineering:** Scraped and processed historical match data, converting categorical team and venue data into numeric features for model ingestion.
- **Time-Series Logic:** Implemented a non-random training split (Past vs. Future) to ensure the model learned from historical trends without "looking ahead."
- **Feature Engineering:** Developed a **3-game rolling average** system for goals scored (GF) and goals against (GA) to capture team "form."
- **Performance:** Achieved an **Improved Precision of 52.34%**, successfully identifying winning patterns in a high-volatility sports environment.

## 🐍 Tech Stack
- **Machine Learning:** Random Forest Classifier (Scikit-Learn)
- **Data Manipulation:** Pandas, NumPy
- **Environment:** Python 3.11 (Conda)