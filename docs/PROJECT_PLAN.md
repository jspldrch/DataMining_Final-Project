# Data Mining Final Project - Weather Forecasting Plan

## 1. Setup & Environment
- [x] Initialize `.gitignore` and `requirements.txt`.
- [x] Setup local `venv` for code preparation.
- [ ] Connect GitHub repo to Google Colab.
- [x] Implement data loading strategy (Chunking/Sampling for local EDA).

## 2. Exploratory Data Analysis (EDA)
- [x] Notebook `01_exploration.ipynb` (local sample + Colab full data).
- **Time Analysis:** Check for missing dates and the frequency of observations.
- **Regional Analysis:** Distribution of scores across different `region_id`s.
- **Feature Correlations:** Relationship between temperature, wind, humidity, and the target score.
- **Seasonality:** Identify weekly or yearly patterns.
- [x] Run notebook locally; analysis documented in `EDA_ANALYSIS.md`.
- [ ] Run notebook in Colab and validate correlations on full data.

## 3. Data Preprocessing & Feature Engineering
- **Handle Missing Values:** Imputation based on regional averages or linear interpolation.
- **Time-Series Features:**
    - Lag features (e.g., weather from 7, 14, 21 days ago).
    - Rolling window statistics (mean, std, max of the last week).
    - Seasonal decomposition (extracting trends and residuals).
- **Encoding:** Proper handling of `region_id` (Target encoding or Embedding).

## 4. Modeling (The "Google Colab" Phase)
- **Baseline:** Simple linear regression or persistence model (last week = next week).
- **Primary Model:** LightGBM (preferred for speed/memory efficiency on 1.1GB data) or XGBoost.
- **Cross-Validation:** Time-Series Cross-Validation (Walk-forward) to prevent data leakage.
- **Hyperparameter Tuning:** Optuna for efficient search.

## 5. Evaluation & Submission
- **Metrics:** Identify the specific competition metric (likely RMSE or MAE).
- **Submission Generation:** Format output to match `sample_submission.csv`.
- **Final Review:** Ensure no overfitting to specific regions.
