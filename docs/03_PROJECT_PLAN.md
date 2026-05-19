# Data Mining Final Project - Weather Forecasting Plan

> **Dokumentation Nr. 03** · [Lesereihenfolge](README.md)

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
- [x] Run notebook locally; analysis documented in `06_EDA_ANALYSIS.md`.
- [ ] Run notebook in Colab and validate correlations on full data.

## 3. Data Preprocessing & Feature Engineering
- [x] `notebooks/03_preprocessing.ipynb` + `scripts/features.py`
- [x] Lags (1,3,7,14,21), Rolling (7,14), Kalender sin/cos
- [x] Train+Test Panel für Lags; Output: `outputs/processed/*.parquet`

## 4. Modeling (The "Google Colab" Phase)
- [x] `notebooks/04_modeling.ipynb` – Baselines + LightGBM + Submission
- [ ] Time-Series CV mit 7-Tage-Gap (aktuell: regionaler Zeit-Split 80/20)
- [ ] Hyperparameter Tuning (Optuna)

## 5. Evaluation & Submission
- **Metrics:** Identify the specific competition metric (likely RMSE or MAE).
- **Submission Generation:** Format output to match `sample_submission.csv`.
- **Final Review:** Ensure no overfitting to specific regions.
