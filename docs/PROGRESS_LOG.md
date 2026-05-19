# Data Mining Final Project - Progress Log

*This document tracks our chronological steps, decisions, and findings for the project presentation.*

## Step 1: Environment Setup & Strategy (May 19, 2026)
**Goal:** Prepare the workspace for a large dataset (1.1GB) on a weak local machine.
**Action:** 
- Created a `venv` and installed dependencies (LightGBM, XGBoost, Pandas).
- Implemented a dual-environment strategy: Wrote a `create_sample.py` script to extract the first 10,000 rows for local development (`train_sample.csv`).
- Configured the main Jupyter Notebook to automatically detect if it runs locally (using the sample) or in Google Colab (using the full dataset via Google Drive mount).

## Step 2: Handling the "Future Date" Problem (May 19, 2026)
**Goal:** Parse the `date` column for time-series analysis.
**Problem:** The dataset contains dates far into the future (e.g., year 3004 up to 58,061). Pandas `pd.to_datetime()` crashed because its internal nanosecond resolution is limited to the year 2262 (OutOfBoundsDatetime error).
**Solution:** Bypassed the standard datetime conversion. We manually split the `date` string (YYYY-MM-DD) into three separate numerical integer features: `year`, `month`, and `day`.
**Result:** Successfully parsed the dates without losing the chronological order or seasonal information.

## Step 3: Initial Exploratory Data Analysis (EDA) (May 19, 2026)
**Goal:** Understand the basic distributions and identify data quality issues.
**Action:** Ran `.describe()` and `.isnull().sum()` on the local 10k sample.
**Key Findings:**
1. **The Score Gap:** Out of 10,000 rows, the target variable `score` is missing in 8,573 rows. It is only present in 1,427 rows. 
2. **Score Distribution:** The score ranges from 0 to 5. The median is 0, but the mean is 0.9, indicating a zero-inflated distribution with rare high-impact events.
3. **Climate:** The data shows distinct seasonal variations (Temp: -6.3°C to 32.9°C) and extreme weather events (Precipitation max: 137.5).

## Step 4: Correlation Analysis (May 19, 2026)
**Goal:** Identify which meteorological features drive the `score`.
**Key Findings:**
1. **Temperature is Leading:** `tmp_range` (0.17) and `tmp_max` (0.15) have the highest correlation. Extreme heat or high daily fluctuations are primary drivers.
2. **Pressure Inverse:** `surf_pre` (-0.11) shows an inverse relationship, suggesting that low-pressure systems (storms) increase the score.
3. **The Precipitation Mystery:** `prec` shows almost no linear correlation (0.01). This suggests that only extreme rainfall matters, or the impact is delayed (lagged).
4. **No Yearly Trend:** The `year` feature has a near-zero correlation, confirming that there isn't a simple year-over-year increase in the score.

## Step 5: Complete EDA Notebook (May 19, 2026)
**Goal:** Full data discovery notebook with local/Colab split — user runs cells themselves.
**Action:**
- Rewrote `01_exploration.ipynb` with 17 sections: structure, missing values, time gaps, regional analysis, correlations, seasonality, lags, rolling windows, extremes, test set, Colab-only chunk summary.
- Every heavy step gated by `IS_COLAB` / `RUN_CHUNK_SUMMARY`; local uses `train_sample.csv`, Colab uses full `train.csv`.
- Section 17: manual findings table to fill after local + Colab runs.

## Step 6: Full-Dataset Stats (chunked, local) (May 19, 2026)
**Source:** Background scan of full `train.csv` (not the 10k sample).

| Metric | Value |
|--------|-------|
| Train rows | 12,319,040 |
| Rows with `score` | 1,757,936 (14.27%) |
| Date range | 10004-12-31 – 8133-12-31 |
| Regions | 2,248 (each 5,480 rows → ~15 years daily) |
| Score per region | 782 / 5,480 (14.27%, uniform) |
| Wetter-Features missing | 0% |
| Score: mean / median | 0.91 / 0.0 |
| Score == 0 | 58.0% |
| Test rows | 204,568 (no `score` column) |

**Implications:** Massive semi-supervised setup; `score` is sparse but evenly distributed across regions. Sample (10k) showed ~14% labeled — matches full data.

## Step 7: EDA Analysis Document (May 19, 2026)
**Action:** Wrote `EDA_ANALYSIS.md` – full interpretation of notebook results + full-dataset stats.
**Covers:** score rhythm (7-day), zero-inflation, correlations, lags/rolling, extremes, train-test shift, modeling recommendations.

## Step 8: Repository Restructure (May 19, 2026)
**Goal:** Cleaner layout for team collaboration.
**Action:**
- `docs/` – PROJECT_PLAN, PROGRESS_LOG, EDA_ANALYSIS, DATA_SETUP, presentation/
- `notebooks/` – numbered notebooks
- `scripts/` – create_sample.py
- `config/paths.py` – shared local/Colab paths
- `data/` – canonical data dir (symlink/legacy supported)
- `outputs/figures/` – EDA plots
- README.md, CONTRIBUTING.md

**Current Task:** Feature engineering notebook (`02_preprocessing.ipynb`) + baseline model (Colab).
