# Natural Disaster Severity Prediction (Drought Forecasting)

**Data Mining Final Project, Spring 2026 — Group 9**

This repository contains the code for the Data Mining Final Project. The objective is to
forecast weekly drought severity scores (0 to 5) for 2,248 regions across the next 5 weeks,
using only historical daily meteorological data.

Our final model ([scripts/kaggle_v31_stratified.py](scripts/kaggle_v31_stratified.py)) achieves
a **Kaggle public-leaderboard MAE of 0.7962**, beating the course's hardest baseline
(Baseline 3: **0.8056**).

> The full chronological progression of every model we tried — including the dead ends
> (deep learning, autoregressive score-lags, spatial features) — is documented in
> [archiv/MODEL_HISTORY.md](archiv/MODEL_HISTORY.md).

---

## Repository Structure

```
├── scripts/
│   ├── kaggle_v31_stratified.py   # FINAL model (SPI + stratified holdout + ensemble)
│   └── convert_to_npz.py          # Converts train.csv/test.csv to compressed NPZ
├── outputs/
│   └── feature_importance.png     # Feature importance figure used in the report
├── report.tex / reference.bib     # IEEE report sources
├── DM_project_Group_9.pdf         # Compiled report (submission file)
├── requirements.txt               # Python dependencies (pinned to reported versions)
├── data/                          # Raw CSVs / NPZs (git-ignored, not committed)
└── archiv/                        # All experimental models, notebooks, EDA, docs
    ├── MODEL_HISTORY.md           # Chronological model timeline (v1 → v34 + dead ends)
    ├── scripts/                   # Experimental Kaggle/local scripts (kaggle_v*, run_v*)
    ├── arthur_scripts/            # Parallel experiment track (spatial, hybrid, DL)
    ├── notebooks/                 # EDA & preprocessing notebooks
    └── docs/                      # Project documentation & progress log
```

---

## How to Reproduce the Final Result

The final model is designed to run in a **Kaggle Notebook** (CPU is sufficient, no GPU
required). The competition `train.csv` is ~1.1 GB, so we first compress it to NumPy `.npz`
format, upload that as a Kaggle Dataset, and run the training script against it.

### Step 1 — Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Compress the raw data locally

Place the competition files in a `data/` folder in the project root:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv` (also accepted under `resources/sample_submission.csv`)

Then convert the CSVs to compressed NPZ (≈71% smaller, ~12x faster to load):

```bash
python scripts/convert_to_npz.py
```

This produces `data/train.npz`, `data/test.npz` and `data/sample_submission.npz`.

### Step 3 — Upload to Kaggle and run

1. Create a **Kaggle Dataset** containing `train.npz`, `test.npz` and
   `sample_submission.csv`.
2. In a Kaggle Notebook, attach that dataset and run
   [scripts/kaggle_v31_stratified.py](scripts/kaggle_v31_stratified.py).
3. The submission file is written to `/kaggle/working/submission_v31_stratified.csv`.

*Runtime:* ~3.2 hours on a standard Kaggle CPU instance. Subsequent runs are much faster
because the script caches engineered features (`cache_weekly_v31s.npz`,
`cache_windows_v31s.npz`) in `/kaggle/working`.

### Data paths used by the script

| Role | Location | Notes |
|------|----------|-------|
| Input NPZ/CSV | `/kaggle/input/<slug>/` | The script auto-detects the files. Accepted slugs: `datafinal`, `datafiles`, `datatrain`, `datatest`, `traindataset`, `testdataset`, `data`, `samplesub`, `samplesubmission`. It also falls back to a recursive `glob` under `/kaggle/input/**/`, so the dataset slug name does not matter as long as the **file names** (`train.npz`, `test.npz`, `sample_submission.csv`) are correct. |
| Output | `/kaggle/working/submission_v31_stratified.csv` | Final Kaggle submission. |
| Feature cache | `/kaggle/working/cache_*_v31s.npz` | Auto-generated, speeds up re-runs. |

---

## Model & Methodology Summary

Our approach addresses several key challenges:

1.  **Feature staleness:** Autoregressive score lags dominate in cross-validation but
    collapse on the 91-day gap of the test set. The final model uses **0% score lag** and
    relies entirely on meteorological features.
2.  **Historical drift:** Training on the full 15-year history degraded performance. We
    restrict training to the most recent **8 years** per region (`RECENT_YEARS=8`).
3.  **Validation mismatch:** A random region split does not guarantee representative
    validation regions. We use a **Stratified Holdout Validation** split, dividing regions
    into 4 quartiles by long-term average score and sampling 20% from each.
4.  **Climatological normalization (SPI):** A per-region, per-month Standardized
    Precipitation Index measures how abnormal current rainfall/temperature is relative to
    that region's historical baseline.
5.  **Ensemble blending:** Predictions blend a 3-seed LightGBM ensemble with XGBoost and
    CatBoost.

For the complete experimental journey and the reasoning behind each dead end, see
[archiv/MODEL_HISTORY.md](archiv/MODEL_HISTORY.md) and the report
([report.tex](report.tex) / `DM_project_Group_9.pdf`).
