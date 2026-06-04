# Data Mining Final Project - Progress Log

> **Dokumentation Nr. 08** · [Lesereihenfolge](README.md) · laufendes Team-Tagebuch (nicht zwingend linear lesen)

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
**Action:** Wrote `06_EDA_ANALYSIS.md` – full interpretation of notebook results + full-dataset stats.
**Covers:** score rhythm (7-day), zero-inflation, correlations, lags/rolling, extremes, train-test shift, modeling recommendations.

## Step 8: Repository Restructure (May 19, 2026)
**Goal:** Cleaner layout for team collaboration.
**Action:**
- `docs/` – nummeriert `01_`–`08_`, siehe `docs/README.md`, presentation/
- `notebooks/` – numbered notebooks
- `scripts/` – create_sample.py
- `config/paths.py` – shared local/Colab paths
- `data/` – canonical data dir (symlink/legacy supported)
- `outputs/figures/` – EDA plots
- README.md, CONTRIBUTING.md

## Step 9: Test-Set Analysis (May 19, 2026)
**Goal:** Investigate suspicious values in `test.csv`.
**Findings:** Synthetic years (3020–58063), string-sort date trap, 91 rows/region, test always after train per region; 6.7k low surf_pre rows, 3 extreme cold rows; no nulls/duplicates/logic errors.
**Doc:** `docs/05_TEST_DATA_ANALYSIS.md`

## Step 10: Train Analysis Reference Doc (May 19, 2026)
**Action:** Created `docs/04_TRAIN_DATA_ANALYSIS.md` – canonical summary of all train findings (full + sample).
**Links:** `06_EDA_ANALYSIS.md` now points to Train/Test reference docs.

## Step 11: Local Chunked EDA Complete (May 19, 2026)
**Source:** `02_eda_analysis_local.ipynb` MODE=chunked, 62 chunks, ~2 min.
**Findings:** 12.3M rows validated; score_rate uniform 14.27%; regional score_mean 0.08–2.26; point corr tmp_range 0.17; 317 regions low pressure; 15-year window per region.
**Doc:** `docs/07_LOCAL_EDA_ANALYSIS.md`, `outputs/regional/region_summary.csv`

## Step 12: Weekly modeling layer in 04 (May 19, 2026)
**Problem:** `04_modeling` built sliding windows on *daily* labeled rows (~1.3M samples) → Colab OOM (11/12 GB).
**Why not fix in 03?** Preprocessing streams 12M rows; daily features/lags need the full panel. Kaggle targets are *weekly* (`pred_week1..5`).
**Solution:** `scripts/weekly_model.py` — `daily_to_weekly()` (`ordinal // 7`, last label per bucket), vectorized windows, single region mask for holdout. **03 unchanged.**
**Doc:** `docs/09_WEEKLY_MODELING.md`

## Step 13: Unified local + Colab env (May 19, 2026)
**Action:** `scripts/project_env.py` + `scripts/notebook_init.setup()` — same MODE/pipeline; only paths differ (repo `data/` vs Drive, `outputs/` vs Drive outputs). Notebooks 03/04 rewritten to one setup cell.

## Step 14: Preprocessing v2 (separate track) (May 19, 2026)
**Action:** `03b_preprocessing_v2.ipynb`, `features_v2.py`, `preprocess_streaming_v2.py`, `docs/10_PREPROCESSING_V2.md`, `04b_modeling_v2.ipynb` — v1 unchanged for ablation; v2 adds score lags, region stats, test91 aggregates.

## Step 15: Parallel 03b / 04b (May 19, 2026)
**Action:** `parallel_util.py`, region-parallel streaming v2, parallel sliding samples + 5-week LGBM; Colab `/content/` CSV copy in 03b.

**Current Task:** `git pull` → **03b** → **04b** → Kaggle `submission_full_v2.csv`.

## Step 16: Colab 04b — Drive, RAM, LightGBM, Kaggle-Ergebnis (May 20, 2026)

### Colab / Infrastruktur
- **`OSError [Errno 107]`** beim `read_parquet` von Drive: FUSE-Mount instabil → Parquet nach `/content/` kopieren oder Runtime neu starten; optional `read_parquet_notebook()` in `project_env.py` (nur 04/04b).
- **RAM-Crash** in `build_region_holdout`: ~1,76 Mio. Tageszeilen + `test_df` + Kopien → Colab-OOM. **03/03b unverändert**; Problem nur in 04/04b. Experiment mit frühem `train_weekly` / `DM_WORKERS=1` in `bootstrap` wieder verworfen (03b soll parallel bleiben).

### 04 vs. 04b — Pipeline vereinheitlicht
- **04** (v1) lief stabil und lieferte **Kaggle MAE ~0,9**.
- **04b** an **gleiche Zelllogik** wie 04 angeglichen (Holdout auf `train_df`, sequentielles LGBM, Final-Fit auf `X_all`).
- Modell-Upgrade in 04b nur dort: `LGB_PARAMS` (800 trees, lr 0,04, L1/L2), v2-Features, optional **`BLEND_PERSIST = 0.35`**.

### LightGBM 4.x (Colab Python 3.12)
- **`TypeError: Wrong type(ndarray) for label`**: kein `y[:, week]`-Slice; Labels als **`list`** via `y_week()` / `labels_for_week()` in `modeling_train.py`.
- **`eval_set`**: pro Woche `y_va[:, week]`, nicht ganzes 2D-`y_va`.
- Paralleles Wochen-Training in 04b durch **sequentielle Schleife wie 04** ersetzt (Colab-stabil).

### Kaggle-Ergebnis v2
| Run | Pipeline | Public MAE (ca.) |
|-----|----------|------------------|
| 1 | **03 + 04** (v1) | **~0,9** |
| 2 | **03b + 04b** (v2 + Blend) | **~1,1** (schlechter) |

**Hypothesen Verschlechterung:**
1. **`BLEND_PERSIST = 0.35`** mischt mit `last_score` vom **Trainingsende** — nicht zum Ende der 91 Test-Tage → zeitlich falsch für `pred_week1..5`.
2. v2-Modell evtl. Overfit (mehr Features, aggressiveres LGBM vs. 04).
3. Abgekürzter Final-Fit (`final_models = models` / weniger Bäume) verschlechtert Submission.

**Nächste Tests:** `BLEND_PERSIST = 0`; v2-Features + **04-Hyperparameter**; 03b-Daten checken (2.248 Regionen, ~204k Test-Zeilen).

### Code-Änderungen (Mai 20)
- `scripts/modeling_train.py` — `labels_for_week()`, `eval_set` pro Woche, `n_workers<=1` ohne ProcessPool.
- `scripts/weekly_model.py` — `already_weekly`, `float32` in `slim_for_modeling`, `labels_for_week`/`weekly_summary`.
- `scripts/project_env.py` — `read_parquet_notebook()` (optional Colab).
- `notebooks/04b_modeling_v2.ipynb` — Pipeline wie 04, `y_week()`, Blend optional.
- `docs/02_COLAB_SETUP.md` — RAM/Drive-Hinweise für 04/04b.

**Current Task:** 04b mit `BLEND_PERSIST=0` und/oder 04-Params erneut submitten; v1 (0,9) als Baseline behalten.

## Step 17: Colab-Erweiterung (Cursor/VS Code CLI) (May 20, 2026)
**Goal:** Notebooks lokal bearbeiten, Kernel auf Colab — ohne erzwungenen `git clone` weg vom lokalen Repo.
**Action:** `resolve_project_root()` / `resolve_outputs_dir()` in `project_env.py`; `notebook_init` nutzt Workspace zuerst; Setup-Zellen in 03/03b/04/04b vereinheitlicht; Doku in `docs/02_COLAB_SETUP.md`.
**Erkennung:** Setup-Print `Colab Extension / Workspace: Code aus …`
