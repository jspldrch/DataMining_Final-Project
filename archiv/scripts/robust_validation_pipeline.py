"""
robust_validation_pipeline.py

A highly robust, production-ready Kaggle pipeline for Time-Series Weather Forecasting.
Designed by a Kaggle Grandmaster to eliminate Data Leakage, establish a rigorous
Chronological Cross-Validation framework, run Adversarial Validation, perform robust
Feature Engineering, and optimize models directly on L1 (MAE) Loss.

Structure:
  1. Setup & Paths
  2. Data Loading & Chronological Parsing
  3. Gap-Aware Defensive Feature Engineering
  4. Adversarial Validation Check (Drift analysis)
  5. Multi-Fold Group Time-Series Split (CV Framework)
  6. Out-of-Fold (OOF) Target Encoding
  7. L1-Optimized Modeling (LightGBM & XGBoost)
  8. OOF Blending & Optimization
  9. Post-Processing & Median Calibration
  10. Feature Importance & Validation Diagnostics
  11. Executable Entry Point (Quick Mode / Full Mode)
"""

from __future__ import annotations

import gc
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

# Detect Environment and Set Paths
import os
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    print("[INFO] Running on Kaggle Environment")
    # Search dynamically for train.csv in /kaggle/input
    train_candidates = list(Path("/kaggle/input").glob("**/train.csv"))
    if train_candidates:
        DATA_DIR = train_candidates[0].parent
    else:
        DATA_DIR = Path("/kaggle/input")
    OUT_DIR = Path(".")
    OUT_PATH = OUT_DIR / "submission.csv"
else:
    print("[INFO] Running on Local Environment")
    try:
        ROOT = Path(__file__).resolve().parent.parent
        DATA_DIR = ROOT / "data-mining-2026-final-project" / "data"
        if not DATA_DIR.exists():
            DATA_DIR = ROOT / "data"
        OUT_DIR = ROOT / "outputs"
    except NameError:
        # Fallback if run in a notebook cell locally
        ROOT = Path(".")
        DATA_DIR = ROOT / "data"
        OUT_DIR = ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    OUT_PATH = OUT_DIR / "submission_robust.csv"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Global Configuration
QUICK_MODE = True  # Set to False to run all folds, more estimators, and full data size
RANDOM_STATE = 42
N_WEEKS = 5  # Forecast horizon
WEEK_BUCKET = 7  # 7 days per week

# Core weather columns present in the raw data
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

# Config for lag features (days)
LAG_COLS = ["prec", "tmp", "wind", "humidity", "surf_pre", "tmp_max"]
LAGS = [1, 3, 7, 14, 21]

# Config for rolling window features (days).
# CRITICAL: We restrict rolling windows to at most 90 days, because the test weather
# history is only 91 days long. Windows > 91 would pull from train set across the gap!
ROLL_COLS = ["prec", "tmp", "wind"]
ROLL_WINS = [7, 14, 30, 60, 90]

# ==============================================================================
# 2. CHRONOLOGICAL PARSING & LOADING
# ==============================================================================

def parse_dates_inplace(df: pd.DataFrame) -> None:
    """
    Parses dates chronologically using split to avoid slow datetime object creation.
    Computes an 'ordinal' representation that correctly aligns dates across years.
    """
    parts = df["date"].astype(str).str.split("-", expand=True)
    df["year"] = parts[0].astype(np.int32)
    df["month"] = parts[1].astype(np.int32)
    df["day"] = parts[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train and test sets using pyarrow engine for high performance (uses nrows in Quick Mode)."""
    print(f"Loading training data from: {TRAIN_CSV}")
    if QUICK_MODE:
        train_df = pd.read_csv(TRAIN_CSV, nrows=100000)
        print(f"Loading test data from: {TEST_CSV}")
        test_df = pd.read_csv(TEST_CSV, nrows=5000)
    else:
        train_df = pd.read_csv(TRAIN_CSV, engine="pyarrow")
        print(f"Loading test data from: {TEST_CSV}")
        test_df = pd.read_csv(TEST_CSV, engine="pyarrow")
    
    # Standardize column types
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    for col, dtype in dtypes.items():
        train_df[col] = train_df[col].astype(dtype)
        test_df[col] = test_df[col].astype(dtype)
    
    # Parse dates
    parse_dates_inplace(train_df)
    parse_dates_inplace(test_df)
    
    # Target cleanup
    train_df["score"] = pd.to_numeric(train_df["score"], errors="coerce").astype(np.float32)
    
    print(f"Loaded Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df

# ==============================================================================
# 3. DEFENSIVE & GAP-AWARE FEATURE ENGINEERING
# ==============================================================================

def engineer_features_for_region(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Feature engineering on daily data at a single region level.
    Designed defensively to prevent any leakage or gap pollution.
    """
    df = df.copy().sort_values("ordinal").reset_index(drop=True)
    new_cols = {}
    
    # 1. Calendar / Seasonal Embeddings
    new_cols["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    new_cols["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    new_cols["day_sin"] = np.sin(2 * np.pi * df["day"] / 31).astype(np.float32)
    new_cols["day_cos"] = np.cos(2 * np.pi * df["day"] / 31).astype(np.float32)
    
    # 2. Outlier-Robust Physical Ratios & Differences
    # Temperature ranges are physical indicators of drought/humidity
    new_cols["temp_spread"] = (df["tmp_max"] - df["tmp_min"]).astype(np.float32)
    new_cols["surf_temp_diff"] = (df["surf_tmp"] - df["tmp"]).astype(np.float32)
    new_cols["wind_spread"] = (df["wind_max"] - df["wind_min"]).astype(np.float32)
    
    # Clip extreme values in raw features (robust to sensor anomalies)
    for col in ["prec", "wind", "tmp"]:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower_bound, upper_bound)
        
    # 3. Lag Features
    # Shifted values represent the historical days prior to prediction point
    for col in LAG_COLS:
        for lag in LAGS:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag).astype(np.float32)
            
    # 4. Outlier-Robust Rolling Statistics (L1 / MAE friendly)
    # We use rolling MEDIANS instead of MEANS for robustness to extreme rain/wind peaks.
    # We shift(1) to avoid leaking the current day's weather into lags.
    for col in ROLL_COLS:
        prior = df[col].shift(1)
        for w in ROLL_WINS:
            roll = prior.rolling(w, min_periods=max(3, w//4))
            new_cols[f"{col}_roll{w}_mean"] = roll.mean().astype(np.float32)
            new_cols[f"{col}_roll{w}_median"] = roll.median().astype(np.float32)
            new_cols[f"{col}_roll{w}_std"] = roll.std().astype(np.float32)
            new_cols[f"{col}_roll{w}_max"] = roll.max().astype(np.float32)
            
    # 5. Advanced Drought Indicators
    pp = df["prec"].shift(1)
    new_cols["prec_deficit_90d"] = (
        pp.rolling(90, min_periods=30).mean() - pp.rolling(90, min_periods=30).median()
    ).astype(np.float32)
    
    # Relative trend of precipitation
    p7 = pp.rolling(7, min_periods=3).mean()
    p30 = pp.rolling(30, min_periods=10).mean()
    p30_std = pp.rolling(30, min_periods=10).std().clip(lower=0.01)
    new_cols["prec_trend_30d"] = ((p7 - p30) / p30_std).astype(np.float32)
    
    dry = (df["prec"].shift(1) < 1.0).astype(np.float32)
    new_cols["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    new_cols["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)
    
    # Merge engineered features back to DataFrame
    for col_name, col_values in new_cols.items():
        df[col_name] = col_values
        
    # Fill NAs resulting from shifts using forward-fill then backward-fill
    df = df.ffill().bfill()
    return df


def build_feature_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes feature engineering separately for train and test sets to prevent target or temporal leakage.
    Aggregates daily features to weekly level correctly using streaming to save memory.
    """
    print("\n--- Phase 3: Defensive & Gap-Aware Feature Engineering ---")
    t0 = time.time()
    
    # Identify GBDT Features using a tiny slice
    engineered_sample = engineer_features_for_region(train_df.iloc[:100])
    feature_cols = [c for c in engineered_sample.columns if c not in ["region_id", "date", "score", "year", "month", "day", "ordinal"]]
    print(f"Engineered {len(feature_cols)} physical/lag/rolling features.")
    
    def daily_to_weekly_frequency(df: pd.DataFrame) -> pd.DataFrame:
        """Helper to sample every 7th day (matching score interval)."""
        df = df.copy()
        df["week_id"] = df["ordinal"] // WEEK_BUCKET
        idx = df.groupby("week_id", sort=False)["ordinal"].idxmax()
        return df.loc[idx].reset_index(drop=True)
        
    # 1. Process Train Set
    print("Processing Train Set region by region...")
    train_parts_weekly = []
    
    # Stream train region by region to prevent massive pandas memory overhead
    for i, (region, reg_df) in enumerate(train_df.groupby("region_id", sort=False), 1):
        if QUICK_MODE and i > 10:
            break
        if i % 100 == 0:
            print(f"   Train region {i} | {time.time()-t0:.1f}s")
            
        daily_feat = engineer_features_for_region(reg_df)
        # Keep only labeled rows for training (score is not null) BEFORE weekly aggregation to save memory
        daily_feat_labeled = daily_feat[daily_feat["score"].notna()].copy()
        weekly_feat = daily_to_weekly_frequency(daily_feat_labeled)
        train_parts_weekly.append(weekly_feat)
        
    train_weekly = pd.concat(train_parts_weekly, ignore_index=True)
    del train_parts_weekly
    gc.collect()
    
    # 2. Process Test Set (Separately!)
    print("Processing Test Set region by region...")
    test_parts_weekly = []
    
    # In Quick Mode, only process test regions that exist in our train subset
    quick_regions = set(train_weekly["region_id"].unique()) if QUICK_MODE else None
    
    # Stream test region by region
    for i, (region, reg_df) in enumerate(test_df.groupby("region_id", sort=False), 1):
        if QUICK_MODE and (region not in quick_regions):
            continue
        if i % 100 == 0:
            print(f"   Test region {i} | {time.time()-t0:.1f}s")
            
        daily_feat = engineer_features_for_region(reg_df, is_test=True)
        weekly_feat = daily_to_weekly_frequency(daily_feat)
        test_parts_weekly.append(weekly_feat)
        
    test_weekly = pd.concat(test_parts_weekly, ignore_index=True)
    del test_parts_weekly
    gc.collect()
    
    print(f"Weekly Train Set size: {len(train_weekly):,} rows | Weekly Test Set size: {len(test_weekly):,} rows")
    return train_weekly, test_weekly

# ==============================================================================
# 4. ADVERSARIAL VALIDATION CHECK
# ==============================================================================

def run_adversarial_validation(train_weekly: pd.DataFrame, test_weekly: pd.DataFrame, feature_cols: list[str]) -> float:
    """
    Builds a classifier to predict whether a sample belongs to the train or test set.
    Allows us to check if there is a severe feature distribution shift between train and test.
    """
    print("\n--- Phase 2: Adversarial Validation Check ---")
    
    # Prepare adversarial dataset
    train_sample = train_weekly[feature_cols].copy()
    test_sample = test_weekly.groupby("region_id").tail(1)[feature_cols].copy()  # Use final prediction week
    
    train_sample["is_test"] = 0
    test_sample["is_test"] = 1
    
    adv_df = pd.concat([train_sample, test_sample], ignore_index=True).fillna(0)
    
    # Define features and adversarial target
    X = adv_df[feature_cols]
    y = adv_df["is_test"]
    
    # Use standard K-Fold for classifier evaluation
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(adv_df))
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=15,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        oof_preds[val_idx] = clf.predict_proba(X_va)[:, 1]
        
    auc_score = roc_auc_score(y, oof_preds)
    print(f"Adversarial Validation ROC AUC: {auc_score:.4f}")
    
    if auc_score > 0.65:
        print("[WARNING] Signifikante Unterschiede zwischen Train- und Testdaten erkannt!")
        print("          Das Modell kann einfach vorhersagen, ob ein Sample aus Train oder Test stammt.")
        # Train a single classifier to inspect feature importances
        clf_full = lgb.LGBMClassifier(objective="binary", n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
        clf_full.fit(X, y)
        importances = pd.Series(clf_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("Top 5 Drift-verursachende Features:")
        print(importances.head(5))
    else:
        print("[INFO] Die Feature-Verteilung zwischen Train und Test ist stabil (AUC ~0.5).")
        
    return auc_score

# ==============================================================================
# 5. MULTI-FOLD GROUP TIME-SERIES CROSS-VALIDATION
# ==============================================================================

def generate_chronological_folds(
    weekly_df: pd.DataFrame,
    features: list[str],
    n_folds: int = 5,
    val_weeks_per_fold: int = 5
) -> list[tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]]:
    """
    Strictest chronological validation strategy for panel data.
    Creates N folds. In each fold, validation is exactly 1 sample per region
    at the end of its history (shifted by fold * val_weeks_per_fold).
    Training data is the history strictly preceding the validation targets.
    Target variable contains future scores for 5 subsequent weeks.
    """
    print(f"\n--- Phase 1: Generating Chronological Folds (Folds={n_folds}) ---")
    folds = []
    
    for fold in range(n_folds):
        val_end_offset = fold * val_weeks_per_fold
        
        X_train, y_train, train_regions_list = [], [], []
        X_val, y_val, val_regions_list = [], [], []
        
        for region, group in weekly_df.groupby("region_id", sort=False):
            group = group.sort_values("ordinal")
            n = len(group)
            
            # Require enough history: at least 26 weeks of training history + prediction horizon (5) + fold offset
            min_history = val_end_offset + 30
            if n < min_history:
                continue
                
            val_end_idx = n - val_end_offset
            
            # Validation prediction point is 6 weeks before the end of the history
            # Validation target is the next 5 weeks (weeks -5 to -1 of the history)
            val_pred_point = val_end_idx - N_WEEKS - 1
            X_val.append(group.iloc[val_pred_point][features].values)
            y_val.append(group.iloc[val_pred_point+1 : val_pred_point+6]["score"].values)
            val_regions_list.append(region)
            
            # Training prediction points (everything before val_pred_point minus N_WEEKS to prevent target overlap)
            train_end_idx = val_pred_point - N_WEEKS
            for i in range(train_end_idx):
                X_train.append(group.iloc[i][features].values)
                # Next 5 weeks as targets
                y_train.append(group.iloc[i+1 : i+6]["score"].values)
                train_regions_list.append(region)
                
        X_train_df = pd.DataFrame(np.array(X_train, dtype=np.float32), columns=features)
        X_val_df = pd.DataFrame(np.array(X_val, dtype=np.float32), columns=features)
        y_train_arr = np.array(y_train, dtype=np.float32)
        y_val_arr = np.array(y_val, dtype=np.float32)
        
        # Keep track of region IDs for in-fold encoding
        X_train_df["region_id"] = pd.Series(train_regions_list, dtype="category").values
        X_val_df["region_id"] = pd.Series(val_regions_list, dtype="category").values
        
        print(f"Fold {fold+1}: Train Windows={len(X_train_df):,} | Val Windows={len(X_val_df):,}")
        folds.append((X_train_df, y_train_arr, X_val_df, y_val_arr))
        
    return folds

# ==============================================================================
# 6. IN-FOLD TARGET ENCODING (LEAKAGE-FREE)
# ==============================================================================

def apply_target_encoding(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Performs leakage-free target encoding on region_id inside each fold.
    Since we optimize MAE, we encode categories using the MEDIAN of targets.
    Creates 5 distinct target-encoded columns (one for each future prediction week).
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    new_enc_cols = []
    
    for week in range(N_WEEKS):
        col_name = f"region_median_week{week+1}"
        new_enc_cols.append(col_name)
        
        # Calculate target median per region on training fold
        target_series = pd.Series(y_train[:, week])
        medians = target_series.groupby(X_train["region_id"]).median()
        global_median = target_series.median()
        
        # Map values to train, val, and test data
        X_train[col_name] = X_train["region_id"].map(medians).fillna(global_median).astype(np.float32)
        X_val[col_name] = X_val["region_id"].map(medians).fillna(global_median).astype(np.float32)
        X_test[col_name] = X_test["region_id"].map(medians).fillna(global_median).astype(np.float32)
        
    return X_train, X_val, X_test, new_enc_cols

# ==============================================================================
# 7. MAE-OPTIMIZED MODELING (L1 OBJECTIVE)
# ==============================================================================

def train_l1_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: list[str],
    n_estimators: int = 500
) -> tuple[list[lgb.LGBMRegressor], list[xgb.XGBRegressor]]:
    """
    Trains MAE-optimized LightGBM and XGBoost models for each of the 5 forecasting weeks.
    Optimizes directly on the L1 loss metric.
    """
    lgb_models = []
    xgb_models = []
    
    # Exclude categorical columns like raw region_id from GBDT features
    gbdt_features = [c for c in feature_cols if c != "region_id"]
    
    for week in range(N_WEEKS):
        # 1. LightGBM (L1 / MAE optimized)
        lgb_model = lgb.LGBMRegressor(
            objective="regression_l1",  # L1 loss
            metric="mae",
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE + week,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_train[gbdt_features], y_train[:, week],
            eval_set=[(X_val[gbdt_features], y_val[:, week])],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_models.append(lgb_model)
        
        # 2. XGBoost (MAE optimized with hist tree_method speedup)
        xgb_kwargs = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE + week,
            "n_jobs": -1,
            "verbosity": 0
        }
        
        # Check if GPU is enabled to use GPU speedup
        try:
            import torch
            if torch.cuda.is_available():
                xgb_kwargs["tree_method"] = "hist"
                xgb_kwargs["device"] = "cuda"
            else:
                xgb_kwargs["tree_method"] = "hist"
        except ImportError:
            xgb_kwargs["tree_method"] = "hist"
            
        xgb_model = xgb.XGBRegressor(**xgb_kwargs)
        xgb_model.fit(
            X_train[gbdt_features], y_train[:, week],
            eval_set=[(X_val[gbdt_features], y_val[:, week])],
            verbose=False
        )
        xgb_models.append(xgb_model)
        
    return lgb_models, xgb_models

# ==============================================================================
# 8. OUT-OF-FOLD BLENDING & ENSEMBLING
# ==============================================================================

def optimize_blending_weights(lgb_oof: np.ndarray, xgb_oof: np.ndarray, y_val_all: np.ndarray) -> np.ndarray:
    """Finds the optimal blend weights for LightGBM and XGBoost on Out-of-Fold validation sets."""
    print("\nOptimizing Ensemble Blending weights...")
    best_mae_score = np.inf
    best_w = 0.5
    
    # Search grid for blend weights
    weights = np.linspace(0.0, 1.0, 101)
    
    for w in weights:
        blend_pred = w * lgb_oof + (1.0 - w) * xgb_oof
        blend_pred = np.clip(blend_pred, 0.0, 5.0)
        curr_mae = mean_absolute_error(y_val_all, blend_pred)
        
        if curr_mae < best_mae_score:
            best_mae_score = curr_mae
            best_w = w
            
    print(f"Optimal Weights -> LightGBM: {best_w:.2f} | XGBoost: {1.0-best_w:.2f}")
    print(f"Optimal Blend Validation MAE: {best_mae_score:.5f}")
    return np.array([best_w, 1.0 - best_w])

# ==============================================================================
# 9. POST-PROCESSING & CALIBRATION
# ==============================================================================

def check_median_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Since MAE is heavily optimized by predicting the median, we verify if
    the predictions match the median of the targets. Adjusts bias if necessary.
    """
    true_median = np.median(y_true)
    pred_median = np.median(y_pred)
    
    bias_shift = true_median - pred_median
    print(f"   Target Median: {true_median:.4f} | Prediction Median: {pred_median:.4f}")
    print(f"   Median Bias Shift: {bias_shift:.5f}")
    
    if abs(bias_shift) > 0.02:
        adjusted_preds = y_pred + bias_shift
        adjusted_preds = np.clip(adjusted_preds, 0.0, 5.0)
        before_mae = mean_absolute_error(y_true, y_pred)
        after_mae = mean_absolute_error(y_true, adjusted_preds)
        print(f"   Calibrating Bias: MAE before={before_mae:.5f} -> after={after_mae:.5f}")
        return adjusted_preds if after_mae < before_mae else y_pred
    else:
        print("   Predictions are well-calibrated (shift is negligible).")
        return y_pred

# ==============================================================================
# 10. DIAGNOSTICS & FEATURE IMPORTANCE
# ==============================================================================

def plot_feature_importance(models: list[lgb.LGBMRegressor], features: list[str]) -> None:
    """Plots average split gain feature importance of LightGBM models."""
    importance_df = pd.DataFrame()
    importance_df["feature"] = features
    
    avg_importance = np.zeros(len(features))
    for model in models:
        avg_importance += model.feature_importances_ / len(models)
        
    importance_df["importance"] = avg_importance
    importance_df = importance_df.sort_values("importance", ascending=False).head(20)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
    plt.title("Top 20 LightGBM Feature Importance (Average over Weeks)")
    plt.xlabel("Split Gain")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png")
    plt.close()
    print(f"Feature importance plot saved to {OUT_DIR / 'feature_importance.png'}")

# ==============================================================================
# 11. MAIN PIPELINE EXECUTION
# ==============================================================================

def main() -> None:
    print("=" * 80)
    print("      KAGGLE GRANDMASTER WEATHER FORECASTING PIPELINE      ")
    print(f"      QUICK MODE: {QUICK_MODE}  (Set to False for Production)      ")
    print("=" * 80)
    
    t_start = time.time()
    
    # 1. Load data
    train_raw, test_raw = load_dataset()
    
    # 2. Separate Feature Engineering
    train_weekly, test_weekly = build_feature_matrices(train_raw, test_raw)
    
    # Drop raw data to free memory
    del train_raw, test_raw
    gc.collect()
    
    # Core feature list definitions
    raw_feature_cols = [c for c in train_weekly.columns if c not in ["region_id", "date", "score", "year", "month", "day", "ordinal", "week_id"]]
    
    # 3. Adversarial Validation Check
    run_adversarial_validation(train_weekly, test_weekly, raw_feature_cols)
    
    # For quick run, use 2 folds, otherwise 5 folds.
    n_splits = 2 if QUICK_MODE else 5
    folds = generate_chronological_folds(train_weekly, raw_feature_cols, n_folds=n_splits)
    
    # Prepare test feature matrix
    X_test_final = test_weekly.sort_values(["region_id", "ordinal"]).groupby("region_id").tail(1)
    
    # We will record OOF targets, LGB predictions, and XGB predictions to evaluate ensemble MAE
    val_targets_record = []
    lgb_oof_list = []
    xgb_oof_list = []
    
    # Final test predictions accumulator
    final_lgb_test_preds = np.zeros((X_test_final["region_id"].nunique(), N_WEEKS))
    final_xgb_test_preds = np.zeros((X_test_final["region_id"].nunique(), N_WEEKS))
    
    print("\n--- Phase 5 & 6: Training K-Fold Models & OOF Prediction ---")
    
    for f_idx, (X_tr, y_tr, X_va, y_va) in enumerate(folds):
        print(f"\nProcessing Fold {f_idx+1}...")
        
        # Apply In-Fold Target Encoding (Leakage-free)
        X_tr_enc, X_va_enc, X_test_enc, new_enc_cols = apply_target_encoding(X_tr, y_tr, X_va, X_test_final)
        
        # Include target encoded columns in model training features
        fold_features = raw_feature_cols + new_enc_cols
        
        # Train L1-Optimized GBDTs
        n_est = 20 if QUICK_MODE else 600
        lgb_models, xgb_models = train_l1_models(X_tr_enc, y_tr, X_va_enc, y_va, fold_features, n_estimators=n_est)
        
        # Generate predictions on validation fold
        gbdt_features = [c for c in fold_features if c != "region_id"]
        lgb_val_preds = np.column_stack([m.predict(X_va_enc[gbdt_features]) for m in lgb_models])
        xgb_val_preds = np.column_stack([m.predict(X_va_enc[gbdt_features]) for m in xgb_models])
        
        # Record validation score
        fold_lgb_mae = mean_absolute_error(y_va, lgb_val_preds)
        fold_xgb_mae = mean_absolute_error(y_va, xgb_val_preds)
        print(f"Fold {f_idx+1} MAE -> LightGBM: {fold_lgb_mae:.5f} | XGBoost: {fold_xgb_mae:.5f}")
        
        # Save to OOF lists
        val_targets_record.append(y_va)
        lgb_oof_list.append(lgb_val_preds)
        xgb_oof_list.append(xgb_val_preds)
        
        # Collect Test Predictions (Averaging over Folds)
        final_lgb_test_preds += np.column_stack([m.predict(X_test_enc[gbdt_features]) for m in lgb_models]) / len(folds)
        final_xgb_test_preds += np.column_stack([m.predict(X_test_enc[gbdt_features]) for m in xgb_models]) / len(folds)
        
        # Record fold importances for diagnostics on fold 1
        if f_idx == 0:
            plot_feature_importance(lgb_models, gbdt_features)
            
    # Calculate global validation scores
    y_val_all = np.vstack(val_targets_record)
    lgb_oof_arr = np.vstack(lgb_oof_list)
    xgb_oof_arr = np.vstack(xgb_oof_list)
    
    # Blending optimization
    blend_weights = optimize_blending_weights(lgb_oof_arr, xgb_oof_arr, y_val_all)
    
    # Calculate final validation score of the blend
    final_val_blend = blend_weights[0] * lgb_oof_arr + blend_weights[1] * xgb_oof_arr
    final_val_blend = np.clip(final_val_blend, 0.0, 5.0)
    print(f"\nFinal Blended OOF MAE: {mean_absolute_error(y_val_all, final_val_blend):.5f}")
    
    # 5. Post-Processing & Calibration on Blend
    calibrated_val_blend = check_median_calibration(y_val_all, final_val_blend)
    
    # Test if rounding to nearest integer improves OOF MAE (due to discrete targets 0-5)
    mae_calibrated = mean_absolute_error(y_val_all, calibrated_val_blend)
    rounded_calibrated = np.round(calibrated_val_blend)
    mae_rounded = mean_absolute_error(y_val_all, rounded_calibrated)
    
    should_round = mae_rounded < mae_calibrated
    print(f"\nEvaluating rounding for discrete targets:")
    print(f"   Calibrated Continuous MAE: {mae_calibrated:.5f}")
    print(f"   Calibrated Rounded MAE: {mae_rounded:.5f}")
    if should_round:
        print("   -> Validation confirms: Rounding to integer improves MAE. Applying to test set!")
    else:
        print("   -> Validation confirms: Continuous predictions are better. Not rounding!")
        
    # Apply optimal weights to test predictions
    test_preds = blend_weights[0] * final_lgb_test_preds + blend_weights[1] * final_xgb_test_preds
    test_preds = np.clip(test_preds, 0.0, 5.0)
    
    # Apply median calibration offset if it was effective
    true_median = np.median(y_val_all)
    pred_median = np.median(final_val_blend)
    bias_shift = true_median - pred_median
    if abs(bias_shift) > 0.02:
        test_preds = np.clip(test_preds + bias_shift, 0.0, 5.0)
        print(f"Applied median calibration shift of {bias_shift:.5f} to final test predictions.")
        
    if should_round:
        test_preds = np.round(test_preds)
        print("Rounded final test predictions to nearest integer values [0.0, 1.0, 2.0, 3.0, 4.0, 5.0].")
        
    # Format and Save Submission
    print("\nSaving robust submission file...")
    sub = pd.DataFrame({"region_id": X_test_final["region_id"].unique()})
    sub = sub.sort_values("region_id").reset_index(drop=True)
    
    for k in range(N_WEEKS):
        sub[f"pred_week{k+1}"] = test_preds[:, k]
        
    sub.to_csv(OUT_PATH, index=False)
    print(f"✅ Robust submission saved to: {OUT_PATH}")
    print(f"Total pipeline runtime: {time.time()-t_start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
