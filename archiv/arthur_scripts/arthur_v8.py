"""
arthur_v8.py  –  Drought Severity Prediction (Arthur v8)

Base: run_v7.py (MAE 0.8303) – best Kaggle submission so far.

Changes from v7:
  1. surf_pre added to ROLL_COLS        — pressure trends (7d-180d) are strong
                                          meteorological signals (r=-0.11)
  2. Two-Stage Hurdle Model             — 58% of scores are 0 (zero-inflated!)
     - Stage 1: LGB binary classifiers   P(score > 0) per week
     - Stage 2: LGB/XGB/CatBoost         regressors (same as v7, all data)
     - Gating:  final = (α·P + (1-α)) × regression_pred
     - α tuned on validation:  α=0 → pure v7,  α=1 → full two-stage
     → If classifier doesn't help, α=0 and we get EXACTLY v7 output.
  3. Kaggle path auto-detection

Usage (local):    python arthur_scripts/arthur_v8.py
Usage (Kaggle):   Just run — paths auto-detected
Output:           submission_arthur_v8.csv
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─── Paths (auto-detect Kaggle vs local) ──────────────────────────────────────
KAGGLE_DIR = Path("/kaggle/input/datasets/axxtur/data-mining-2026-final-assignment")
LOCAL_ROOT = Path(__file__).resolve().parent.parent

if KAGGLE_DIR.exists():
    DATA_DIR   = KAGGLE_DIR / "data"
    SAMPLE_SUB = KAGGLE_DIR / "sample_submission.csv"
    OUT_DIR    = Path("/kaggle/working")
    IS_KAGGLE  = True
else:
    # Try multiple local paths
    for _cand in [LOCAL_ROOT / "data-mining-2026-final-project" / "data",
                  LOCAL_ROOT / "data"]:
        if (_cand / "train.csv").exists():
            DATA_DIR = _cand
            break
    else:
        DATA_DIR = LOCAL_ROOT / "data"

    for _cand in [LOCAL_ROOT / "data-mining-2026-final-project" / "sample_submission.csv",
                  LOCAL_ROOT / "resources" / "sample_submission.csv"]:
        if _cand.exists():
            SAMPLE_SUB = _cand
            break
    else:
        SAMPLE_SUB = LOCAL_ROOT / "resources" / "sample_submission.csv"

    OUT_DIR   = LOCAL_ROOT / "outputs"
    IS_KAGGLE = False

OUT_DIR.mkdir(exist_ok=True)
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"
OUT_PATH  = OUT_DIR / "submission_arthur_v8.csv"

# ─── Mode ─────────────────────────────────────────────────────────────────────
QUICK_MODE = False   # True = ~20 min;  False = full run

RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0

WINDOW_STRIDE = 1 if not QUICK_MODE else 4
N_ESTIMATORS  = 1000 if not QUICK_MODE else 400

# ─── Feature config ───────────────────────────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
# ── CHANGE 1: surf_pre added to ROLL_COLS (v7 only had it in LAG_COLS) ──
ROLL_COLS = ["prec", "wind", "tmp", "humidity", "surf_pre"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]

# ─── Model params ─────────────────────────────────────────────────────────────
LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    num_leaves=127,
    min_child_samples=60,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    max_depth=6,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    verbosity=0,
)
CAT_PARAMS = dict(
    iterations=N_ESTIMATORS,
    learning_rate=0.04,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=RANDOM_STATE,
    verbose=False,
    thread_count=-1,
)

# ── CHANGE 2: Classifier params for the two-stage hurdle model ──
CLF_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    num_leaves=63,            # simpler than regressor (binary is an easier task)
    min_child_samples=100,    # conservative to avoid overfitting on class boundary
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)

NUM_FEATURES: list[str] = []


# ─── Feature list ─────────────────────────────────────────────────────────────

def build_feature_list() -> list[str]:
    lag_names  = [f"{c}_lag{lag}"       for c in LAG_COLS  for lag in LAGS]
    roll_names = [
        f"{col}_roll{w}_{stat}"
        for col in ROLL_COLS
        for w in ROLL_WINS
        for stat in ("mean", "std", "max")
    ]
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos",
                "week_sin", "week_cos"]
    drought_indices = [
        "prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
        "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d",
    ]
    extra = ["regional_mean_score"]
    return WEATHER_COLS + lag_names + roll_names + calendar + drought_indices + extra


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_dates_inplace(df: pd.DataFrame) -> None:
    parts = df["date"].str.split("-", expand=True)
    df["year"]  = parts[0].astype(np.int32)
    df["month"] = parts[1].astype(np.int32)
    df["day"]   = parts[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]


def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} Min." if s >= 60 else f"{s:.0f}s"


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(y_pred, 0, 5) - y_true)))


def show_mae(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"  {name:<52s}  MAE = {mae(y_true, y_pred):.4f}")


# ─── Regional mean score ─────────────────────────────────────────────────────

def compute_regional_mean_score(train_raw: pd.DataFrame) -> pd.Series:
    return train_raw.groupby("region_id")["score"].mean()


def add_regional_mean_score(df: pd.DataFrame, region_means: pd.Series) -> pd.DataFrame:
    df["regional_mean_score"] = df["region_id"].map(region_means).astype(np.float32)
    return df


# ─── Feature engineering per region ──────────────────────────────────────────

def compute_region_features(
    tr: pd.DataFrame,
    te: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)

    new_cols: dict[str, np.ndarray] = {}

    # Calendar
    new_cols["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["day_sin"]   = np.sin(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    new_cols["day_cos"]   = np.cos(2 * np.pi * panel["day"]   / 31).astype(np.float32)

    week_of_year = (panel["ordinal"] // 7) % 52
    new_cols["week_sin"] = np.sin(2 * np.pi * week_of_year / 52).astype(np.float32)
    new_cols["week_cos"] = np.cos(2 * np.pi * week_of_year / 52).astype(np.float32)

    # Weather lags
    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS:
            new_cols[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)

    # Rolling features (shift(1) avoids look-ahead)
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            min_p = max(3, w // 10)
            r = prior.rolling(w, min_periods=min_p)
            new_cols[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            new_cols[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            new_cols[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)

    # Drought indices
    prec_prior = panel["prec"].shift(1)
    new_cols["prec_deficit_90d"] = (
        prec_prior.rolling(90, min_periods=30).mean()
        - prec_prior.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    p7   = prec_prior.rolling(7,  min_periods=3).mean()
    p30  = prec_prior.rolling(30, min_periods=10).mean()
    p30s = prec_prior.rolling(30, min_periods=10).std().clip(lower=0.01)
    new_cols["prec_trend_30d"] = ((p7 - p30) / p30s).astype(np.float32)

    hum_prior = panel["humidity"].shift(1)
    new_cols["humidity_deficit_90d"] = (
        hum_prior.rolling(90, min_periods=30).mean()
        - hum_prior.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    tmp_prior   = panel["tmp"].shift(1)
    tmp_anomaly = (
        tmp_prior.rolling(90,  min_periods=30).mean()
        - tmp_prior.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    new_cols["tmp_anomaly_90d"] = tmp_anomaly

    new_cols["heat_drought_idx"] = (
        new_cols["prec_deficit_90d"] * tmp_anomaly.clip(lower=0)
    ).astype(np.float32)

    dry = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    new_cols["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    new_cols["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)

    panel = pd.concat([panel, pd.DataFrame(new_cols, index=panel.index)], axis=1)

    n_tr = len(tr)
    return panel.iloc[:n_tr].copy(), panel.iloc[n_tr:].copy()


# ─── Dataset assembly ─────────────────────────────────────────────────────────

def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    week = df["ordinal"] // WEEK_BUCKET
    idx  = df.groupby(week, sort=False)["ordinal"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_sliding_windows(
    weekly: pd.DataFrame,
    skip_regions: set,
    num_features: list[str],
    stride: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    X_parts, y_parts, region_parts = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions:
            continue
        g = g.sort_values("ordinal")
        scores = g["score"].to_numpy(dtype=np.float32)
        X_num  = g[num_features].to_numpy(dtype=np.float32)
        n = len(g)
        if n < 6:
            continue
        n_win = n - 5
        y_reg = np.lib.stride_tricks.sliding_window_view(scores[1:], 5)[:n_win]
        idx   = list(range(0, n_win, stride))
        if (n_win - 1) not in idx:
            idx.append(n_win - 1)
        X_parts.append(X_num[idx])
        y_parts.append(y_reg[idx])
        region_parts.extend([region] * len(idx))

    X_df = pd.DataFrame(np.vstack(X_parts).astype(np.float32), columns=num_features)
    X_df["region_id"] = pd.Categorical(region_parts)
    return X_df, np.vstack(y_parts).astype(np.float32)


def build_val_samples(
    weekly: pd.DataFrame,
    val_regions: list,
    num_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    X_parts, y_parts, r_parts = [], [], []
    for region in val_regions:
        g = weekly.loc[weekly["region_id"] == region].sort_values("ordinal")
        if len(g) < 6:
            continue
        X_parts.append(g.iloc[-6][num_features].to_numpy(dtype=np.float32))
        y_parts.append(g.iloc[-5:]["score"].to_numpy(dtype=np.float32))
        r_parts.append(region)
    X_df = pd.DataFrame(np.vstack(X_parts), columns=num_features)
    X_df["region_id"] = pd.Categorical(r_parts)
    return X_df, np.vstack(y_parts)


# ─── Model training: Regressors (same as v7) ─────────────────────────────────

def train_lgb_models(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame | None, y_va: np.ndarray | None,
    n_trees_per_week: list[int] | None = None,
) -> list[lgb.LGBMRegressor]:
    models = []
    for week in range(5):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or LGB_PARAMS["n_estimators"]
        p = dict(LGB_PARAMS, random_state=RANDOM_STATE + week, n_estimators=n)
        m = lgb.LGBMRegressor(**p)
        fit_kw: dict = dict(categorical_feature=["region_id"])
        if X_va is not None:
            fit_kw["eval_set"] = [(X_va, y_va[:, week].ravel())]
            fit_kw["eval_metric"] = "mae"
            fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        m.fit(X_tr, y_tr[:, week].ravel(), **fit_kw)
        models.append(m)
    return models


def train_xgb_models(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame | None, y_va: np.ndarray | None,
    num_features: list[str],
    n_trees_per_week: list[int] | None = None,
) -> list[xgb.XGBRegressor]:
    X_tr_n = X_tr[num_features].to_numpy(dtype=np.float32)
    X_va_n = X_va[num_features].to_numpy(dtype=np.float32) if X_va is not None else None
    models = []
    for week in range(5):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or XGB_PARAMS["n_estimators"]
        p = dict(XGB_PARAMS, random_state=RANDOM_STATE + week, n_estimators=n)
        fit_kw: dict = {}
        if X_va_n is not None:
            p["early_stopping_rounds"] = 50
            fit_kw["eval_set"] = [(X_va_n, y_va[:, week].ravel())]
            fit_kw["verbose"] = False
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr_n, y_tr[:, week].ravel(), **fit_kw)
        models.append(m)
    return models


def train_cat_models(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame | None, y_va: np.ndarray | None,
    num_features: list[str],
    n_trees_per_week: list[int] | None = None,
) -> list | None:
    if not CATBOOST_AVAILABLE:
        return None
    X_tr_n = X_tr[num_features].to_numpy(dtype=np.float32)
    X_va_n = X_va[num_features].to_numpy(dtype=np.float32) if X_va is not None else None
    models = []
    for week in range(5):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or CAT_PARAMS["iterations"]
        p = dict(CAT_PARAMS, iterations=n, random_seed=RANDOM_STATE + week)
        fit_kw: dict = {}
        if X_va_n is not None:
            fit_kw["eval_set"] = (X_va_n, y_va[:, week].ravel())
            fit_kw["early_stopping_rounds"] = 50
        m = CatBoostRegressor(**p)
        m.fit(X_tr_n, y_tr[:, week].ravel(), **fit_kw)
        models.append(m)
    return models


# ─── Model training: Classifiers (NEW — two-stage hurdle) ────────────────────

def train_lgb_classifiers(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame | None, y_va: np.ndarray | None,
    n_trees_per_week: list[int] | None = None,
) -> list[lgb.LGBMClassifier]:
    """Train 5 binary classifiers: P(score > 0) per prediction week."""
    models = []
    for week in range(5):
        y_bin = (y_tr[:, week] > 0).astype(np.int32)
        n = (n_trees_per_week[week] if n_trees_per_week else None) or CLF_PARAMS["n_estimators"]
        p = dict(CLF_PARAMS, random_state=RANDOM_STATE + week + 100, n_estimators=n)
        m = lgb.LGBMClassifier(**p)
        fit_kw: dict = dict(categorical_feature=["region_id"])
        if X_va is not None:
            y_va_bin = (y_va[:, week] > 0).astype(np.int32)
            fit_kw["eval_set"] = [(X_va, y_va_bin)]
            fit_kw["eval_metric"] = "binary_logloss"
            fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        m.fit(X_tr, y_bin, **fit_kw)
        models.append(m)
    return models


# ─── Prediction functions ────────────────────────────────────────────────────

def predict_lgb(models: list, X: pd.DataFrame) -> np.ndarray:
    return np.clip(np.column_stack([m.predict(X) for m in models]), 0.0, 5.0).astype(np.float32)


def predict_xgb(models: list, X: pd.DataFrame, num_features: list[str]) -> np.ndarray:
    X_n = X[num_features].to_numpy(dtype=np.float32)
    return np.clip(np.column_stack([m.predict(X_n) for m in models]), 0.0, 5.0).astype(np.float32)


def predict_cat(models: list | None, X: pd.DataFrame, num_features: list[str]) -> np.ndarray | None:
    if models is None:
        return None
    X_n = X[num_features].to_numpy(dtype=np.float32)
    return np.clip(np.column_stack([m.predict(X_n) for m in models]), 0.0, 5.0).astype(np.float32)


def predict_clf_proba(models: list, X: pd.DataFrame) -> np.ndarray:
    """Get P(score > 0) from classifier ensemble. Shape: (n_samples, 5)."""
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models]).astype(np.float32)


def _cat_best_iter(m, default: int) -> int:
    try:
        bi = m.get_best_iteration()
        return int(bi) if bi is not None else default
    except Exception:
        return default


# ─── Blend + Two-Stage optimisation ──────────────────────────────────────────

def optimize_blend(
    y_va: np.ndarray,
    lgb_val: np.ndarray,
    xgb_val: np.ndarray,
    cat_val: np.ndarray | None = None,
) -> tuple[tuple[float, float, float], float]:
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_mae_v, best_weights = 999.0, (0.5, 0.5, 0.0)
    if cat_val is not None:
        for a in alphas:
            for b in alphas:
                c = round(1.0 - a - b, 8)
                if c < 0:
                    continue
                m = mae(y_va, a * lgb_val + b * xgb_val + c * cat_val)
                if m < best_mae_v:
                    best_mae_v, best_weights = m, (a, b, c)
    else:
        for a in alphas:
            m = mae(y_va, a * lgb_val + (1 - a) * xgb_val)
            if m < best_mae_v:
                best_mae_v, best_weights = m, (a, 1 - a, 0.0)
    return best_weights, best_mae_v


def optimize_hurdle_alpha(
    y_va: np.ndarray,
    blend_val: np.ndarray,
    clf_proba_val: np.ndarray,
) -> tuple[float, float]:
    """
    Find best α for:  final = (α·P(score>0) + (1-α)) × blend_regression

    α = 0  →  pure regression (identical to v7)
    α = 1  →  full hurdle gating
    
    Safe: if classifier doesn't help, α stays at 0.
    """
    best_alpha = 0.0
    best_mae_v = mae(y_va, blend_val)  # baseline = pure regression (α=0)

    for alpha in np.arange(0.05, 1.01, 0.05):
        gate = alpha * clf_proba_val + (1.0 - alpha)
        dampened = np.clip(gate * blend_val, 0.0, 5.0)
        m = mae(y_va, dampened)
        if m < best_mae_v:
            best_alpha, best_mae_v = float(alpha), m

    return best_alpha, best_mae_v


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 70)
    print("  Drought Severity Prediction  –  arthur_v8.py")
    mode_label = "QUICK" if QUICK_MODE else "FULL"
    env_label  = "KAGGLE" if IS_KAGGLE else "LOCAL"
    print(f"  Mode: {mode_label}  |  Env: {env_label}  |  stride={WINDOW_STRIDE}  est={N_ESTIMATORS}")
    cat_label = "ON" if CATBOOST_AVAILABLE else "OFF  (pip install catboost)"
    print(f"  CatBoost: {cat_label}  |  Features: {len(NUM_FEATURES)}")
    print(f"  Changes vs v7: +surf_pre rolls, +two-stage hurdle model")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading CSV files from {DATA_DIR} ...")
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw  = pd.read_csv(TEST_CSV,  dtype=dtypes)
    _parse_dates_inplace(train_raw)
    _parse_dates_inplace(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    print(f"   Train: {len(train_raw):>10,} rows  |  Test: {len(test_raw):>8,} rows")
    print(f"   Regions: {len(regions)}  |  [{elapsed(t0)}]")

    # Score distribution info
    labeled = train_raw["score"].notna()
    n_labeled = labeled.sum()
    n_zero = (train_raw.loc[labeled, "score"] == 0).sum()
    print(f"   Labeled: {n_labeled:,} ({100*n_labeled/len(train_raw):.1f}%)  |  "
          f"Zero-scores: {n_zero:,} ({100*n_zero/n_labeled:.1f}%)")

    region_means = compute_regional_mean_score(train_raw)

    # ── 2. Feature engineering per region ─────────────────────────────────────
    print(f"\n[2/7] Feature engineering per region ...")
    train_by_region = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    test_by_region  = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw

    all_tr_feat, all_te_feat = [], []
    n = len(regions)
    for i, region in enumerate(regions, 1):
        if i % 500 == 0 or i == n:
            print(f"   Region {i}/{n}  |  [{elapsed(t0)}]")
        tr = train_by_region[region]
        te = test_by_region.get(region, pd.DataFrame())
        tr_f, te_f = compute_region_features(tr, te)
        all_tr_feat.append(tr_f)
        all_te_feat.append(te_f)

    train_feat = pd.concat(all_tr_feat, ignore_index=True)
    test_feat  = pd.concat(all_te_feat, ignore_index=True)
    del all_tr_feat, all_te_feat

    train_feat = add_regional_mean_score(train_feat, region_means)
    test_feat  = add_regional_mean_score(test_feat,  region_means)
    print(f"   Done  |  [{elapsed(t0)}]")

    # ── 3. Weekly aggregation ─────────────────────────────────────────────────
    print(f"\n[3/7] Weekly aggregation ...")
    labeled_df = train_feat[train_feat["score"].notna()].copy()
    weekly_parts = []
    for region, g in labeled_df.groupby("region_id", sort=False):
        weekly_parts.append(daily_to_weekly(g))
    train_weekly = pd.concat(weekly_parts, ignore_index=True)
    del labeled_df
    weeks_per_region = int(len(train_weekly) / len(regions))
    print(f"   {len(train_weekly):,} weekly rows  (~{weeks_per_region}/region)  [{elapsed(t0)}]")

    # ── 4. Train/val split (region holdout 20%) ───────────────────────────────
    print(f"\n[4/7] Building train/val split ...")
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg = sorted(train_weekly["region_id"].unique())
    n_val   = max(1, int(len(all_reg) * VAL_REGION_FRAC))
    val_regions = set(rng.choice(all_reg, size=n_val, replace=False))

    X_tr, y_tr = build_sliding_windows(train_weekly, val_regions, NUM_FEATURES, stride=WINDOW_STRIDE)
    X_va, y_va = build_val_samples(train_weekly, sorted(val_regions), NUM_FEATURES)
    print(f"   Train windows: {len(X_tr):,}  |  Val regions: {len(val_regions)}")

    last_score = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    persist_va = np.column_stack([
        last_score.reindex(sorted(val_regions)).fillna(0).to_numpy() for _ in range(5)
    ])
    show_mae("Persistence-Baseline", y_va, persist_va)

    # ── 5. Train regressors (same as v7) ──────────────────────────────────────
    print(f"\n[5/7] Training regressors ...")
    print("       LightGBM ...")
    lgb_models = train_lgb_models(X_tr, y_tr, X_va, y_va)
    lgb_val = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (val)", y_va, lgb_val)

    print("       XGBoost ...")
    xgb_models = train_xgb_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
    xgb_val = predict_xgb(xgb_models, X_va, NUM_FEATURES)
    show_mae("XGBoost (val)", y_va, xgb_val)

    cat_val = None
    cat_models_val = None
    if CATBOOST_AVAILABLE:
        print("       CatBoost ...")
        cat_models_val = train_cat_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
        cat_val = predict_cat(cat_models_val, X_va, NUM_FEATURES)
        show_mae("CatBoost (val)", y_va, cat_val)

    print("\n  Blend optimisation:")
    best_weights, best_mae_val = optimize_blend(y_va, lgb_val, xgb_val, cat_val)
    lgb_w, xgb_w, cat_w = best_weights
    if cat_val is not None:
        blend_val = lgb_w * lgb_val + xgb_w * xgb_val + cat_w * cat_val
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}  CAT={cat_w:.2f}   MAE={best_mae_val:.4f}")
    else:
        blend_val = lgb_w * lgb_val + xgb_w * xgb_val
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}   MAE={best_mae_val:.4f}")
    show_mae("Ensemble (val, no hurdle)", y_va, blend_val)

    # ── 6. Two-stage hurdle model (NEW) ───────────────────────────────────────
    print(f"\n[6/7] Two-stage hurdle model ...")
    print("       Training classifiers P(score > 0) ...")
    clf_models = train_lgb_classifiers(X_tr, y_tr, X_va, y_va)
    clf_proba_val = predict_clf_proba(clf_models, X_va)

    # Show classifier accuracy
    for wk in range(5):
        y_bin = (y_va[:, wk] > 0).astype(int)
        p_bin = (clf_proba_val[:, wk] > 0.5).astype(int)
        acc = (y_bin == p_bin).mean()
        print(f"   Classifier week {wk+1}: accuracy={acc:.3f}  "
              f"P(>0) mean={clf_proba_val[:, wk].mean():.3f}")

    # Find best alpha
    best_alpha, best_hurdle_mae = optimize_hurdle_alpha(y_va, blend_val, clf_proba_val)
    print(f"\n  Hurdle α optimisation:")
    print(f"   Best α = {best_alpha:.2f}  |  MAE = {best_hurdle_mae:.4f}  "
          f"(vs {best_mae_val:.4f} without hurdle)")

    if best_alpha > 0:
        improvement = best_mae_val - best_hurdle_mae
        print(f"   → Hurdle HELPS: Δ MAE = -{improvement:.4f}")
        gate_val = best_alpha * clf_proba_val + (1.0 - best_alpha)
        final_val = np.clip(gate_val * blend_val, 0.0, 5.0)
    else:
        print(f"   → Hurdle does NOT help, using pure regression (same as v7)")
        final_val = blend_val

    show_mae("Final ensemble (val)", y_va, final_val)

    # ── 7. Final training on all data + test predictions ──────────────────────
    print(f"\n[7/7] Final training (all regions) + test predictions ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE)

    # Regressors
    n_lgb_trees = [int(getattr(m, "best_iteration_", None) or LGB_PARAMS["n_estimators"]) for m in lgb_models]
    n_xgb_trees = [int(getattr(m, "best_iteration",  None) or XGB_PARAMS["n_estimators"]) for m in xgb_models]

    final_lgb = train_lgb_models(X_all, y_all, None, None, n_lgb_trees)
    final_xgb = train_xgb_models(X_all, y_all, None, None, NUM_FEATURES, n_xgb_trees)

    final_cat = None
    if CATBOOST_AVAILABLE and cat_models_val is not None:
        n_cat_trees = [_cat_best_iter(m, CAT_PARAMS["iterations"]) for m in cat_models_val]
        final_cat = train_cat_models(X_all, y_all, None, None, NUM_FEATURES, n_cat_trees)

    # Classifiers (retrain on all data if alpha > 0)
    final_clf = None
    if best_alpha > 0:
        n_clf_trees = [int(getattr(m, "best_iteration_", None) or CLF_PARAMS["n_estimators"]) for m in clf_models]
        final_clf = train_lgb_classifiers(X_all, y_all, None, None, n_clf_trees)

    print(f"   Models trained  |  [{elapsed(t0)}]")

    # Test predictions
    X_test = (
        test_feat.sort_values(["region_id", "ordinal"])
        .groupby("region_id", sort=False)
        .tail(1)[["region_id"] + NUM_FEATURES]
        .reset_index(drop=True)
    )
    X_test["region_id"] = X_test["region_id"].astype("category")

    lgb_test = predict_lgb(final_lgb, X_test)
    xgb_test = predict_xgb(final_xgb, X_test, NUM_FEATURES)

    if final_cat is not None:
        cat_test   = predict_cat(final_cat, X_test, NUM_FEATURES)
        test_preds = lgb_w * lgb_test + xgb_w * xgb_test + cat_w * cat_test
    else:
        test_preds = lgb_w * lgb_test + xgb_w * xgb_test

    # Apply hurdle gating if alpha > 0
    if best_alpha > 0 and final_clf is not None:
        clf_proba_test = predict_clf_proba(final_clf, X_test)
        gate_test = best_alpha * clf_proba_test + (1.0 - best_alpha)
        test_preds = np.clip(gate_test * test_preds, 0.0, 5.0)
        print(f"   Hurdle applied (α={best_alpha:.2f})  |  [{elapsed(t0)}]")
    else:
        print(f"   No hurdle (α=0)  |  [{elapsed(t0)}]")

    # Build submission
    sub = pd.DataFrame({"region_id": X_test["region_id"].values})
    for k in range(5):
        sub[f"pred_week{k+1}"] = test_preds[:, k]

    try:
        template = pd.read_csv(SAMPLE_SUB)
        sub = template[["region_id"]].merge(sub, on="region_id", how="left")
        for col in [f"pred_week{k+1}" for k in range(5)]:
            sub[col] = sub[col].fillna(0.0)
    except Exception as e:
        print(f"   [WARN] Could not load SAMPLE_SUB for merging ({e}). Saving direct predictions instead.")

    sub.to_csv(OUT_PATH, index=False)

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  Saved: {OUT_PATH}")
    print(f"  Rows: {len(sub):,}  |  Total: {total_min:.1f} Min.")
    print(f"  Changes vs v7: +surf_pre rolls (+18 features), +hurdle (α={best_alpha:.2f})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
