"""
run_v11.py  –  Drought Severity Prediction v11

Base: run_v7 features (MAE 0.8303).

Root cause of stagnation identified:
  Region holdout validation had val MAE ~0.28 but Kaggle MAE ~0.83 (3x gap).
  The val set tested cross-region generalisation within the training period (easy).
  The Kaggle test needs temporal forecasting: predict future weeks for all regions (hard).
  This means every hyperparameter decision since v7 was guided by a misleading signal.

Fix 1 – Temporal validation (main fix):
  Instead of holding out 20% of regions, hold out the LAST 5 WEEKS of ALL regions.
  Training:    all sliding windows with targets ending before the last 5 weeks.
  Validation:  one window per region – feature at week T-6, targets at T-5..T-1.
  Effect:      val set now mirrors the Kaggle test scenario (predict future weeks).
               Training uses ALL 2248 regions → more data than region holdout.
               Val has 2248 samples (vs 449 before) → more stable estimate.

Fix 2 – Regional climatology z-scores (carefully implemented):
  v6 used z-scores BUT also changed objective to regression_l1 (which hurt).
  Here: z-scores added on top of v7 (L2 objective kept). NaN handled with global fallback.

Fix 3 – Balanced blend (min weight 0.15 per model):
  v10 had LGB=0.05 / XGB=0.95 because the misleading val favoured XGB.
  With temporal val, blend weights should be more realistic.
  Clip minimum weight to 0.15 to prevent extreme single-model dominance.

v7 features + hyperparameters otherwise unchanged.

Usage:
    python scripts/run_v11.py
Output: outputs/submission_v11.csv
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

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV  = DATA_DIR / "train.csv"
TEST_CSV   = DATA_DIR / "test.csv"
SAMPLE_SUB = ROOT / "resources" / "sample_submission.csv"
OUT_PATH   = OUT_DIR / "submission_v11.csv"

# ─── Mode ─────────────────────────────────────────────────────────────────────
QUICK_MODE = False

RANDOM_STATE    = 42
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0

WINDOW_STRIDE = 1 if not QUICK_MODE else 4
N_ESTIMATORS  = 1000 if not QUICK_MODE else 400

# ─── Feature config (v7 base + z-scores) ──────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp", "humidity"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]
CLIM_COLS = ["prec", "tmp", "humidity", "wind", "surf_pre", "tmp_max", "tmp_min"]

# ─── Model params (identical to v7) ───────────────────────────────────────────
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

NUM_FEATURES: list[str] = []


# ─── Feature list ─────────────────────────────────────────────────────────────

def build_feature_list() -> list[str]:
    lag_names  = [f"{c}_lag{lag}"      for c in LAG_COLS  for lag in LAGS]
    roll_names = [
        f"{col}_roll{w}_{stat}"
        for col in ROLL_COLS for w in ROLL_WINS for stat in ("mean", "std", "max")
    ]
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos", "week_sin", "week_cos"]
    drought  = [
        "prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
        "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d",
    ]
    zscore_names = [f"{c}_zscore" for c in CLIM_COLS]
    extra = ["regional_mean_score"]
    return WEATHER_COLS + lag_names + roll_names + calendar + drought + zscore_names + extra


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


# ─── Climatology z-scores ─────────────────────────────────────────────────────

def compute_climatology(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Per (region_id, month) mean+std for CLIM_COLS from training data only."""
    stats = train_raw.groupby(["region_id", "month"])[CLIM_COLS].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    return stats.reset_index()


def add_zscore_features(df: pd.DataFrame, clim: pd.DataFrame, global_means: dict, global_stds: dict) -> pd.DataFrame:
    """Z-score = (value - regional_monthly_mean) / regional_monthly_std.
    Falls back to global mean/std for unseen (region, month) pairs."""
    merged = df[["region_id", "month"]].merge(clim, on=["region_id", "month"], how="left")
    new_cols: dict = {}
    for col in CLIM_COLS:
        mean_v = merged[f"{col}_mean"].fillna(global_means[col]).values
        std_v  = merged[f"{col}_std"].fillna(global_stds[col]).values
        std_v  = np.where(std_v < 1e-8, 1.0, std_v)
        new_cols[f"{col}_zscore"] = ((df[col].values - mean_v) / std_v).astype(np.float32)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ─── Feature engineering per region (v7 identical) ────────────────────────────

def compute_region_features(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    new_cols: dict[str, np.ndarray] = {}

    new_cols["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["day_sin"]   = np.sin(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    new_cols["day_cos"]   = np.cos(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    week_of_year = (panel["ordinal"] // 7) % 52
    new_cols["week_sin"]  = np.sin(2 * np.pi * week_of_year / 52).astype(np.float32)
    new_cols["week_cos"]  = np.cos(2 * np.pi * week_of_year / 52).astype(np.float32)

    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS:
            new_cols[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)

    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=max(3, w // 10))
            new_cols[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            new_cols[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            new_cols[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)

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
    new_cols["heat_drought_idx"] = (new_cols["prec_deficit_90d"] * tmp_anomaly.clip(lower=0)).astype(np.float32)

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


def build_temporal_split(
    weekly: pd.DataFrame,
    num_features: list[str],
    stride: int = 1,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Temporal split matching the Kaggle test scenario.

    Training:   all sliding windows with targets ending before the last 5 weeks.
    Validation: one window per region — feature at week T-6, targets at T-5..T-1.

    vs region holdout (old):
      old val MAE ~0.28 (cross-region, easy, not representative of Kaggle)
      new val MAE ~0.8x (temporal, hard, mirrors actual test setup)
    """
    X_tr_parts, y_tr_parts, r_tr_parts = [], [], []
    X_va_parts, y_va_parts, r_va_parts = [], [], []

    for region, g in weekly.groupby("region_id", sort=False):
        g = g.sort_values("ordinal")
        scores = g["score"].to_numpy(dtype=np.float32)
        X_num  = g[num_features].to_numpy(dtype=np.float32)
        n = len(g)
        if n < 11:
            continue

        # y_reg[i] = scores[i+1..i+5]  (shape n-5, 5)
        y_reg = np.lib.stride_tricks.sliding_window_view(scores[1:], 5)

        # Training: feature indices 0..n-11 (targets end at row n-6, before val territory)
        n_win_tr = n - 10
        idx = list(range(0, n_win_tr, stride))
        if (n_win_tr - 1) not in idx:
            idx.append(n_win_tr - 1)
        X_tr_parts.append(X_num[idx])
        y_tr_parts.append(y_reg[idx])
        r_tr_parts.extend([region] * len(idx))

        # Validation: feature at n-6, targets = scores[n-5..n-1]
        X_va_parts.append(X_num[n - 6: n - 5])
        y_va_parts.append(y_reg[n - 6: n - 5])
        r_va_parts.append(region)

    X_tr = pd.DataFrame(np.vstack(X_tr_parts).astype(np.float32), columns=num_features)
    X_tr["region_id"] = pd.Categorical(r_tr_parts)
    X_va = pd.DataFrame(np.vstack(X_va_parts).astype(np.float32), columns=num_features)
    X_va["region_id"] = pd.Categorical(r_va_parts)
    return X_tr, np.vstack(y_tr_parts).astype(np.float32), X_va, np.vstack(y_va_parts).astype(np.float32)


def build_all_windows(
    weekly: pd.DataFrame,
    num_features: list[str],
    stride: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """All sliding windows for final training (no val holdout)."""
    X_parts, y_parts, r_parts = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
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
        r_parts.extend([region] * len(idx))
    X_df = pd.DataFrame(np.vstack(X_parts).astype(np.float32), columns=num_features)
    X_df["region_id"] = pd.Categorical(r_parts)
    return X_df, np.vstack(y_parts).astype(np.float32)


# ─── Model training ───────────────────────────────────────────────────────────

def train_lgb_models(X_tr, y_tr, X_va, y_va, n_trees_per_week=None):
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


def train_xgb_models(X_tr, y_tr, X_va, y_va, num_features, n_trees_per_week=None):
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


def train_cat_models(X_tr, y_tr, X_va, y_va, num_features, n_trees_per_week=None):
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


def predict_lgb(models, X):
    return np.clip(np.column_stack([m.predict(X) for m in models]), 0.0, 5.0).astype(np.float32)


def predict_xgb(models, X, num_features):
    return np.clip(np.column_stack([m.predict(X[num_features].to_numpy(dtype=np.float32)) for m in models]), 0.0, 5.0).astype(np.float32)


def predict_cat(models, X, num_features):
    if models is None:
        return None
    return np.clip(np.column_stack([m.predict(X[num_features].to_numpy(dtype=np.float32)) for m in models]), 0.0, 5.0).astype(np.float32)


def _best_n(m, default):
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(m, attr, None)
        if v is not None:
            return int(v)
    try:
        v = m.get_best_iteration()
        if v is not None:
            return int(v)
    except Exception:
        pass
    return default


def optimize_blend(y_va, lgb_val, xgb_val, cat_val=None):
    """Grid search blend weights. Minimum weight per model = 0.15 (prevents extreme values)."""
    alphas = [round(x * 0.05, 2) for x in range(3, 18)]  # 0.15 .. 0.85 step 0.05
    best_mae_v, best_weights = 999.0, (0.5, 0.5, 0.0)
    if cat_val is not None:
        for a in alphas:
            for b in alphas:
                c = round(1.0 - a - b, 8)
                if c < 0.15:
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


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 68)
    print("  Natural Disaster Severity Prediction  -  run_v11.py")
    mode_label = "QUICK (~20 min)" if QUICK_MODE else "FULL (~90 min)"
    print(f"  Mode: {mode_label}  |  stride={WINDOW_STRIDE}  estimators={N_ESTIMATORS}")
    cat_label = "ON" if CATBOOST_AVAILABLE else "OFF"
    print(f"  CatBoost: {cat_label}  |  Features: {len(NUM_FEATURES)}")
    print(f"  Val: TEMPORAL (last 5 weeks all regions) — not region holdout")
    print("=" * 68)

    # 1. Load data
    print("\n[1/6] Loading CSV files ...")
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw  = pd.read_csv(TEST_CSV,  dtype=dtypes)
    _parse_dates_inplace(train_raw)
    _parse_dates_inplace(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    print(f"   Train: {len(train_raw):>10,} rows  |  Test: {len(test_raw):>8,} rows")
    print(f"   Regions: {len(regions)}  |  [{elapsed(t0)}]")

    # Compute climatology and regional means before deleting train_raw
    print("\n[2/6] Computing climatology + regional stats ...")
    clim_df = compute_climatology(train_raw)
    global_means = {c: float(train_raw[c].mean()) for c in CLIM_COLS}
    global_stds  = {c: max(float(train_raw[c].std()), 1e-8) for c in CLIM_COLS}
    region_means = train_raw.groupby("region_id")["score"].mean()
    print(f"   Climatology: {len(clim_df):,} (region, month) entries  [{elapsed(t0)}]")

    # 3. Feature engineering per region
    print("\n[3/6] Feature engineering per region ...")
    train_by_region = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    test_by_region  = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw

    all_tr, all_te = [], []
    n = len(regions)
    for i, region in enumerate(regions, 1):
        if i % 500 == 0 or i == n:
            print(f"   Region {i}/{n}  |  [{elapsed(t0)}]")
        tr_f, te_f = compute_region_features(train_by_region[region],
                                              test_by_region.get(region, pd.DataFrame()))
        all_tr.append(tr_f)
        all_te.append(te_f)

    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te

    # Add z-scores (train climatology applied to both — no leakage)
    train_feat = add_zscore_features(train_feat, clim_df, global_means, global_stds)
    test_feat  = add_zscore_features(test_feat,  clim_df, global_means, global_stds)
    del clim_df

    # Add regional mean score
    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)
    print(f"   Done  |  [{elapsed(t0)}]")

    # 4. Weekly aggregation
    print("\n[4/6] Weekly aggregation ...")
    labeled = train_feat[train_feat["score"].notna()].copy()
    weekly_parts = []
    for region, g in labeled.groupby("region_id", sort=False):
        weekly_parts.append(daily_to_weekly(g))
    train_weekly = pd.concat(weekly_parts, ignore_index=True)
    del labeled
    print(f"   {len(train_weekly):,} weekly rows  |  [{elapsed(t0)}]")

    # 5. Temporal split
    print("\n[5/6] Building temporal train/val split ...")
    X_tr, y_tr, X_va, y_va = build_temporal_split(train_weekly, NUM_FEATURES, stride=WINDOW_STRIDE)
    print(f"   Train windows: {len(X_tr):,}  |  Val windows: {len(X_va):,} (all regions)")
    print(f"   Note: val MAE should now be close to ~0.8x (realistic test estimate)")

    last_score = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    persist_val_regions = X_va["region_id"].values
    persist_va = np.column_stack([
        last_score.reindex(persist_val_regions).fillna(0).to_numpy() for _ in range(5)
    ])
    show_mae("Persistence-Baseline (temporal)", y_va, persist_va)

    # 6. Train models
    print("\n[6/6] Training models ...")

    print("  Training LightGBM ...")
    lgb_models = train_lgb_models(X_tr, y_tr, X_va, y_va)
    lgb_val = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (temporal val)", y_va, lgb_val)

    print("  Training XGBoost ...")
    xgb_models = train_xgb_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
    xgb_val = predict_xgb(xgb_models, X_va, NUM_FEATURES)
    show_mae("XGBoost (temporal val)", y_va, xgb_val)

    cat_val = None
    cat_models = None
    if CATBOOST_AVAILABLE:
        print("  Training CatBoost ...")
        cat_models = train_cat_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
        cat_val = predict_cat(cat_models, X_va, NUM_FEATURES)
        show_mae("CatBoost (temporal val)", y_va, cat_val)

    print("  Blend optimisation (min weight 0.15) ...")
    best_weights, best_mae_val = optimize_blend(y_va, lgb_val, xgb_val, cat_val)
    lgb_w, xgb_w, cat_w = best_weights
    if cat_val is not None:
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}  CAT={cat_w:.2f}   MAE={best_mae_val:.4f}")
    else:
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}   MAE={best_mae_val:.4f}")

    # Final training on ALL data
    print("\n  Final training (all regions, all windows) ...")
    X_all, y_all = build_all_windows(train_weekly, NUM_FEATURES, stride=WINDOW_STRIDE)

    n_lgb = [_best_n(m, LGB_PARAMS["n_estimators"]) for m in lgb_models]
    n_xgb = [_best_n(m, XGB_PARAMS["n_estimators"]) for m in xgb_models]

    final_lgb = train_lgb_models(X_all, y_all, None, None, n_lgb)
    final_xgb = train_xgb_models(X_all, y_all, None, None, NUM_FEATURES, n_xgb)

    final_cat = None
    if CATBOOST_AVAILABLE and cat_models is not None:
        n_cat = [_best_n(m, CAT_PARAMS["iterations"]) for m in cat_models]
        final_cat = train_cat_models(X_all, y_all, None, None, NUM_FEATURES, n_cat)

    print(f"   Done  |  [{elapsed(t0)}]")

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
        cat_test  = predict_cat(final_cat, X_test, NUM_FEATURES)
        test_preds = lgb_w * lgb_test + xgb_w * xgb_test + cat_w * cat_test
    else:
        test_preds = lgb_w * lgb_test + xgb_w * xgb_test

    sub = pd.DataFrame({"region_id": X_test["region_id"].values})
    for k in range(5):
        sub[f"pred_week{k+1}"] = test_preds[:, k]

    template = pd.read_csv(SAMPLE_SUB)
    sub = template[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col] = sub[col].fillna(0.0)

    sub.to_csv(OUT_PATH, index=False)

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*68}")
    print(f"  Saved: {OUT_PATH}")
    print(f"  Rows: {len(sub):,}  |  Total: {total_min:.1f} Min.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
