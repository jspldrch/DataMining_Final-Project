"""
run_v8.py  –  Drought Severity Prediction v8

Base: run_v7 (MAE 0.8303). Conservative additions only.

New vs v7:
  1. Air dryness index  (tmp - dp_tmp)  — dew point depression is the single best
                                          atmospheric drought indicator; higher = drier air
  2. Air dryness rolling + lags         — 7d/30d/90d rolling means + 7d/21d lags
  3. Compound drought stress            — dry_days_30d × max(0, tmp_anomaly_90d)
  4. LAGS extended to 28 days           — [1,3,7,14,21,28] captures 4 weeks back
  5. min_child_samples 60 → 100         — stronger regularisation for 127 leaves
  6. Finer blend grid (0.05 steps)      — finds better LGB/XGB/CAT weights

Kept identical to v7:
  - LGB objective = "regression" (L2) — proven better than L1 here
  - Same validation / sliding window structure
  - No Optuna, no z-scores

Usage:
    python scripts/run_v8.py
Output: outputs/submission_v8.csv
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
OUT_PATH   = OUT_DIR / "submission_v8.csv"

# ─── Mode ─────────────────────────────────────────────────────────────────────
QUICK_MODE = False

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
LAGS      = [1, 3, 7, 14, 21, 28]       # +28 vs v7 (4-week memory)
ROLL_COLS = ["prec", "wind", "tmp", "humidity"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    num_leaves=127,
    min_child_samples=100,              # 60 → 100: stronger regularisation for 127 leaves
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
    lag_names  = [f"{c}_lag{lag}"       for c in LAG_COLS  for lag in LAGS]
    roll_names = [
        f"{col}_roll{w}_{stat}"
        for col in ROLL_COLS
        for w in ROLL_WINS
        for stat in ("mean", "std", "max")
    ]
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos", "week_sin", "week_cos"]
    drought_indices = [
        "prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
        "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d",
        "compound_drought_stress",          # new: dry_days × heat anomaly
    ]
    # Air dryness = tmp - dp_tmp (dew point depression)
    # High value = dry air = high evaporation = drought amplifier
    air_dryness_features = [
        "air_dryness",
        "air_dryness_roll7_mean", "air_dryness_roll30_mean", "air_dryness_roll90_mean",
        "air_dryness_lag7", "air_dryness_lag21",
    ]
    extra = ["regional_mean_score"]
    return WEATHER_COLS + lag_names + roll_names + calendar + drought_indices + air_dryness_features + extra


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


# ─── Regional mean score ──────────────────────────────────────────────────────

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
    new_cols["week_sin"]  = np.sin(2 * np.pi * week_of_year / 52).astype(np.float32)
    new_cols["week_cos"]  = np.cos(2 * np.pi * week_of_year / 52).astype(np.float32)

    # Weather lags
    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS:
            new_cols[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)

    # Rolling stats (shift(1) prevents look-ahead)
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            min_p = max(3, w // 10)
            r = prior.rolling(w, min_periods=min_p)
            new_cols[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            new_cols[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            new_cols[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)

    # Drought indices (same as v7)
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
    dry_days_14 = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    dry_days_30 = dry.rolling(30, min_periods=7).sum().astype(np.float32)
    new_cols["dry_days_14d"] = dry_days_14
    new_cols["dry_days_30d"] = dry_days_30

    # Compound drought stress: dry days × heat — both stressors together
    new_cols["compound_drought_stress"] = (
        dry_days_30 * tmp_anomaly.clip(lower=0)
    ).astype(np.float32)

    # Air dryness index: tmp - dp_tmp (dew point depression)
    # High value = dry air = high potential evapotranspiration = drought amplifier
    dryness       = panel["tmp"] - panel["dp_tmp"]
    dryness_prior = dryness.shift(1)
    new_cols["air_dryness"]          = dryness.astype(np.float32)
    new_cols["air_dryness_roll7_mean"]  = dryness_prior.rolling(7,  min_periods=3).mean().astype(np.float32)
    new_cols["air_dryness_roll30_mean"] = dryness_prior.rolling(30, min_periods=5).mean().astype(np.float32)
    new_cols["air_dryness_roll90_mean"] = dryness_prior.rolling(90, min_periods=14).mean().astype(np.float32)
    new_cols["air_dryness_lag7"]  = dryness.shift(7).astype(np.float32)
    new_cols["air_dryness_lag21"] = dryness.shift(21).astype(np.float32)

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


# ─── Model training ───────────────────────────────────────────────────────────

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


def _cat_best_iter(m, default: int) -> int:
    try:
        bi = m.get_best_iteration()
        return int(bi) if bi is not None else default
    except Exception:
        return default


def optimize_blend(
    y_va: np.ndarray,
    lgb_val: np.ndarray,
    xgb_val: np.ndarray,
    cat_val: np.ndarray | None = None,
) -> tuple[tuple[float, float, float], float]:
    # 0.05 steps: finer than v7's 0.1 → finds better weights
    alphas = [round(x * 0.05, 2) for x in range(1, 20)]
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


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 66)
    print("  Natural Disaster Severity Prediction  -  run_v8.py")
    mode_label = "QUICK (~20 min)" if QUICK_MODE else "FULL (~90 min)"
    print(f"  Mode: {mode_label}  |  stride={WINDOW_STRIDE}  estimators={N_ESTIMATORS}")
    cat_label = "ON" if CATBOOST_AVAILABLE else "OFF  (pip install catboost)"
    print(f"  CatBoost: {cat_label}  |  Features: {len(NUM_FEATURES)}")
    print(f"  New vs v7: +air_dryness features, +lag28, +compound_stress, min_child=100")
    print("=" * 66)

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

    region_means = compute_regional_mean_score(train_raw)

    # 2. Feature engineering per region
    print("\n[2/6] Feature engineering per region ...")
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

    # 3. Weekly aggregation
    print("\n[3/6] Weekly aggregation ...")
    labeled = train_feat[train_feat["score"].notna()].copy()
    weekly_parts = []
    for region, g in labeled.groupby("region_id", sort=False):
        weekly_parts.append(daily_to_weekly(g))
    train_weekly = pd.concat(weekly_parts, ignore_index=True)
    del labeled
    weeks_per_region = int(len(train_weekly) / len(regions))
    print(f"   {len(train_weekly):,} weekly rows  (~{weeks_per_region}/region)  [{elapsed(t0)}]")

    # 4. Train/val split
    print("\n[4/6] Building train/val split ...")
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

    # 5. Train models
    print("\n[5/6] Training LightGBM ...")
    lgb_models = train_lgb_models(X_tr, y_tr, X_va, y_va)
    lgb_val = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (val)", y_va, lgb_val)

    print("\n       Training XGBoost ...")
    xgb_models = train_xgb_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
    xgb_val = predict_xgb(xgb_models, X_va, NUM_FEATURES)
    show_mae("XGBoost (val)", y_va, xgb_val)

    cat_val = None
    cat_models_val = None
    if CATBOOST_AVAILABLE:
        print("\n       Training CatBoost ...")
        cat_models_val = train_cat_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
        cat_val = predict_cat(cat_models_val, X_va, NUM_FEATURES)
        show_mae("CatBoost (val)", y_va, cat_val)

    print("\n  Blend optimisation (0.05 steps):")
    best_weights, best_mae_val = optimize_blend(y_va, lgb_val, xgb_val, cat_val)
    lgb_w, xgb_w, cat_w = best_weights
    if cat_val is not None:
        blend_val = lgb_w * lgb_val + xgb_w * xgb_val + cat_w * cat_val
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}  CAT={cat_w:.2f}   MAE={best_mae_val:.4f}")
    else:
        blend_val = lgb_w * lgb_val + xgb_w * xgb_val
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}   MAE={best_mae_val:.4f}")
    show_mae("Ensemble (val)", y_va, blend_val)

    # Final training on all data
    print("\n  Final training (all regions) ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE)

    n_lgb_trees = [int(getattr(m, "best_iteration_", None) or LGB_PARAMS["n_estimators"]) for m in lgb_models]
    n_xgb_trees = [int(getattr(m, "best_iteration",  None) or XGB_PARAMS["n_estimators"]) for m in xgb_models]

    final_lgb = train_lgb_models(X_all, y_all, None, None, n_lgb_trees)
    final_xgb = train_xgb_models(X_all, y_all, None, None, NUM_FEATURES, n_xgb_trees)

    final_cat = None
    if CATBOOST_AVAILABLE and cat_models_val is not None:
        n_cat_trees = [_cat_best_iter(m, CAT_PARAMS["iterations"]) for m in cat_models_val]
        final_cat = train_cat_models(X_all, y_all, None, None, NUM_FEATURES, n_cat_trees)

    print(f"   Done  |  [{elapsed(t0)}]")

    # 6. Test predictions + submission
    print("\n[6/6] Test predictions ...")
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
    print(f"\n{'='*66}")
    print(f"  Saved: {OUT_PATH}")
    print(f"  Rows: {len(sub):,}  |  Total: {total_min:.1f} Min.")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
