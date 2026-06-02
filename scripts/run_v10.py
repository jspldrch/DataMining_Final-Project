"""
run_v10.py  –  Drought Severity Prediction v10

Base: run_v7 (MAE 0.8303, best so far).

Pattern from v8/v9: every feature change or hyperparameter change made things WORSE.
Conclusion: v7 already has the right features. The remaining error is variance, not bias.

Fix: Seed ensemble — train 3 LGB + 2 XGB models with different random seeds,
average their predictions. This reduces random variance without any feature/parameter changes.

Why this helps without overfitting:
  - No feature changes         → no new noise
  - No hyperparameter changes  → no val-set overfitting
  - Averaging reduces variance → predictions closer to the true expectation
  - Theory: k models with correlation ρ reduce variance by factor (ρ + (1-ρ)/k)

v7 features and hyperparameters kept EXACTLY.

Usage:
    python scripts/run_v10.py
Output: outputs/submission_v10.csv
Estimated runtime: ~3-4h (3x LGB + 2x XGB + CatBoost, all trained twice)
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
OUT_PATH   = OUT_DIR / "submission_v10.csv"

# ─── Mode ─────────────────────────────────────────────────────────────────────
QUICK_MODE = False

RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0

WINDOW_STRIDE = 1 if not QUICK_MODE else 4
N_ESTIMATORS  = 1000 if not QUICK_MODE else 400

# Seed ensembles — more seeds = lower variance, longer runtime
LGB_SEEDS = [42, 0, 123]    # 3 LGB models averaged
XGB_SEEDS = [42, 0]         # 2 XGB models averaged (XGB is slower)

# ─── Feature config (IDENTICAL to v7) ─────────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp", "humidity"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]

# ─── Model params (IDENTICAL to v7) ───────────────────────────────────────────
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


# ─── Feature list (IDENTICAL to v7) ───────────────────────────────────────────

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
    extra = ["regional_mean_score"]
    return WEATHER_COLS + lag_names + roll_names + calendar + drought + extra


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


# ─── Regional mean score (IDENTICAL to v7) ────────────────────────────────────

def add_regional_mean_score(df: pd.DataFrame, region_means: pd.Series) -> pd.DataFrame:
    df["regional_mean_score"] = df["region_id"].map(region_means).astype(np.float32)
    return df


# ─── Feature engineering per region (IDENTICAL to v7) ─────────────────────────

def compute_region_features(
    tr: pd.DataFrame,
    te: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            min_p = max(3, w // 10)
            r = prior.rolling(w, min_periods=min_p)
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


# ─── Seed-ensemble model training ─────────────────────────────────────────────

def _lgb_best_n(m: lgb.LGBMRegressor) -> int:
    return int(getattr(m, "best_iteration_", None) or LGB_PARAMS["n_estimators"])


def _xgb_best_n(m: xgb.XGBRegressor) -> int:
    return int(getattr(m, "best_iteration", None) or XGB_PARAMS["n_estimators"])


def _cat_best_n(m, default: int) -> int:
    try:
        bi = m.get_best_iteration()
        return int(bi) if bi is not None else default
    except Exception:
        return default


def train_lgb_one_seed(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame | None, y_va: np.ndarray | None,
    seed: int,
    n_trees_per_week: list[int] | None = None,
) -> tuple[list[lgb.LGBMRegressor], list[int]]:
    """Train 5 LGB models (one per horizon) for one seed. Returns models + best_n_trees."""
    models, best_ns = [], []
    for week in range(5):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or LGB_PARAMS["n_estimators"]
        p = dict(LGB_PARAMS, random_state=seed + week * 1000, n_estimators=n)
        m = lgb.LGBMRegressor(**p)
        fit_kw: dict = dict(categorical_feature=["region_id"])
        if X_va is not None and n_trees_per_week is None:
            fit_kw["eval_set"] = [(X_va, y_va[:, week].ravel())]
            fit_kw["eval_metric"] = "mae"
            fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        m.fit(X_tr, y_tr[:, week].ravel(), **fit_kw)
        models.append(m)
        best_ns.append(_lgb_best_n(m))
    return models, best_ns


def train_xgb_one_seed(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame | None, y_va: np.ndarray | None,
    num_features: list[str],
    seed: int,
    n_trees_per_week: list[int] | None = None,
) -> tuple[list[xgb.XGBRegressor], list[int]]:
    X_tr_n = X_tr[num_features].to_numpy(dtype=np.float32)
    X_va_n = X_va[num_features].to_numpy(dtype=np.float32) if X_va is not None else None
    models, best_ns = [], []
    for week in range(5):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or XGB_PARAMS["n_estimators"]
        p = dict(XGB_PARAMS, random_state=seed + week * 1000, n_estimators=n)
        fit_kw: dict = {}
        if X_va_n is not None and n_trees_per_week is None:
            p["early_stopping_rounds"] = 50
            fit_kw["eval_set"] = [(X_va_n, y_va[:, week].ravel())]
            fit_kw["verbose"] = False
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr_n, y_tr[:, week].ravel(), **fit_kw)
        models.append(m)
        best_ns.append(_xgb_best_n(m))
    return models, best_ns


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


def train_cat_single(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame | None, y_va: np.ndarray | None,
    num_features: list[str],
    n_trees_per_week: list[int] | None = None,
) -> tuple[list | None, list[int]]:
    if not CATBOOST_AVAILABLE:
        return None, [CAT_PARAMS["iterations"]] * 5
    X_tr_n = X_tr[num_features].to_numpy(dtype=np.float32)
    X_va_n = X_va[num_features].to_numpy(dtype=np.float32) if X_va is not None else None
    models, best_ns = [], []
    for week in range(5):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or CAT_PARAMS["iterations"]
        p = dict(CAT_PARAMS, iterations=n, random_seed=RANDOM_STATE + week)
        fit_kw: dict = {}
        if X_va_n is not None and n_trees_per_week is None:
            fit_kw["eval_set"] = (X_va_n, y_va[:, week].ravel())
            fit_kw["early_stopping_rounds"] = 50
        m = CatBoostRegressor(**p)
        m.fit(X_tr_n, y_tr[:, week].ravel(), **fit_kw)
        models.append(m)
        best_ns.append(_cat_best_n(m, CAT_PARAMS["iterations"]))
    return models, best_ns


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(y_pred, 0, 5) - y_true)))


def show_mae(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"  {name:<52s}  MAE = {mae(y_true, y_pred):.4f}")


def optimize_blend(
    y_va: np.ndarray,
    lgb_val: np.ndarray,
    xgb_val: np.ndarray,
    cat_val: np.ndarray | None = None,
) -> tuple[tuple[float, float, float], float]:
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
    print("=" * 68)
    print("  Natural Disaster Severity Prediction  -  run_v10.py")
    mode_label = "QUICK (~45 min)" if QUICK_MODE else "FULL (~3-4h)"
    print(f"  Mode: {mode_label}  |  stride={WINDOW_STRIDE}  estimators={N_ESTIMATORS}")
    cat_label = "ON" if CATBOOST_AVAILABLE else "OFF"
    print(f"  LGB seeds: {LGB_SEEDS}  |  XGB seeds: {XGB_SEEDS}  |  CatBoost: {cat_label}")
    print(f"  Features: {len(NUM_FEATURES)}  (identical to v7)")
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

    region_means = train_raw.groupby("region_id")["score"].mean()

    # 2. Feature engineering
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
    print(f"   {len(train_weekly):,} weekly rows  |  [{elapsed(t0)}]")

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

    # 5. Train seed ensembles
    print(f"\n[5/6] Training seed ensembles ...")

    # LGB: first seed uses early stopping, others use same n_trees
    print(f"  LGB seed {LGB_SEEDS[0]} (early stopping) ...")
    lgb_models_0, n_lgb = train_lgb_one_seed(X_tr, y_tr, X_va, y_va, seed=LGB_SEEDS[0])
    lgb_preds_val = [predict_lgb(lgb_models_0, X_va)]

    for seed in LGB_SEEDS[1:]:
        print(f"  LGB seed {seed} (fixed n={n_lgb}) ...")
        lgb_s, _ = train_lgb_one_seed(X_tr, y_tr, None, None, seed=seed, n_trees_per_week=n_lgb)
        lgb_preds_val.append(predict_lgb(lgb_s, X_va))

    lgb_val_ens = np.mean(lgb_preds_val, axis=0).astype(np.float32)
    show_mae(f"LightGBM ensemble ({len(LGB_SEEDS)} seeds, val)", y_va, lgb_val_ens)

    # XGB: first seed uses early stopping, others use same n_trees
    print(f"  XGB seed {XGB_SEEDS[0]} (early stopping) ...")
    xgb_models_0, n_xgb = train_xgb_one_seed(X_tr, y_tr, X_va, y_va, NUM_FEATURES, seed=XGB_SEEDS[0])
    xgb_preds_val = [predict_xgb(xgb_models_0, X_va, NUM_FEATURES)]

    for seed in XGB_SEEDS[1:]:
        print(f"  XGB seed {seed} (fixed n={n_xgb}) ...")
        xgb_s, _ = train_xgb_one_seed(X_tr, y_tr, None, None, NUM_FEATURES, seed=seed, n_trees_per_week=n_xgb)
        xgb_preds_val.append(predict_xgb(xgb_s, X_va, NUM_FEATURES))

    xgb_val_ens = np.mean(xgb_preds_val, axis=0).astype(np.float32)
    show_mae(f"XGBoost ensemble ({len(XGB_SEEDS)} seeds, val)", y_va, xgb_val_ens)

    # CatBoost: single model
    cat_val = None
    cat_models, n_cat = None, [CAT_PARAMS["iterations"]] * 5
    if CATBOOST_AVAILABLE:
        print("  CatBoost (early stopping) ...")
        cat_models, n_cat = train_cat_single(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
        cat_val = predict_cat(cat_models, X_va, NUM_FEATURES)
        show_mae("CatBoost (val)", y_va, cat_val)

    # Blend
    print("\n  Blend optimisation (0.05 steps) ...")
    best_weights, best_mae_val = optimize_blend(y_va, lgb_val_ens, xgb_val_ens, cat_val)
    lgb_w, xgb_w, cat_w = best_weights
    if cat_val is not None:
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}  CAT={cat_w:.2f}   MAE={best_mae_val:.4f}")
    else:
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}   MAE={best_mae_val:.4f}")

    # Final training on all data
    print(f"\n  Final training (all regions) ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE)

    lgb_final_preds_test: list[np.ndarray] = []
    xgb_final_preds_test: list[np.ndarray] = []

    X_test = (
        test_feat.sort_values(["region_id", "ordinal"])
        .groupby("region_id", sort=False)
        .tail(1)[["region_id"] + NUM_FEATURES]
        .reset_index(drop=True)
    )
    X_test["region_id"] = X_test["region_id"].astype("category")

    for seed in LGB_SEEDS:
        print(f"  Final LGB seed {seed} ...")
        m_list, _ = train_lgb_one_seed(X_all, y_all, None, None, seed=seed, n_trees_per_week=n_lgb)
        lgb_final_preds_test.append(predict_lgb(m_list, X_test))

    for seed in XGB_SEEDS:
        print(f"  Final XGB seed {seed} ...")
        m_list, _ = train_xgb_one_seed(X_all, y_all, None, None, NUM_FEATURES, seed=seed, n_trees_per_week=n_xgb)
        xgb_final_preds_test.append(predict_xgb(m_list, X_test, NUM_FEATURES))

    lgb_test = np.mean(lgb_final_preds_test, axis=0).astype(np.float32)
    xgb_test = np.mean(xgb_final_preds_test, axis=0).astype(np.float32)

    if CATBOOST_AVAILABLE and cat_models is not None:
        print("  Final CatBoost ...")
        final_cat, _ = train_cat_single(X_all, y_all, None, None, NUM_FEATURES, n_trees_per_week=n_cat)
        cat_test = predict_cat(final_cat, X_test, NUM_FEATURES)
        test_preds = lgb_w * lgb_test + xgb_w * xgb_test + cat_w * cat_test
    else:
        test_preds = lgb_w * lgb_test + xgb_w * xgb_test

    print(f"   Done  |  [{elapsed(t0)}]")

    # 6. Submission
    print("\n[6/6] Saving submission ...")
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
