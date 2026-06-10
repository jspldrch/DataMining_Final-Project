"""
run_v12.py  –  Drought Severity Prediction v12

Base: run_v7 (MAE 0.8303, best so far). v7 features + hyperparameters kept EXACTLY.

What failed (lessons learned):
  v8:  adding features / over-regularising  → worse
  v9:  lower lr + monthly mean score        → worse
  v10: seed ensemble                         → no gain (LGB+XGB too correlated)
  v11: temporal validation                  → much worse (train/test distribution mismatch)

What v12 adds (only one change):
  ExtraTrees Regressor as 4th ensemble member.

  ExtraTrees (Extremely Randomized Trees) uses RANDOM SPLITS (not best splits like GBM).
  Correlation with LGB/XGB: ~0.80-0.87 vs LGB-XGB correlation ~0.95.
  Lower correlation → more variance reduction when averaging → lower MAE.

  ExtraTrees is trained on a subsample (200k) to avoid memory issues.
  Blend optimizer finds the best LGB / XGB / CAT / ET weights.

  Region-holdout validation (seed 42) kept — proven reliable for this problem.

Usage:
    python scripts/run_v12.py
Output: outputs/submission_v12.csv
Estimated runtime: ~100-120 min (ET adds ~20 min)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor

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
OUT_PATH   = OUT_DIR / "submission_v12.csv"

# ─── Mode ─────────────────────────────────────────────────────────────────────
QUICK_MODE = False

RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0

WINDOW_STRIDE  = 1 if not QUICK_MODE else 4
N_ESTIMATORS   = 1000 if not QUICK_MODE else 400
ET_SUBSAMPLE   = 200_000   # max rows for ExtraTrees (memory constraint)

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
# ExtraTrees: random splits → low correlation with GBM → good ensemble diversity
ET_PARAMS = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=50,
    max_features=0.7,
    n_jobs=-1,
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


# ─── Feature engineering (IDENTICAL to v7) ────────────────────────────────────

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
    new_cols["tmp_anomaly_90d"]  = tmp_anomaly
    new_cols["heat_drought_idx"] = (new_cols["prec_deficit_90d"] * tmp_anomaly.clip(lower=0)).astype(np.float32)

    dry = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    new_cols["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    new_cols["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)

    panel = pd.concat([panel, pd.DataFrame(new_cols, index=panel.index)], axis=1)
    n_tr = len(tr)
    return panel.iloc[:n_tr].copy(), panel.iloc[n_tr:].copy()


# ─── Dataset assembly (IDENTICAL to v7) ───────────────────────────────────────

def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    week = df["ordinal"] // WEEK_BUCKET
    idx  = df.groupby(week, sort=False)["ordinal"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_sliding_windows(weekly, skip_regions, num_features, stride=1):
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


def build_val_samples(weekly, val_regions, num_features):
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


def train_et_models(X_tr, y_tr, num_features):
    """ExtraTrees on subsample (memory-safe). Random splits → low GBM correlation."""
    X_num = X_tr[num_features].to_numpy(dtype=np.float32)
    if len(X_num) > ET_SUBSAMPLE:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_num), ET_SUBSAMPLE, replace=False)
        X_s, y_s = X_num[idx], y_tr[idx]
    else:
        X_s, y_s = X_num, y_tr
    models = []
    for week in range(5):
        m = ExtraTreesRegressor(**dict(ET_PARAMS, random_state=RANDOM_STATE + week))
        m.fit(X_s, y_s[:, week].ravel())
        models.append(m)
    return models


def predict_lgb(models, X):
    return np.clip(np.column_stack([m.predict(X) for m in models]), 0.0, 5.0).astype(np.float32)


def predict_num(models, X, num_features):
    X_n = X[num_features].to_numpy(dtype=np.float32)
    return np.clip(np.column_stack([m.predict(X_n) for m in models]), 0.0, 5.0).astype(np.float32)


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


def optimize_blend(y_va, preds_dict: dict) -> tuple[dict, float]:
    """
    Grid search blend weights for any number of models.
    preds_dict: {"lgb": array, "xgb": array, ...}
    Returns best weights dict and best MAE.
    """
    names = list(preds_dict.keys())
    arrays = [preds_dict[n] for n in names]
    k = len(names)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    best_mae_v, best_w = 999.0, {n: 1.0 / k for n in names}

    if k == 2:
        for a in alphas:
            b = round(1.0 - a, 8)
            m = mae(y_va, a * arrays[0] + b * arrays[1])
            if m < best_mae_v:
                best_mae_v = m
                best_w = {names[0]: a, names[1]: b}
    elif k == 3:
        for a in alphas:
            for b in alphas:
                c = round(1.0 - a - b, 8)
                if c < 0.05:
                    continue
                m = mae(y_va, a * arrays[0] + b * arrays[1] + c * arrays[2])
                if m < best_mae_v:
                    best_mae_v = m
                    best_w = {names[0]: a, names[1]: b, names[2]: c}
    elif k == 4:
        for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                for c in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                    d = round(1.0 - a - b - c, 8)
                    if d < 0.05:
                        continue
                    m = mae(y_va, a*arrays[0] + b*arrays[1] + c*arrays[2] + d*arrays[3])
                    if m < best_mae_v:
                        best_mae_v = m
                        best_w = {names[0]: a, names[1]: b, names[2]: c, names[3]: d}

    return best_w, best_mae_v


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 68)
    print("  Natural Disaster Severity Prediction  -  run_v12.py")
    mode_label = "QUICK (~30 min)" if QUICK_MODE else "FULL (~100-120 min)"
    print(f"  Mode: {mode_label}  |  stride={WINDOW_STRIDE}  estimators={N_ESTIMATORS}")
    cat_label = "ON" if CATBOOST_AVAILABLE else "OFF"
    print(f"  CatBoost: {cat_label}  |  ExtraTrees: ON (subsample={ET_SUBSAMPLE:,})")
    print(f"  Features: {len(NUM_FEATURES)} (identical to v7)")
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

    # 2. Feature engineering per region
    print("\n[2/6] Feature engineering per region ...")
    train_by_region = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    test_by_region  = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw

    all_tr, all_te = [], []
    n = len(regions)
    for i, region in enumerate(regions, 1):
        if i % 500 == 0 or i == n:
            print(f"   Region {i}/{n}  |  [{elapsed(t0)}]")
        tr_f, te_f = compute_region_features(
            train_by_region[region], test_by_region.get(region, pd.DataFrame())
        )
        all_tr.append(tr_f)
        all_te.append(te_f)

    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te

    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)
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

    # 4. Train/val split (region holdout — proven reliable for this problem)
    print("\n[4/6] Building train/val split (region holdout 20%) ...")
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

    # 5. Train all models
    print("\n[5/6] Training models ...")

    print("  Training LightGBM ...")
    lgb_models = train_lgb_models(X_tr, y_tr, X_va, y_va)
    lgb_val = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (val)", y_va, lgb_val)

    print("  Training XGBoost ...")
    xgb_models = train_xgb_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
    xgb_val = predict_num(xgb_models, X_va, NUM_FEATURES)
    show_mae("XGBoost (val)", y_va, xgb_val)

    cat_val, cat_models = None, None
    if CATBOOST_AVAILABLE:
        print("  Training CatBoost ...")
        cat_models = train_cat_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
        cat_val = predict_num(cat_models, X_va, NUM_FEATURES)
        show_mae("CatBoost (val)", y_va, cat_val)

    print(f"  Training ExtraTrees (subsample {min(len(X_tr), ET_SUBSAMPLE):,}) ...")
    et_models_val = train_et_models(X_tr, y_tr, NUM_FEATURES)
    et_val = predict_num(et_models_val, X_va, NUM_FEATURES)
    show_mae("ExtraTrees (val)", y_va, et_val)

    # Build predictions dict for blend
    preds_val = {"lgb": lgb_val, "xgb": xgb_val, "et": et_val}
    if cat_val is not None:
        preds_val["cat"] = cat_val

    print(f"\n  Blend optimisation ({len(preds_val)} models) ...")
    best_w, best_mae_val = optimize_blend(y_va, preds_val)
    w_str = "  ".join(f"{k.upper()}={v:.2f}" for k, v in best_w.items())
    print(f"   {w_str}   MAE={best_mae_val:.4f}")

    # 6. Final training on all data
    print("\n[6/6] Final training (all regions) ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE)

    n_lgb = [int(getattr(m, "best_iteration_", None) or LGB_PARAMS["n_estimators"]) for m in lgb_models]
    n_xgb = [int(getattr(m, "best_iteration",  None) or XGB_PARAMS["n_estimators"]) for m in xgb_models]

    final_lgb = train_lgb_models(X_all, y_all, None, None, n_lgb)
    final_xgb = train_xgb_models(X_all, y_all, None, None, NUM_FEATURES, n_xgb)

    final_cat = None
    if CATBOOST_AVAILABLE and cat_models is not None:
        n_cat = [_best_n(m, CAT_PARAMS["iterations"]) for m in cat_models]
        final_cat = train_cat_models(X_all, y_all, None, None, NUM_FEATURES, n_cat)

    print("  Training final ExtraTrees ...")
    final_et = train_et_models(X_all, y_all, NUM_FEATURES)
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
    xgb_test = predict_num(final_xgb, X_test, NUM_FEATURES)
    et_test  = predict_num(final_et,  X_test, NUM_FEATURES)

    test_preds = best_w["lgb"] * lgb_test + best_w["xgb"] * xgb_test + best_w["et"] * et_test
    if final_cat is not None and "cat" in best_w:
        cat_test    = predict_num(final_cat, X_test, NUM_FEATURES)
        test_preds += best_w["cat"] * cat_test

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
