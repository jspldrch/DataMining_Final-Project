"""
run_v4.py  –  Enhanced model with richer features + optional CatBoost

Run from project root:
    python scripts/run_v4.py

Output: outputs/submission_v4.csv

Improvements over v3 (MAE ~0.8x):
  1. Extended score lags   - up to 112 days (16 weeks) back
  2. Score trend           - drought acceleration indicator (worsening/improving)
  3. Dry-day counts        - days below 1mm in last 14/30 days
  4. Humidity deficit      - humidity 90d vs 365d baseline (heat-drought indicator)
  5. Temperature anomaly   - tmp deviation from 365-day baseline
  6. Heat-drought index    - interaction: prec_deficit x temp_anomaly
  7. CatBoost              - optional 3rd ensemble model (pip install catboost)
  8. 3-way blend search    - optimal LGB/XGB/CAT weights on validation
"""

from __future__ import annotations

import time
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

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB = ROOT / "resources" / "sample_submission.csv"
OUT_PATH = OUT_DIR / "submission_v4.csv"

# ─── Mode ──────────────────────────────────────────────────────────────────────
# QUICK_MODE = True  -> ~20 min  (test pipeline, not for Kaggle)
# QUICK_MODE = False -> ~90 min  (full Kaggle run)
QUICK_MODE = True

RANDOM_STATE = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET = 7

WINDOW_STRIDE = 1 if not QUICK_MODE else 4
N_ESTIMATORS = 1000 if not QUICK_MODE else 400

DRY_THRESHOLD = 1.0  # mm/day – below this a day counts as "dry"

# ─── Feature definitions ───────────────────────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre"]
LAGS = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp"]
ROLL_WINS = [7, 14, 30, 60, 90]
# v4: extended from [7..35] to [7..112]  (up to 16 weeks back into training history)
SCORE_LAGS = [7, 14, 21, 28, 35, 42, 56, 84, 112]

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    num_leaves=63,
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


# ─── Feature list ──────────────────────────────────────────────────────────────

def build_feature_list() -> list[str]:
    lag_names = [f"{c}_lag{lag}" for c in LAG_COLS for lag in LAGS]
    roll_names = [
        f"{col}_roll{w}_{stat}"
        for col in ROLL_COLS
        for w in ROLL_WINS
        for stat in ("mean", "std", "max")
    ]
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos"]
    score_names = ["score_persist"] + [f"score_lag{lag}" for lag in SCORE_LAGS]
    drought_indices = [
        "prec_deficit_90d",
        "prec_trend_30d",
        "humidity_deficit_90d",
        "tmp_anomaly_90d",
        "heat_drought_idx",
        "score_trend",
        "dry_days_14d",
        "dry_days_30d",
    ]
    return WEATHER_COLS + lag_names + roll_names + calendar + score_names + drought_indices


# ─── Feature engineering ───────────────────────────────────────────────────────

def _parse_dates_inplace(df: pd.DataFrame) -> None:
    parts = df["date"].str.split("-", expand=True)
    df["year"] = parts[0].astype(np.int16)
    df["month"] = parts[1].astype(np.int8)
    df["day"] = parts[2].astype(np.int8)
    df["ordinal"] = (
        df["year"].astype(np.int32) * 372
        + df["month"].astype(np.int32) * 31
        + df["day"].astype(np.int32)
    )


def compute_region_features(
    tr: pd.DataFrame,
    te: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train + test combined so rolling/lag features flow from training into test
    rows without look-ahead.  shift(1) on all rolling ops prevents data leakage.
    """
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)

    # Calendar
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    panel["day_sin"] = np.sin(2 * np.pi * panel["day"] / 31).astype(np.float32)
    panel["day_cos"] = np.cos(2 * np.pi * panel["day"] / 31).astype(np.float32)

    # Weather lags
    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS:
            panel[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)

    # Rolling features (shift(1) avoids look-ahead)
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=3)
            panel[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            panel[f"{col}_roll{w}_std"] = r.std().astype(np.float32)
            panel[f"{col}_roll{w}_max"] = r.max().astype(np.float32)

    # Precipitation deficit: negative = drier than annual baseline → drought signal
    prec_prior = panel["prec"].shift(1)
    panel["prec_deficit_90d"] = (
        prec_prior.rolling(90, min_periods=30).mean()
        - prec_prior.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    # Precipitation trend: is it becoming drier short-term vs long-term?
    p7 = prec_prior.rolling(7, min_periods=3).mean()
    p30 = prec_prior.rolling(30, min_periods=10).mean()
    p30_std = prec_prior.rolling(30, min_periods=10).std().clip(lower=0.01)
    panel["prec_trend_30d"] = ((p7 - p30) / p30_std).astype(np.float32)

    # v4: Humidity deficit (low humidity reinforces drought severity)
    hum_prior = panel["humidity"].shift(1)
    panel["humidity_deficit_90d"] = (
        hum_prior.rolling(90, min_periods=30).mean()
        - hum_prior.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    # v4: Temperature anomaly (above-normal heat accelerates drought)
    tmp_prior = panel["tmp"].shift(1)
    panel["tmp_anomaly_90d"] = (
        tmp_prior.rolling(90, min_periods=30).mean()
        - tmp_prior.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    # v4: Heat-drought interaction: simultaneous heat + low precip amplifies severity
    panel["heat_drought_idx"] = (
        panel["prec_deficit_90d"] * panel["tmp_anomaly_90d"].clip(lower=0)
    ).astype(np.float32)

    # v4: Dry day counts (consecutive lack of rain is the physical drought mechanism)
    dry = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    panel["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    panel["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)

    # Score lags: ffill propagates last known score from training into test rows
    score_filled = panel["score"].ffill()
    panel["score_persist"] = score_filled.shift(7).astype(np.float32)
    for lag in SCORE_LAGS:
        panel[f"score_lag{lag}"] = score_filled.shift(lag).astype(np.float32)

    # v4: Score trend: positive = drought worsening, negative = improving
    panel["score_trend"] = (
        panel["score_lag7"] - panel["score_lag35"]
    ).astype(np.float32)

    n_tr = len(tr)
    return panel.iloc[:n_tr].copy(), panel.iloc[n_tr:].copy()


# ─── Dataset assembly ──────────────────────────────────────────────────────────

def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    week = df["ordinal"] // WEEK_BUCKET
    idx = df.groupby(week, sort=False)["ordinal"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_sliding_windows(
    weekly: pd.DataFrame,
    skip_regions: set,
    num_features: list[str],
    stride: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature at week i → scores at weeks i+1..i+5.  stride subsamples windows."""
    X_parts, y_parts, region_parts = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions:
            continue
        g = g.sort_values("ordinal")
        scores = g["score"].to_numpy(dtype=np.float32)
        X_num = g[num_features].to_numpy(dtype=np.float32)
        n = len(g)
        if n < 6:
            continue
        n_win = n - 5
        y_reg = np.lib.stride_tricks.sliding_window_view(scores[1:], 5)[:n_win]
        idx = list(range(0, n_win, stride))
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


# ─── Model training ────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(y_pred, 0, 5) - y_true)))


def show_mae(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"  {name:<52s}  MAE = {mae(y_true, y_pred):.4f}")


def train_lgb_models(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
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
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
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
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
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
    """Grid search for best (lgb_w, xgb_w, cat_w) on validation set."""
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


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 66)
    print("  Natural Disaster Severity Prediction  -  run_v4.py")
    mode_label = "QUICK (~20 min)" if QUICK_MODE else "FULL (~90 min)"
    print(f"  Mode: {mode_label}  |  stride={WINDOW_STRIDE}  estimators={N_ESTIMATORS}")
    cat_label = "CatBoost: ON" if CATBOOST_AVAILABLE else "CatBoost: OFF  (pip install catboost)"
    print(f"  {cat_label}  |  Features: {len(NUM_FEATURES)}")
    print("=" * 66)

    # 1. Load data
    print("\n[1/6] Lade CSV-Dateien ...")
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw = pd.read_csv(TEST_CSV, dtype=dtypes)
    _parse_dates_inplace(train_raw)
    _parse_dates_inplace(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    print(f"   Train: {len(train_raw):>10,} Zeilen  |  Test: {len(test_raw):>8,} Zeilen")
    print(f"   Regionen: {len(regions)}  |  Zeit: {time.time()-t0:.1f}s")

    # 2. Feature engineering per region
    print("\n[2/6] Feature Engineering (pro Region) ...")
    all_tr_feat, all_te_feat = [], []
    n = len(regions)
    for i, region in enumerate(regions, 1):
        if i % 500 == 0 or i == n:
            print(f"   Region {i}/{n}  |  {time.time()-t0:.1f}s")
        tr = train_raw.loc[train_raw["region_id"] == region].reset_index(drop=True)
        te = test_raw.loc[test_raw["region_id"] == region].reset_index(drop=True)
        tr_f, te_f = compute_region_features(tr, te)
        all_tr_feat.append(tr_f)
        all_te_feat.append(te_f)

    train_feat = pd.concat(all_tr_feat, ignore_index=True)
    test_feat = pd.concat(all_te_feat, ignore_index=True)
    del all_tr_feat, all_te_feat, train_raw, test_raw
    print(f"   Fertig  |  {time.time()-t0:.1f}s")

    # 3. Weekly aggregation
    print("\n[3/6] Woechentliche Aggregation ...")
    labeled = train_feat[train_feat["score"].notna()].copy()
    weekly_parts = []
    for region, g in labeled.groupby("region_id", sort=False):
        weekly_parts.append(daily_to_weekly(g))
    train_weekly = pd.concat(weekly_parts, ignore_index=True)
    del labeled
    weeks_per_region = int(len(train_weekly) / len(regions))
    print(f"   {len(train_weekly):,} Wochen-Zeilen  (~{weeks_per_region}/Region)")

    # 4. Train/val split (region holdout)
    print("\n[4/6] Train/Validierung aufbauen ...")
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg = sorted(train_weekly["region_id"].unique())
    n_val = max(1, int(len(all_reg) * VAL_REGION_FRAC))
    val_regions = set(rng.choice(all_reg, size=n_val, replace=False))

    X_tr, y_tr = build_sliding_windows(train_weekly, val_regions, NUM_FEATURES, stride=WINDOW_STRIDE)
    X_va, y_va = build_val_samples(train_weekly, sorted(val_regions), NUM_FEATURES)
    print(f"   Train-Fenster: {len(X_tr):,}  |  Val-Regionen: {len(val_regions)}")

    last_score = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    persist_va = np.column_stack([
        last_score.reindex(sorted(val_regions)).fillna(0).to_numpy() for _ in range(5)
    ])
    show_mae("Persistence-Baseline", y_va, persist_va)

    # 5. Model training
    print("\n[5/6] Training LightGBM ...")
    lgb_models = train_lgb_models(X_tr, y_tr, X_va, y_va)
    lgb_val = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (Validierung)", y_va, lgb_val)

    print("\n       Training XGBoost ...")
    xgb_models = train_xgb_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
    xgb_val = predict_xgb(xgb_models, X_va, NUM_FEATURES)
    show_mae("XGBoost (Validierung)", y_va, xgb_val)

    cat_val = None
    cat_models_val = None
    if CATBOOST_AVAILABLE:
        print("\n       Training CatBoost ...")
        cat_models_val = train_cat_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
        cat_val = predict_cat(cat_models_val, X_va, NUM_FEATURES)
        show_mae("CatBoost (Validierung)", y_va, cat_val)

    print("\n  Blend-Optimierung:")
    best_weights, best_mae_val = optimize_blend(y_va, lgb_val, xgb_val, cat_val)
    lgb_w, xgb_w, cat_w = best_weights
    if cat_val is not None:
        blend_val = lgb_w * lgb_val + xgb_w * xgb_val + cat_w * cat_val
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}  CAT={cat_w:.2f}   MAE={best_mae_val:.4f}")
    else:
        blend_val = lgb_w * lgb_val + xgb_w * xgb_val
        print(f"   LGB={lgb_w:.2f}  XGB={xgb_w:.2f}   MAE={best_mae_val:.4f}")
    show_mae("Ensemble (Validierung)", y_va, blend_val)

    # Final training on all data
    print("\n  Finales Training (alle Regionen) ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE)

    n_lgb_trees = [
        int(getattr(m, "best_iteration_", None) or LGB_PARAMS["n_estimators"])
        for m in lgb_models
    ]
    n_xgb_trees = [
        int(getattr(m, "best_iteration", None) or XGB_PARAMS["n_estimators"])
        for m in xgb_models
    ]

    final_lgb = train_lgb_models(X_all, y_all, None, None, n_lgb_trees)
    final_xgb = train_xgb_models(X_all, y_all, None, None, NUM_FEATURES, n_xgb_trees)

    final_cat = None
    if CATBOOST_AVAILABLE and cat_models_val is not None:
        n_cat_trees = [_cat_best_iter(m, CAT_PARAMS["iterations"]) for m in cat_models_val]
        final_cat = train_cat_models(X_all, y_all, None, None, NUM_FEATURES, n_cat_trees)

    print(f"   Fertig  |  {time.time()-t0:.1f}s")

    # 6. Test predictions + submission
    print("\n[6/6] Test-Vorhersagen ...")
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
        cat_test = predict_cat(final_cat, X_test, NUM_FEATURES)
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
    print(f"  Gespeichert: {OUT_PATH}")
    print(f"  Zeilen: {len(sub):,}  |  Gesamtzeit: {total_min:.1f} Min.")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
