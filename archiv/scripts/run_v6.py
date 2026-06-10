"""
run_v6.py  –  Drought Severity Prediction v6

Improvements over v4:
  1. LGB objective = regression_l1 (MAE loss)  — v4 used MSE loss with MAE metric (inconsistent)
  2. Regional monthly climatology z-scores      — "is this month drier/hotter than normal here?"
  3. Year-trend feature                         — captures long-term climate shift signal
  4. Humidity added to lag + rolling features   — strong drought predictor missing from v4
  5. Optuna Bayesian hyperparameter search      — better than manual tuning
  6. No score-based features                    — score_persist causes train/test distribution shift

Usage:
    pip install optuna          # only needed once
    python scripts/run_v6.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"
SAMPLE_SUB = ROOT / "resources" / "sample_submission.csv"
OUT_PATH = OUT_DIR / "submission_v6.csv"

# ─── Mode ─────────────────────────────────────────────────────────────────────
QUICK_MODE = False   # True = ~20 min test run;  False = full Kaggle run (~2-3h with Optuna)
USE_OPTUNA = True    # Bayesian HP search (adds ~15-30 min; well worth it)
OPTUNA_TRIALS = 40   # trials for LGB tuning (tunes on week-1 proxy, applies to all 5)

RANDOM_STATE = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET = 7
DRY_THRESHOLD = 1.0
HORIZONS = 5

WINDOW_STRIDE = 1 if not QUICK_MODE else 4
N_ESTIMATORS = 800 if not QUICK_MODE else 300

# ─── Feature config ───────────────────────────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]  # +humidity vs v4
LAGS = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp", "humidity"]  # +humidity vs v4
ROLL_WINS = [7, 14, 30, 60, 90]

# Z-score columns: for each (region_id, month) we compute mean/std from train, then
# zscore = (current_value - regional_monthly_mean) / regional_monthly_std
# This tells the model: "how unusual is today's weather for this region at this time of year?"
CLIM_COLS = ["prec", "tmp", "humidity", "wind", "surf_pre", "tmp_max", "tmp_min"]

# ─── Default model params ─────────────────────────────────────────────────────
LGB_PARAMS = dict(
    objective="regression_l1",  # MAE loss — matches our evaluation metric (v4 used MSE here)
    metric="mae",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    num_leaves=127,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.2,
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
    print(f"  {name:<55s}  MAE = {mae(y_true, y_pred):.4f}")


# ─── Feature list ─────────────────────────────────────────────────────────────

def build_feature_list() -> list[str]:
    lag_names   = [f"{c}_lag{l}"            for c in LAG_COLS  for l in LAGS]
    roll_names  = [f"{c}_roll{w}_{s}"       for c in ROLL_COLS for w in ROLL_WINS for s in ("mean", "std", "max")]
    calendar    = ["month_sin", "month_cos", "day_sin", "day_cos", "year_norm"]
    drought     = ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
                   "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    zscores     = [f"{c}_zscore" for c in CLIM_COLS]
    return WEATHER_COLS + lag_names + roll_names + calendar + drought + zscores


# ─── Climatology computation ──────────────────────────────────────────────────

def compute_climatology(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Per (region_id, month) mean + std for CLIM_COLS from training data only."""
    stats = train_raw.groupby(["region_id", "month"])[CLIM_COLS].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    return stats.reset_index()


def add_zscore_features(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Merge regional climatology and add z-score features (no data leakage)."""
    merged = df[["region_id", "month"]].merge(clim, on=["region_id", "month"], how="left")
    new_cols: dict = {}
    for col in CLIM_COLS:
        mean_v = merged[f"{col}_mean"].values
        std_v  = merged[f"{col}_std"].fillna(1.0).values
        std_v  = np.where(std_v < 1e-8, 1.0, std_v)
        new_cols[f"{col}_zscore"] = ((df[col].values - mean_v) / std_v).astype(np.float32)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ─── Feature engineering per region ──────────────────────────────────────────

def compute_region_features(
    tr: pd.DataFrame,
    te: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined panel so rolling/lag features flow from train into test without look-ahead."""
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)

    new_cols: dict[str, np.ndarray] = {}

    # Calendar + long-term trend
    new_cols["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["day_sin"]   = np.sin(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    new_cols["day_cos"]   = np.cos(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    new_cols["year_norm"] = ((panel["year"].values - 2000) / 20.0).astype(np.float32)

    # Weather lags (shift(1..21) days)
    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS:
            new_cols[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)

    # Rolling stats — shift(1) before window prevents look-ahead leakage
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=3)
            new_cols[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            new_cols[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            new_cols[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)

    # Drought indices
    prec_p = panel["prec"].shift(1)
    hum_p  = panel["humidity"].shift(1)
    tmp_p  = panel["tmp"].shift(1)

    # Precipitation deficit: negative = drier than annual baseline
    new_cols["prec_deficit_90d"] = (
        prec_p.rolling(90, min_periods=30).mean()
        - prec_p.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    # Precipitation trend: is it becoming drier short-term vs long-term?
    p7   = prec_p.rolling(7, min_periods=3).mean()
    p30  = prec_p.rolling(30, min_periods=10).mean()
    p30s = prec_p.rolling(30, min_periods=10).std().clip(lower=0.01)
    new_cols["prec_trend_30d"] = ((p7 - p30) / p30s).astype(np.float32)

    # Humidity deficit (drier air = more evaporation = drought signal)
    new_cols["humidity_deficit_90d"] = (
        hum_p.rolling(90, min_periods=30).mean()
        - hum_p.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    # Temperature anomaly (heat amplifies drought)
    tmp_anomaly = (
        tmp_p.rolling(90, min_periods=30).mean()
        - tmp_p.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    new_cols["tmp_anomaly_90d"] = tmp_anomaly

    # Interaction: heat × dry → combined drought stress
    new_cols["heat_drought_idx"] = (
        new_cols["prec_deficit_90d"] * tmp_anomaly.clip(lower=0)
    ).astype(np.float32)

    # Dry day counts
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
    """Feature at week i → scores at weeks i+1..i+5.  stride subsamples windows."""
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


# ─── Optuna hyperparameter tuning ─────────────────────────────────────────────

def tune_lgb_params(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    n_trials: int,
) -> dict:
    """Bayesian search over LGB hyperparameters using week-1 as proxy for all 5."""
    if not OPTUNA_AVAILABLE:
        print("  Optuna not installed – skipping. pip install optuna")
        return LGB_PARAMS.copy()

    # Subsample training for faster trials (120k samples, representative)
    max_tr = 120_000
    if len(X_tr) > max_tr:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_tr), max_tr, replace=False)
        X_sub = X_tr.iloc[idx]
        y_sub = y_tr[idx]
    else:
        X_sub, y_sub = X_tr, y_tr

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="regression_l1",
            metric="mae",
            n_estimators=500,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            n_jobs=-1,
            verbose=-1,
            random_state=RANDOM_STATE,
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_sub, y_sub[:, 0].ravel(),
            eval_set=[(X_va, y_va[:, 0].ravel())],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(30, verbose=False)],
            categorical_feature=["region_id"],
        )
        return mae(y_va[:, 0], m.predict(X_va))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(
        f"  Optuna best: lr={best['learning_rate']:.4f}  leaves={best['num_leaves']}  "
        f"child_samples={best['min_child_samples']}  MAE={study.best_value:.4f}"
    )

    result = dict(LGB_PARAMS)
    result.update(best)
    result["n_estimators"] = N_ESTIMATORS
    return result


# ─── Model training ───────────────────────────────────────────────────────────

def train_lgb_models(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame | None, y_va: np.ndarray | None,
    params: dict | None = None,
    n_trees_per_week: list[int] | None = None,
) -> list[lgb.LGBMRegressor]:
    p_base = params if params else LGB_PARAMS
    models = []
    for week in range(HORIZONS):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or p_base["n_estimators"]
        p = dict(p_base, random_state=RANDOM_STATE + week, n_estimators=n)
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
    for week in range(HORIZONS):
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
    for week in range(HORIZONS):
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
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    best_mae_v, best_weights = 999.0, (0.6, 0.4, 0.0)
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
    print("  Natural Disaster Severity Prediction  -  run_v6.py")
    mode_label = "QUICK (~20 min)" if QUICK_MODE else "FULL (~2-3h with Optuna)"
    print(f"  Mode: {mode_label}  |  stride={WINDOW_STRIDE}  estimators={N_ESTIMATORS}")
    opt_label = f"ON ({OPTUNA_TRIALS} trials)" if (USE_OPTUNA and OPTUNA_AVAILABLE) else "OFF"
    cat_label = "ON" if CATBOOST_AVAILABLE else "OFF (pip install catboost)"
    print(f"  Optuna: {opt_label}  |  CatBoost: {cat_label}")
    print(f"  Features: {len(NUM_FEATURES)}  (vs ~107 in v4)")
    print("=" * 68)

    # 1. Load data
    print("\n[1/8] Loading CSV files ...")
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw  = pd.read_csv(TEST_CSV,  dtype=dtypes)
    _parse_dates_inplace(train_raw)
    _parse_dates_inplace(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    print(f"   Train: {len(train_raw):>10,} rows  |  Test: {len(test_raw):>8,} rows")
    print(f"   Regions: {len(regions)}  |  [{elapsed(t0)}]")

    # 2. Regional monthly climatology (training data only — no leakage)
    print("\n[2/8] Computing regional monthly climatology ...")
    clim_df = compute_climatology(train_raw)
    print(f"   {len(clim_df):,} (region, month) climatology entries  [{elapsed(t0)}]")

    # 3. Feature engineering per region
    print("\n[3/8] Feature engineering per region ...")
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
    print(f"   Done  [{elapsed(t0)}]")

    # 4. Add climatology z-scores
    print("\n[4/8] Adding climatology z-scores ...")
    train_feat = add_zscore_features(train_feat, clim_df)
    test_feat  = add_zscore_features(test_feat,  clim_df)
    del clim_df
    print(f"   Done  [{elapsed(t0)}]")

    # 5. Weekly aggregation
    print("\n[5/8] Weekly aggregation ...")
    labeled = train_feat[train_feat["score"].notna()].copy()
    weekly_parts = []
    for region, g in labeled.groupby("region_id", sort=False):
        weekly_parts.append(daily_to_weekly(g))
    train_weekly = pd.concat(weekly_parts, ignore_index=True)
    del labeled
    weeks_per_region = int(len(train_weekly) / len(regions))
    print(f"   {len(train_weekly):,} weekly rows  (~{weeks_per_region}/region)  [{elapsed(t0)}]")

    # 6. Train/val split (region holdout 20%)
    print("\n[6/8] Building train/val split ...")
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

    # 7. Optuna hyperparameter tuning for LGB
    lgb_tuned_params = LGB_PARAMS.copy()
    if USE_OPTUNA and OPTUNA_AVAILABLE:
        print(f"\n  Optuna LGB tuning ({OPTUNA_TRIALS} trials, week-1 proxy) ...")
        t_opt = time.time()
        lgb_tuned_params = tune_lgb_params(X_tr, y_tr, X_va, y_va, OPTUNA_TRIALS)
        print(f"  Optuna done  [{elapsed(t_opt)}]")
    elif USE_OPTUNA and not OPTUNA_AVAILABLE:
        print("\n  Optuna not available – using default LGB params. Install: pip install optuna")

    # 8. Train models
    print("\n[7/8] Training models ...")

    print("  Training LightGBM ...")
    lgb_models = train_lgb_models(X_tr, y_tr, X_va, y_va, lgb_tuned_params)
    lgb_val = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (val)", y_va, lgb_val)

    print("  Training XGBoost ...")
    xgb_models = train_xgb_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
    xgb_val = predict_xgb(xgb_models, X_va, NUM_FEATURES)
    show_mae("XGBoost (val)", y_va, xgb_val)

    cat_val = None
    cat_models_val = None
    if CATBOOST_AVAILABLE:
        print("  Training CatBoost ...")
        cat_models_val = train_cat_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
        cat_val = predict_cat(cat_models_val, X_va, NUM_FEATURES)
        show_mae("CatBoost (val)", y_va, cat_val)

    print("  Blend optimization ...")
    best_weights, best_mae_val = optimize_blend(y_va, lgb_val, xgb_val, cat_val)
    lgb_w, xgb_w, cat_w = best_weights
    if cat_val is not None:
        print(f"  LGB={lgb_w:.2f}  XGB={xgb_w:.2f}  CAT={cat_w:.2f}   MAE={best_mae_val:.4f}")
    else:
        print(f"  LGB={lgb_w:.2f}  XGB={xgb_w:.2f}   MAE={best_mae_val:.4f}")

    # Final training on all data
    print("\n  Final training (all regions) ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE)

    n_lgb_trees = [int(getattr(m, "best_iteration_", None) or lgb_tuned_params["n_estimators"]) for m in lgb_models]
    n_xgb_trees = [int(getattr(m, "best_iteration",  None) or XGB_PARAMS["n_estimators"])       for m in xgb_models]

    final_lgb = train_lgb_models(X_all, y_all, None, None, lgb_tuned_params, n_lgb_trees)
    final_xgb = train_xgb_models(X_all, y_all, None, None, NUM_FEATURES, n_xgb_trees)

    final_cat = None
    if CATBOOST_AVAILABLE and cat_models_val is not None:
        n_cat_trees = [_cat_best_iter(m, CAT_PARAMS["iterations"]) for m in cat_models_val]
        final_cat = train_cat_models(X_all, y_all, None, None, NUM_FEATURES, n_cat_trees)

    print(f"  Done  [{elapsed(t0)}]")

    # Test predictions + submission
    print("\n[8/8] Test predictions ...")
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
    for k in range(HORIZONS):
        sub[f"pred_week{k+1}"] = test_preds[:, k]

    template = pd.read_csv(SAMPLE_SUB)
    sub = template[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(HORIZONS)]:
        sub[col] = sub[col].fillna(0.0)

    sub.to_csv(OUT_PATH, index=False)

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*68}")
    print(f"  Saved: {OUT_PATH}")
    print(f"  Rows: {len(sub):,}  |  Total: {total_min:.1f} Min.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
