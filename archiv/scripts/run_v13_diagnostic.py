"""
run_v13_diagnostic.py  –  Drought Severity Prediction: Feature Analysis

Key experiments vs v12:
  1. LightGBM only (fast: ~15 min full / ~5 min quick)
  2. score_lag1/2/3: recent known drought scores as features
     → Score autocorrelation lag1 = 0.966! This is the biggest missing signal.
  3. RECENT_YEARS: limit training to last N years (test: climate-drift hypothesis)
  4. Prints top-40 feature importances to identify noise

Usage:
    python scripts/run_v13_diagnostic.py
Output: outputs/submission_v13_diagnostic.csv
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
OUT_DIR    = ROOT / "outputs"
CACHE_DIR  = DATA_DIR / "precomputed"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

TRAIN_CSV  = DATA_DIR / "train.csv"
TEST_CSV   = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
OUT_PATH   = OUT_DIR / "submission_v13_diagnostic.csv"

WEEKLY_CACHE = CACHE_DIR / "_checkpoint_weekly.npz"

# ─── Experiment knobs ─────────────────────────────────────────────────────────
QUICK_MODE      = True    # True = fast diagnostics, False = full run
RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0

RECENT_YEARS    = None    # None = all data; 5 = last 5 years; 10 = last 10 years
ADD_SCORE_LAGS  = True    # Add last 1-3 known weekly scores as features

WINDOW_STRIDE   = 4 if QUICK_MODE else 1
N_ESTIMATORS    = 500 if QUICK_MODE else 1000

# ─── Feature config (same base as v12) ────────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp", "humidity"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]

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

NUM_FEATURES: list[str] = []


# ─── Feature list ─────────────────────────────────────────────────────────────

def build_feature_list() -> list[str]:
    lag_names  = [f"{c}_lag{lag}" for c in LAG_COLS for lag in LAGS]
    roll_names = [
        f"{col}_roll{w}_{stat}"
        for col in ROLL_COLS for w in ROLL_WINS for stat in ("mean", "std", "max")
    ]
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos", "week_sin", "week_cos"]
    drought  = [
        "prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
        "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d",
    ]
    extra      = ["regional_mean_score"]
    score_lags = ["score_lag1", "score_lag2", "score_lag3"] if ADD_SCORE_LAGS else []
    return WEATHER_COLS + lag_names + roll_names + calendar + drought + extra + score_lags


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


# ─── Feature engineering (identical to v12) ───────────────────────────────────

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


# ─── Dataset assembly ─────────────────────────────────────────────────────────

def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    week = df["ordinal"] // WEEK_BUCKET
    idx  = df.groupby(week, sort=False)["ordinal"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def add_score_lag_columns(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add score_lag1/2/3 from known weekly scores (within each region group)."""
    weekly = weekly.sort_values(["region_id", "ordinal"]).copy()
    # score_lag1 = score at current week (the most recently known score at prediction time)
    # score_lag2 = score one week prior, score_lag3 = two weeks prior
    weekly["score_lag1"] = weekly.groupby("region_id")["score"].transform(lambda x: x).astype(np.float32)
    weekly["score_lag2"] = weekly.groupby("region_id")["score"].shift(1).astype(np.float32)
    weekly["score_lag3"] = weekly.groupby("region_id")["score"].shift(2).astype(np.float32)
    # Fill NaN lags with score_lag1 (conservative: use best available)
    weekly["score_lag2"] = weekly["score_lag2"].fillna(weekly["score_lag1"])
    weekly["score_lag3"] = weekly["score_lag3"].fillna(weekly["score_lag2"])
    return weekly


def build_sliding_windows(weekly, skip_regions, num_features, stride=1, recent_cutoff=None):
    X_parts, y_parts, region_parts = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions:
            continue
        g = g.sort_values("ordinal")
        if recent_cutoff is not None:
            g = g[g["ordinal"] >= recent_cutoff]
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


# ─── Feature importance ───────────────────────────────────────────────────────

def print_feature_importance(models: list, top_n: int = 40) -> None:
    # Feature names come from the model (includes region_id)
    feat_names = np.array(models[0].booster_.feature_name())
    importance  = np.zeros(len(feat_names))
    for m in models:
        importance += m.booster_.feature_importance(importance_type="gain")
    importance /= len(models)
    # Drop region_id from display
    mask       = feat_names != "region_id"
    feat_names = feat_names[mask]
    importance = importance[mask]
    order = np.argsort(importance)[::-1]
    print(f"\n{'─'*60}")
    print(f"  TOP {top_n} FEATURES (mean gain across 5 week-models)")
    print(f"{'─'*60}")
    print(f"  {'Rank':<5s}  {'Feature':<38s}  {'Mean Gain':>12s}")
    print(f"  {'----':<5s}  {'-------':<38s}  {'----------':>12s}")
    for rank, i in enumerate(order[:top_n], 1):
        print(f"  {rank:<5d}  {feat_names[i]:<38s}  {importance[i]:>12.1f}")
    print(f"\n  BOTTOM 10 (noise candidates):")
    for i in order[-10:]:
        print(f"  {'':5s}  {feat_names[i]:<38s}  {importance[i]:>12.1f}")


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 68)
    print("  Drought Severity Prediction  -  run_v13_diagnostic.py")
    print(f"  Mode: {'QUICK' if QUICK_MODE else 'FULL'}  |  score_lags={ADD_SCORE_LAGS}  |  RECENT_YEARS={RECENT_YEARS}")
    print(f"  Features: {len(NUM_FEATURES)}  |  stride={WINDOW_STRIDE}")
    print("=" * 68)

    # 1. Load (from checkpoint if available — skips ~20 min feature engineering)
    if WEEKLY_CACHE.exists():
        print(f"\n[1/6] Loading from cache: {WEEKLY_CACHE.name} ...")
        ck = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base_features = list(ck["feature_names"])
        train_weekly  = pd.DataFrame(ck["weekly_feats"], columns=base_features)
        train_weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        train_weekly["region_id"] = ck["weekly_region"].astype(str)
        train_weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
        X_test_arr      = ck["X_test"].astype(np.float32)
        test_region_ids = ck["test_region_ids"].astype(str)
        print(f"   Loaded {len(train_weekly):,} weekly rows, {len(base_features)} base features  [{elapsed(t0)}]")
        print(f"   (Skipped CSV loading + feature engineering — delete cache to recompute)")

        if ADD_SCORE_LAGS:
            print("   Adding score lag features ...")
            train_weekly = add_score_lag_columns(train_weekly)

        # Build X_test DataFrame with correct feature columns
        base_feat_idx = {f: i for i, f in enumerate(base_features)}
        score_lag_cols = ["score_lag1", "score_lag2", "score_lag3"] if ADD_SCORE_LAGS else []
        NUM_FEATURES_BASE = [f for f in NUM_FEATURES if f not in score_lag_cols]

        # Verify all base features are in cache
        missing = [f for f in NUM_FEATURES_BASE if f not in base_feat_idx]
        if missing:
            print(f"   WARNING: {len(missing)} features not in cache, will be NaN: {missing[:5]}")

        # last_score for persistence baseline and test fill
        last_score = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()

    else:
        print(f"\n[1/6] No cache found — running full feature engineering (~20 min) ...")
        print(f"   Tip: run precompute.py first to create the cache.")
        dtypes = {c: np.float32 for c in WEATHER_COLS}
        train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
        test_raw  = pd.read_csv(TEST_CSV,  dtype=dtypes)
        _parse_dates_inplace(train_raw)
        _parse_dates_inplace(test_raw)
        train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
        regions = train_raw["region_id"].unique()
        print(f"   Train: {len(train_raw):,}  Test: {len(test_raw):,}  Regions: {len(regions)}  [{elapsed(t0)}]")

        region_means = train_raw.groupby("region_id")["score"].mean()

        print("\n[2/6] Feature engineering per region ...")
        train_by_region = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
        test_by_region  = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
        del train_raw, test_raw

        all_tr, all_te = [], []
        n = len(regions)
        for i, region in enumerate(regions, 1):
            if i % 500 == 0 or i == n:
                print(f"   Region {i}/{n}  [{elapsed(t0)}]")
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

        print("\n[3/6] Weekly aggregation ...")
        labeled = train_feat[train_feat["score"].notna()].copy()
        weekly_parts = []
        for region, g in labeled.groupby("region_id", sort=False):
            weekly_parts.append(daily_to_weekly(g))
        train_weekly = pd.concat(weekly_parts, ignore_index=True)
        del labeled

        # Extract test feature vectors (last row per region from test_feat)
        base_features   = [f for f in NUM_FEATURES if not f.startswith("score_lag")]
        X_test_df       = (
            test_feat.sort_values(["region_id", "ordinal"])
            .groupby("region_id", sort=False).tail(1)
            [["region_id"] + base_features].reset_index(drop=True)
        )
        test_region_ids = X_test_df["region_id"].values.astype(str)
        X_test_arr      = X_test_df[base_features].to_numpy(np.float32)
        del test_feat

        if ADD_SCORE_LAGS:
            print("   Adding score lag features ...")
            train_weekly = add_score_lag_columns(train_weekly)

        last_score = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()
        print(f"   {len(train_weekly):,} weekly rows  [{elapsed(t0)}]")

        # Save checkpoint so future runs skip this step
        print(f"   Saving cache to {WEEKLY_CACHE.name} ...")
        np.savez_compressed(
            WEEKLY_CACHE,
            weekly_feats   = train_weekly[base_features].to_numpy(np.float32),
            weekly_scores  = train_weekly["score"].to_numpy(np.float32),
            weekly_region  = train_weekly["region_id"].values.astype(str),
            weekly_ordinal = train_weekly["ordinal"].to_numpy(np.int32),
            X_test         = X_test_arr,
            test_region_ids= test_region_ids,
            feature_names  = np.array(base_features, dtype=object),
        )
        print(f"   Cache saved ({WEEKLY_CACHE.stat().st_size/1e6:.0f} MB) — next run will be ~5 min")

    # 4. Recency filter + train/val split
    print("\n[4/6] Preparing train/val split ...")

    recent_cutoff = None
    if RECENT_YEARS is not None:
        max_ord = int(train_weekly["ordinal"].max())
        recent_cutoff = max_ord - int(RECENT_YEARS * 365)
        n_before = len(train_weekly)
        n_after  = (train_weekly["ordinal"] >= recent_cutoff).sum()
        print(f"   Recency filter ({RECENT_YEARS} years): {n_before:,} → {n_after:,} rows "
              f"({100*n_after/n_before:.0f}%)")

    rng = np.random.default_rng(RANDOM_STATE)
    all_reg = sorted(train_weekly["region_id"].unique())
    n_val   = max(1, int(len(all_reg) * VAL_REGION_FRAC))
    val_regions = set(rng.choice(all_reg, size=n_val, replace=False))

    X_tr, y_tr = build_sliding_windows(
        train_weekly, val_regions, NUM_FEATURES, stride=WINDOW_STRIDE, recent_cutoff=recent_cutoff
    )
    X_va, y_va = build_val_samples(train_weekly, sorted(val_regions), NUM_FEATURES)
    print(f"   Train windows: {len(X_tr):,}  |  Val regions: {len(val_regions)}")

    # Baselines
    persist_va = np.column_stack([
        last_score.reindex(sorted(val_regions)).fillna(0).to_numpy() for _ in range(5)
    ])
    show_mae("Persistence-Baseline (last score repeated)", y_va, persist_va)
    if ADD_SCORE_LAGS:
        # Show how good just regional_mean is vs just score_lag1
        sl1_va = X_va["score_lag1"].to_numpy()
        sl1_pred = np.column_stack([sl1_va for _ in range(5)])
        show_mae("score_lag1 only (oracle-style baseline)", y_va, sl1_pred)

    # 5. Train LightGBM
    print("\n[5/6] Training LightGBM ...")
    lgb_models = []
    for week in range(5):
        p = dict(LGB_PARAMS, random_state=RANDOM_STATE + week)
        m = lgb.LGBMRegressor(**p)
        m.fit(
            X_tr, y_tr[:, week].ravel(),
            eval_set=[(X_va, y_va[:, week].ravel())],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=["region_id"],
        )
        lgb_models.append(m)
        best_it = getattr(m, "best_iteration_", N_ESTIMATORS)
        val_mae = mae(y_va[:, week], np.clip(m.predict(X_va), 0, 5))
        print(f"   Week {week+1}: best_iter={best_it}  val_MAE={val_mae:.4f}")

    lgb_val = np.clip(np.column_stack([m.predict(X_va) for m in lgb_models]), 0, 5).astype(np.float32)
    show_mae("LightGBM (val, all weeks)", y_va, lgb_val)

    # Feature importance
    print_feature_importance(lgb_models, top_n=40)

    # 6. Final training on all data
    print(f"\n[6/6] Final training (all regions) ... [{elapsed(t0)}]")
    X_all, y_all = build_sliding_windows(
        train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE, recent_cutoff=recent_cutoff
    )
    n_iters = [int(getattr(m, "best_iteration_", None) or N_ESTIMATORS) for m in lgb_models]
    final_models = []
    for week in range(5):
        p = dict(LGB_PARAMS, random_state=RANDOM_STATE + week, n_estimators=n_iters[week])
        m = lgb.LGBMRegressor(**p)
        m.fit(X_all, y_all[:, week].ravel(), categorical_feature=["region_id"])
        final_models.append(m)

    # Test predictions — build X_test from cached arrays (works for both paths)
    base_features_in_cache = [f for f in NUM_FEATURES if not f.startswith("score_lag")]
    X_test = pd.DataFrame(X_test_arr, columns=base_features_in_cache)
    X_test["region_id"] = pd.Categorical(test_region_ids)

    if ADD_SCORE_LAGS:
        X_test["score_lag1"] = pd.Categorical(test_region_ids).map(last_score).values.astype(np.float32)
        X_test["score_lag1"] = X_test["region_id"].map(last_score).astype(np.float32).fillna(0)
        X_test["score_lag2"] = X_test["score_lag1"]
        X_test["score_lag3"] = X_test["score_lag1"]

    test_preds = np.clip(
        np.column_stack([m.predict(X_test) for m in final_models]), 0, 5
    ).astype(np.float32)

    sub = pd.DataFrame({"region_id": test_region_ids})
    for k in range(5):
        sub[f"pred_week{k+1}"] = test_preds[:, k]

    template = pd.read_csv(SAMPLE_SUB)
    sub = template[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col] = sub[col].fillna(0.0)

    sub.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*68}")
    print(f"  Saved: {OUT_PATH}")
    print(f"  Rows: {len(sub):,}  |  Total runtime: {elapsed(t0)}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
