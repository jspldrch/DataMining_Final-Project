"""
kaggle_v29_fwd_seasonal.py  --  Forward Seasonal Features + Holdout 20% Val
=============================================================================
Base:        v24_seasonal  (RECENT_YEARS=8, regional_seasonal_mean, 134 features)
Kaggle best: recent_local_8y = 0.8095  |  v24 = 0.8106  |  v22 = 0.8132
Target:      Baseline 3 = 0.8056

THREE CHANGES vs v24:
  1. NEW  +rsm_fw_wk{1..5}  →  139 features total
     regional_seasonal_mean encodes the avg score for the CURRENT month M.
     But the 5 target weeks may span months M+1 or M+2 — the model didn't know
     what drought severity is seasonally expected for each prediction step.
     rsm_fw_wk{k} = avg_score[(region, month_of_training_week(i+k))]
     For test: future month k = _future_month(last_month, last_day, k)
     using the proprietary 31-days-per-month ordinal calendar arithmetic.

  2. FIX  Test row = idxmax(ordinal) in last ordinal bucket
     Mirrors _daily_to_weekly logic used for ALL training rows.
     v22/v24 used tail(1) which may not land on the "scored weekday".
     (Effect is minor for consecutive 91d data but keeps pipeline consistent.)

  3. VAL  20% region holdout  (same as recent_local_8y → best known 0.8095)
     80% of regions → training windows (stride 1)
     20% of regions → val (last window only, cross-region generalization test)
     v24 used last-window of ALL regions → less realistic Kaggle proxy.

Dataset slug:  /kaggle/input/datafinal/{train,test}.npz
               /kaggle/input/datafinal/sample_submission.csv
Output:        /kaggle/working/submission_v29_fwd_seasonal.csv
"""
from __future__ import annotations
import time, warnings
from pathlib import Path
import glob as _g

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    CAT = True
except ImportError:
    CAT = False

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR      = Path("/kaggle/working")
WEEKLY_CACHE  = WORK_DIR / "cache_weekly_v29.npz"   # new cols → separate file
WINDOWS_CACHE = WORK_DIR / "cache_windows_v29.npz"  # new val scheme → separate file
OUT_PATH      = WORK_DIR / "submission_v29_fwd_seasonal.csv"

def _find_npz(name: str) -> Path:
    """Look for train.npz / test.npz, datafinal slug first."""
    for slug in ["datafinal", "datafiles", "datatrain", "datatest",
                 "traindataset", "testdataset", "trainthis", "testset", "data"]:
        p = Path(f"/kaggle/input/{slug}/{name}")
        if p.exists(): return p
    found = sorted(_g.glob(f"/kaggle/input/**/{name}", recursive=True))
    if found: return Path(found[0])
    p = WORK_DIR / name
    if p.exists(): return p
    avail = sorted(str(x) for x in Path("/kaggle/input/").iterdir()) \
            if Path("/kaggle/input/").exists() else ["(empty)"]
    raise FileNotFoundError(f"'{name}' not found.\nAvailable input dirs:\n  " +
                            "\n  ".join(avail))

def _find_sample_sub() -> Path | None:
    for p in [
        "/kaggle/input/datafinal/sample_submission.csv",
        "/kaggle/input/samplesub/sample_submission.csv",
        "/kaggle/input/samplesubmission/sample_submission.csv",
        "/kaggle/input/sample-submission/sample_submission.csv",
    ]:
        if Path(p).exists(): return Path(p)
    found = _g.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(sorted(found)[0]) if found else None

SAMPLE_SUB = _find_sample_sub()

# ── Knobs ──────────────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
HOLDOUT_FRAC     = 0.20        # 20% of regions held out as val
HOLDOUT_SEED     = 42
WEEK_BUCKET      = 7
DRY_THRESHOLD    = 1.0
WINDOW_STRIDE    = 1
N_ESTIMATORS     = 1000
RECENT_YEARS     = 8
ORDINAL_PER_YEAR = 372
DAYS_PER_MONTH   = 31          # proprietary calendar: all months = 31 ordinal days

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "humidity", "tmp", "wind"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]

LGB_P = dict(
    objective="regression", metric="mae", n_estimators=N_ESTIMATORS,
    learning_rate=0.04, num_leaves=127, min_child_samples=60,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, verbose=-1,
)
XGB_P = dict(
    objective="reg:squarederror", n_estimators=N_ESTIMATORS, learning_rate=0.04,
    max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", n_jobs=-1, verbosity=0,
)
CAT_P = dict(
    iterations=N_ESTIMATORS, learning_rate=0.04, depth=6,
    loss_function="MAE", eval_metric="MAE",
    random_seed=RANDOM_STATE, verbose=False, thread_count=-1,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f}m" if s >= 60 else f"{s:.0f}s"

def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))

def show(name: str, y: np.ndarray, p: np.ndarray) -> None:
    print(f"  {name:<54s}  MAE={mae(y, p):.4f}")

def _best_n(m, default: int) -> int:
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(m, attr, None)
        if v is not None: return int(v)
    try: return int(m.get_best_iteration())
    except: return default

# ── Feature list (139 features) ────────────────────────────────────────────────
def build_features() -> list[str]:
    f  = list(WEATHER_COLS)                                          # 14
    f += [f"{c}_lag{l}"       for c in LAG_COLS for l in LAGS]      # 35
    f += [f"{c}_roll{w}_{s}"  for c in ROLL_COLS for w in ROLL_WINS  # 72
          for s in ("mean", "std", "max")]
    f += ["month_sin", "month_cos", "day_sin", "day_cos"]            # 4
    f += ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",  # 7
          "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    f.append("regional_mean_score")                                  # 1
    f.append("regional_seasonal_mean")                               # 1  current month (v24)
    f += [f"rsm_fw_wk{k}" for k in range(1, 6)]                     # 5  NEW forward months
    return f  # total: 139

# ── Ordinal calendar: future month calculation ─────────────────────────────────
def _future_month(month: int, day: int, k_weeks: int) -> int:
    """
    Month k_weeks ahead in the proprietary 31-days-per-month ordinal calendar.
    ordinal = year*372 + month*31 + day  →  adding 7 to ordinal = 7 calendar days.
    Within a year: month*31 + day ∈ [32, 403], so advancing k*7 ordinal days:
        total = (month-1)*31 + day + k_weeks*7
        new_month = ((total-1) // 31) % 12 + 1
    Verified correct at month and year boundaries.
    """
    total = (int(month) - 1) * DAYS_PER_MONTH + int(day) + k_weeks * 7
    return ((total - 1) // DAYS_PER_MONTH) % 12 + 1

# ── NPZ loading ────────────────────────────────────────────────────────────────
def load_npz(path: Path) -> pd.DataFrame:
    d = np.load(path, allow_pickle=True)
    names = d["region_names"]
    df = pd.DataFrame({
        "region_id": names[d["region_id"]],
        "year":      d["year"].astype(np.int32),
        "month":     d["month"].astype(np.int32),
        "day":       d["day"].astype(np.int32),
    })
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]
    for col in WEATHER_COLS:
        if col in d: df[col] = d[col].astype(np.float32)
    if "score" in d: df["score"] = d["score"].astype(np.float32)
    return df

# ── Feature engineering (per region, identical to v22/v24) ────────────────────
def _region_features(tr: pd.DataFrame, te: pd.DataFrame):
    te = te.copy(); te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc = {}
    nc["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    nc["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    nc["day_sin"]   = np.sin(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    nc["day_cos"]   = np.cos(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    for col in LAG_COLS:
        for lag in LAGS:
            nc[f"{col}_lag{lag}"] = panel[col].shift(lag).astype(np.float32)
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=max(3, w // 10))
            nc[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            nc[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            nc[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)
    pp   = panel["prec"].shift(1)
    nc["prec_deficit_90d"]    = (pp.rolling(90,  min_periods=30).mean() -
                                  pp.rolling(365, min_periods=60).mean()).astype(np.float32)
    p7   = pp.rolling(7,  min_periods=3).mean()
    p30  = pp.rolling(30, min_periods=10).mean()
    nc["prec_trend_30d"]      = ((p7 - p30) /
                                  pp.rolling(30, min_periods=10).std().clip(lower=0.01)).astype(np.float32)
    hp   = panel["humidity"].shift(1)
    nc["humidity_deficit_90d"]= (hp.rolling(90,  min_periods=30).mean() -
                                  hp.rolling(365, min_periods=60).mean()).astype(np.float32)
    tp   = panel["tmp"].shift(1)
    anom = (tp.rolling(90, min_periods=30).mean() -
            tp.rolling(365, min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]     = anom
    nc["heat_drought_idx"]    = (nc["prec_deficit_90d"] * anom.clip(lower=0)).astype(np.float32)
    dry  = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    nc["dry_days_14d"]        = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"]        = dry.rolling(30, min_periods=7).sum().astype(np.float32)
    panel = pd.concat([panel, pd.DataFrame(nc, index=panel.index)], axis=1)
    n = len(tr)
    return panel.iloc[:n].copy(), panel.iloc[n:].copy()

def _daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Select last row within each ordinal // WEEK_BUCKET group (= scored day)."""
    wk = df["ordinal"] // WEEK_BUCKET
    return df.loc[df.groupby(wk, sort=False)["ordinal"].idxmax()].reset_index(drop=True)

# ── Weekly cache (builds once, ~20 min; reused on subsequent runs) ─────────────
def load_weekly(t0: float):
    if WEEKLY_CACHE.exists():
        print(f"  [Cache] Weekly v29: {WEEKLY_CACHE.stat().st_size / 1e6:.0f} MB")
        ck   = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base = list(ck["feature_names"])
        weekly = pd.DataFrame(ck["weekly_feats"], columns=base)
        weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        weekly["region_id"] = ck["weekly_region"].astype(str)
        weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
        return weekly, ck["X_test_base"].astype(np.float32), ck["test_region_ids"].astype(str), base

    print(f"  No cache — full feature engineering (~20 min) ... [{elapsed(t0)}]")
    train_raw = load_npz(_find_npz("train.npz"))
    test_raw  = load_npz(_find_npz("test.npz"))
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)

    regions      = train_raw["region_id"].unique()
    region_means = train_raw.groupby("region_id")["score"].mean()
    tr_by = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    te_by = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw

    all_tr, all_te = [], []
    for i, region in enumerate(regions, 1):
        tf, ef = _region_features(tr_by[region], te_by.get(region, pd.DataFrame()))
        all_tr.append(tf); all_te.append(ef)
        if i % 500 == 0 or i == len(regions):
            print(f"    {i}/{len(regions)} regions [{elapsed(t0)}]")
    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te

    # ── Regional mean score (all-time average per region) ──
    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)

    # ── Seasonal means: avg score per (region, month) from full 15-year history ──
    labeled = train_feat[train_feat["score"].notna()].copy()
    s_ser   = labeled.groupby(["region_id", "month"])["score"].mean()
    s_map   = s_ser.to_dict()                      # {(region_id, month): mean_score}
    fallback= region_means.to_dict()               # {region_id: mean_score}  (backup)

    # Current-month seasonal mean (v24 feature, kept for continuity)
    labeled["regional_seasonal_mean"] = np.array(
        [s_map.get((r, int(m)), fallback.get(r, 0.0))
         for r, m in zip(labeled["region_id"], labeled["month"])],
        dtype=np.float32,
    )

    # ── Weekly aggregation: one row per scored day per region ──
    weekly = pd.concat(
        [_daily_to_weekly(g) for _, g in labeled.groupby("region_id", sort=False)],
        ignore_index=True,
    )
    del labeled

    # ── NEW: forward seasonal features rsm_fw_wk{1..5} ──────────────────────
    # rsm_fw_wk{k}[i] = avg_score[(region, month_of_week(i+k))]
    # Tells each model what drought level is seasonally expected at target week k.
    # Capped at last known month when i+k >= n (edge case for tail windows).
    weekly = weekly.sort_values(["region_id", "ordinal"]).reset_index(drop=True)
    fw_bufs = {
        f"rsm_fw_wk{k}": np.zeros(len(weekly), dtype=np.float32)
        for k in range(1, 6)
    }
    for region, g in weekly.groupby("region_id", sort=True):
        idx    = g.index.tolist()          # row indices in the full weekly df
        months = g["month"].tolist()       # months in ordinal-sorted order
        n      = len(months)
        for k in range(1, 6):
            col = f"rsm_fw_wk{k}"
            for i in range(n):
                m_fwd = months[min(i + k, n - 1)]
                fw_bufs[col][idx[i]] = s_map.get(
                    (region, int(m_fwd)), fallback.get(region, 0.0)
                )
    for k in range(1, 6):
        weekly[f"rsm_fw_wk{k}"] = fw_bufs[f"rsm_fw_wk{k}"]

    # base_cols: all feature columns; excludes metadata (month/day/year/ordinal/score/id)
    base_cols = [c for c in weekly.columns
                 if c not in ("score", "region_id", "ordinal", "date", "year", "month", "day")]

    # ── FIX: test feature row = idxmax(ordinal) within last ordinal bucket ────
    # Mirrors _daily_to_weekly: always selects the "scored-day equivalent" row.
    # v22/v24 used tail(1) which is nearly identical for consecutive 91d data
    # but this is strictly more consistent with how training rows are selected.
    test_parts = []
    for region, g in test_feat.groupby("region_id", sort=False):
        g        = g.sort_values("ordinal")
        last_ord = int(g["ordinal"].max())
        bucket   = last_ord // WEEK_BUCKET
        mask     = g["ordinal"] // WEEK_BUCKET == bucket
        row      = g.loc[[g.loc[mask, "ordinal"].idxmax()]]
        test_parts.append(row)
    X_test_df = pd.concat(test_parts, ignore_index=True)

    # Map current-month seasonal mean to test row
    X_test_df["regional_seasonal_mean"] = np.array(
        [s_map.get((r, int(m)), fallback.get(r, 0.0))
         for r, m in zip(X_test_df["region_id"], X_test_df["month"])],
        dtype=np.float32,
    )

    # ── NEW: forward seasonal for test ──────────────────────────────────────
    # future month k = _future_month(last_test_month, last_test_day, k)
    for k in range(1, 6):
        X_test_df[f"rsm_fw_wk{k}"] = np.array(
            [s_map.get(
                (r, _future_month(int(m), int(d), k)),
                fallback.get(r, 0.0),
             )
             for r, m, d in zip(
                 X_test_df["region_id"], X_test_df["month"], X_test_df["day"]
             )],
            dtype=np.float32,
        )

    test_ids = X_test_df["region_id"].values.astype(str)
    X_test   = X_test_df[base_cols].to_numpy(np.float32)

    np.savez_compressed(
        WEEKLY_CACHE,
        weekly_feats    = weekly[base_cols].to_numpy(np.float32),
        weekly_scores   = weekly["score"].to_numpy(np.float32),
        weekly_region   = weekly["region_id"].values.astype(str),
        weekly_ordinal  = weekly["ordinal"].to_numpy(np.int32),
        X_test_base     = X_test,
        test_region_ids = test_ids,
        feature_names   = np.array(base_cols, dtype=object),
    )
    print(f"  Weekly cache v29 saved [{elapsed(t0)}]")
    return weekly, X_test, test_ids, base_cols

# ── Recent filter: keep only last RECENT_YEARS per region ─────────────────────
def filter_recent_per_region(weekly: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in weekly.groupby("region_id", sort=False):
        cutoff = int(g["ordinal"].max()) - RECENT_YEARS * ORDINAL_PER_YEAR
        parts.append(g[g["ordinal"] >= cutoff])
    return pd.concat(parts, ignore_index=True)

# ── Val: 20% region holdout  (= recent_local scheme) ──────────────────────────
def build_holdout_local_windows(weekly_recent: pd.DataFrame, features: list[str]):
    """
    Hold out HOLDOUT_FRAC of regions entirely from training.
    Val = last window of each holdout region (cross-region generalization).
    Train = all windows of remaining 80% of regions.
    All = all windows of all regions (used for final model after val is done).
    """
    rng         = np.random.default_rng(HOLDOUT_SEED)
    all_regions = weekly_recent["region_id"].unique()
    n_val       = max(1, int(len(all_regions) * HOLDOUT_FRAC))
    val_set     = set(rng.choice(all_regions, n_val, replace=False))
    n_train_reg = len(all_regions) - n_val
    print(f"  Holdout split: {n_train_reg} train regions / {n_val} val regions")

    Xtr, ytr, rtr = [], [], []
    Xva, yva, rva = [], [], []
    Xal, yal, ral = [], [], []

    for region, g in weekly_recent.groupby("region_id", sort=False):
        g  = g.sort_values("ordinal")
        sc = g["score"].to_numpy(np.float32)
        Xn = g[features].to_numpy(np.float32)
        n  = len(g)
        if n < 6: continue
        nw = n - 5
        yr = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]

        # All windows: for final training after val calibration
        idx_all = list(range(0, nw, WINDOW_STRIDE))
        if (nw - 1) not in idx_all: idx_all.append(nw - 1)
        Xal.append(Xn[idx_all]); yal.append(yr[idx_all]); ral.extend([region] * len(idx_all))

        if region in val_set:
            # Val region: only last window, never seen during training
            Xva.append(Xn[nw - 1]); yva.append(yr[nw - 1]); rva.append(region)
        else:
            # Train region: all windows (stride 1)
            idx = list(range(0, nw, WINDOW_STRIDE))
            if (nw - 1) not in idx: idx.append(nw - 1)
            Xtr.append(Xn[idx]); ytr.append(yr[idx]); rtr.extend([region] * len(idx))

    X_tr  = pd.DataFrame(np.vstack(Xtr).astype(np.float32), columns=features)
    X_tr["region_id"]  = pd.Categorical(rtr)
    X_va  = pd.DataFrame(np.vstack(Xva).astype(np.float32), columns=features)
    X_va["region_id"]  = pd.Categorical(rva)
    X_all = pd.DataFrame(np.vstack(Xal).astype(np.float32), columns=features)
    X_all["region_id"] = pd.Categorical(ral)
    return (
        X_tr,  np.vstack(ytr).astype(np.float32),
        X_va,  np.vstack(yva).astype(np.float32),
        X_all, np.vstack(yal).astype(np.float32),
    )

def load_or_build_windows(weekly_recent: pd.DataFrame, features: list[str], t0: float):
    if WINDOWS_CACHE.exists():
        ck = dict(np.load(WINDOWS_CACHE, allow_pickle=True))
        if list(ck["feature_names"]) == features:
            print(f"  [Cache] Windows v29: {WINDOWS_CACHE.stat().st_size / 1e6:.0f} MB")
            def _r(p):
                X = pd.DataFrame(ck[f"X_{p}"], columns=features)
                X["region_id"] = pd.Categorical(ck[f"r_{p}"].astype(str).tolist())
                return X, ck[f"y_{p}"]
            return *_r("tr"), *_r("va"), *_r("all")
        print("  Windows cache outdated — rebuilding ...")

    print(f"  Building windows (holdout 20% val, recent {RECENT_YEARS}y) [{elapsed(t0)}]")
    X_tr, y_tr, X_va, y_va, X_all, y_all = build_holdout_local_windows(weekly_recent, features)
    np.savez_compressed(
        WINDOWS_CACHE,
        X_tr  = X_tr[features].to_numpy(np.float32),  y_tr  = y_tr,
        r_tr  = np.array(X_tr["region_id"].astype(str),  dtype=object),
        X_va  = X_va[features].to_numpy(np.float32),  y_va  = y_va,
        r_va  = np.array(X_va["region_id"].astype(str),  dtype=object),
        X_all = X_all[features].to_numpy(np.float32), y_all = y_all,
        r_all = np.array(X_all["region_id"].astype(str), dtype=object),
        feature_names = np.array(features, dtype=object),
    )
    print(f"  Windows cache v29 saved [{elapsed(t0)}]")
    return X_tr, y_tr, X_va, y_va, X_all, y_all

# ── Training ───────────────────────────────────────────────────────────────────
def train_lgb(X_tr, y_tr, X_va, y_va, n_trees=None):
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or LGB_P["n_estimators"]
        m = lgb.LGBMRegressor(**dict(LGB_P, random_state=RANDOM_STATE + wk, n_estimators=n))
        kw = dict(categorical_feature=["region_id"])
        if X_va is not None:
            kw.update(
                eval_set=[(X_va, y_va[:, wk].ravel())], eval_metric="mae",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        m.fit(X_tr, y_tr[:, wk].ravel(), **kw)
        models.append(m)
    return models

def train_xgb(X_tr, y_tr, X_va, y_va, features, n_trees=None):
    Xn = X_tr[features].to_numpy(np.float32)
    Vn = X_va[features].to_numpy(np.float32) if X_va is not None else None
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or XGB_P["n_estimators"]
        p = dict(XGB_P, random_state=RANDOM_STATE + wk, n_estimators=n)
        kw = {}
        if Vn is not None:
            p["early_stopping_rounds"] = 50
            kw.update(eval_set=[(Vn, y_va[:, wk].ravel())], verbose=False)
        m = xgb.XGBRegressor(**p)
        m.fit(Xn, y_tr[:, wk].ravel(), **kw)
        models.append(m)
    return models

def train_cat(X_tr, y_tr, X_va, y_va, features, n_trees=None):
    if not CAT: return None
    Xn = X_tr[features].to_numpy(np.float32)
    Vn = X_va[features].to_numpy(np.float32) if X_va is not None else None
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or CAT_P["iterations"]
        p = dict(CAT_P, iterations=n, random_seed=RANDOM_STATE + wk)
        kw = {}
        if Vn is not None:
            kw.update(eval_set=(Vn, y_va[:, wk].ravel()), early_stopping_rounds=50)
        m = CatBoostRegressor(**p)
        m.fit(Xn, y_tr[:, wk].ravel(), **kw)
        models.append(m)
    return models

def pred_lgb(models, X: pd.DataFrame) -> np.ndarray:
    feat = models[0].booster_.feature_name()
    return np.clip(
        np.column_stack([m.predict(X[feat]) for m in models]), 0, 5
    ).astype(np.float32)

def pred_num(models, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    Xn = X[features].to_numpy(np.float32)
    return np.clip(
        np.column_stack([m.predict(Xn) for m in models]), 0, 5
    ).astype(np.float32)

def blend(y_va: np.ndarray, preds: dict) -> tuple[dict, float]:
    names  = list(preds); arrays = [preds[n] for n in names]
    alphas = [round(x * 0.05, 2) for x in range(1, 20)]
    best_mae, best_w = 999.0, {n: 1 / len(names) for n in names}
    if len(names) == 2:
        for a in alphas:
            m = mae(y_va, a * arrays[0] + (1 - a) * arrays[1])
            if m < best_mae:
                best_mae, best_w = m, {names[0]: a, names[1]: round(1 - a, 8)}
    elif len(names) == 3:
        for a in alphas:
            for b in alphas:
                c = round(1 - a - b, 8)
                if c < 0.05: continue
                m = mae(y_va, a * arrays[0] + b * arrays[1] + c * arrays[2])
                if m < best_mae:
                    best_mae, best_w = m, {names[0]: a, names[1]: b, names[2]: c}
    return best_w, best_mae

def print_importance(lgb_models, features: list[str]) -> None:
    feat  = np.array(lgb_models[0].booster_.feature_name())
    imp   = sum(m.booster_.feature_importance("gain") for m in lgb_models) / len(lgb_models)
    mask  = feat != "region_id"
    feat  = feat[mask]; imp = imp[mask]
    total = imp.sum(); order = np.argsort(imp)[::-1]
    print(f"\n{'='*62}")
    print(f"  FEATURE IMPORTANCE (LGB Gain, avg weeks 1-5, top 25)")
    print(f"  {'Rank':<4}  {'Feature':<38}  {'%':>6}")
    for rank, i in enumerate(order[:25], 1):
        tag = " ◄ NEW" if feat[i].startswith("rsm_fw") else ""
        print(f"  {rank:<4d}  {feat[i]:<38}  {100*imp[i]/total:>5.2f}%{tag}")
    groups = {
        "Weather raw":        [f in WEATHER_COLS                                        for f in feat],
        "Rolling stats":      ["roll" in f                                              for f in feat],
        "Lags":               ["_lag" in f                                              for f in feat],
        "Drought indices":    [any(k in f for k in
                               ["deficit","trend","anomaly","drought","dry_days"])       for f in feat],
        "Regional mean":      [f == "regional_mean_score"                               for f in feat],
        "Seasonal current":   [f == "regional_seasonal_mean"                            for f in feat],
        "Seasonal forward ◄": [f.startswith("rsm_fw_wk")                               for f in feat],
    }
    print(f"\n  Group totals:")
    for gname, mask_list in groups.items():
        g_imp = imp[[i for i, v in enumerate(mask_list) if v]].sum()
        print(f"    {gname:<26}  {100*g_imp/total:>5.1f}%")
    print(f"  Top-10 cumulative:            {100*imp[order[:10]].sum()/total:.1f}%")
    print(f"{'='*62}\n")

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    t0       = time.time()
    FEATURES = build_features()

    print("=" * 62)
    print(f"  kaggle_v29_fwd_seasonal  |  {len(FEATURES)} features  |  no score_lag")
    print(f"  RECENT_YEARS={RECENT_YEARS}  |  Holdout={HOLDOUT_FRAC*100:.0f}%  |  seed={HOLDOUT_SEED}")
    print(f"  NEW: rsm_fw_wk{{1..5}} — seasonal mean for each target week's month")
    print(f"  FIX: test row = idxmax(ordinal) in last ordinal-bucket (not tail(1))")
    print(f"  VAL: 20% region holdout  →  best known Kaggle proxy (0.8095)")
    print(f"  Dataset slug: datafinal  →  {_find_npz('train.npz').parent}")
    print("=" * 62)

    # 1. Load weekly features (builds from scratch if no cache)
    print(f"\n[1/5] Weekly features ... [{elapsed(t0)}]")
    weekly, X_test_base, test_ids, base_cols = load_weekly(t0)
    n_regions   = weekly["region_id"].nunique()
    n_weeks_all = len(weekly)
    print(f"  {n_weeks_all:,} weekly rows  |  {n_regions} regions  |  {len(base_cols)} base cols")

    # Ensure all expected features exist (fill 0 if a col is missing from cache)
    for f in FEATURES:
        if f not in weekly.columns:
            weekly[f] = np.float32(0)

    # 2. Recent filter: last RECENT_YEARS per region
    print(f"\n[2/5] Recent filter: last {RECENT_YEARS} years per region ... [{elapsed(t0)}]")
    weekly_recent = filter_recent_per_region(weekly)
    n_recent = len(weekly_recent)
    pct      = 100 * n_recent / n_weeks_all
    print(f"  Full history : {n_weeks_all:,}  (~{n_weeks_all/n_regions:.0f}/region)")
    print(f"  After filter : {n_recent:,}  (~{n_recent/n_regions:.0f}/region)  [{pct:.0f}% retained]")

    # 3. Sliding windows with holdout val
    print(f"\n[3/5] Build windows (holdout 20% val) ... [{elapsed(t0)}]")
    X_tr, y_tr, X_va, y_va, X_all, y_all = load_or_build_windows(weekly_recent, FEATURES, t0)
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")

    # Persistence baseline on val
    last_score  = weekly_recent.sort_values("ordinal").groupby("region_id")["score"].last()
    val_regions = X_va["region_id"].astype(str).tolist()
    persist     = np.column_stack([last_score.reindex(val_regions).fillna(0).to_numpy()] * 5)
    show("Persistence (last score × 5)", y_va, persist)

    # 4. Training
    print(f"\n[4/5] Training LGB / XGB / CatBoost ... [{elapsed(t0)}]")
    lgb_m   = train_lgb(X_tr, y_tr, X_va, y_va)
    lgb_val = pred_lgb(lgb_m, X_va)
    show("LightGBM", y_va, lgb_val)
    lgb_iters = [_best_n(m, N_ESTIMATORS) for m in lgb_m]
    for wk in range(5):
        feat = lgb_m[wk].booster_.feature_name()
        v    = mae(y_va[:, wk], np.clip(lgb_m[wk].predict(X_va[feat]), 0, 5))
        hit  = "  ← HIT LIMIT" if lgb_iters[wk] >= N_ESTIMATORS - 5 else ""
        print(f"    LGB Week {wk+1}: iter={lgb_iters[wk]:4d}  MAE={v:.4f}{hit}")

    xgb_m   = train_xgb(X_tr, y_tr, X_va, y_va, FEATURES)
    xgb_val = pred_num(xgb_m, X_va, FEATURES)
    show("XGBoost", y_va, xgb_val)

    preds_val = {"lgb": lgb_val, "xgb": xgb_val}
    cat_m = train_cat(X_tr, y_tr, X_va, y_va, FEATURES)
    if cat_m:
        cat_val = pred_num(cat_m, X_va, FEATURES)
        show("CatBoost", y_va, cat_val)
        preds_val["cat"] = cat_val

    best_w, best_val_mae = blend(y_va, preds_val)
    print(f"  Blend weights: {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}  "
          f"MAE={best_val_mae:.4f}")
    print_importance(lgb_m, FEATURES)

    # 5. Final training on ALL data + submission
    print(f"\n[5/5] Final training + submission ... [{elapsed(t0)}]")
    n_lgb = [_best_n(m, N_ESTIMATORS) for m in lgb_m]
    n_xgb = [_best_n(m, N_ESTIMATORS) for m in xgb_m]
    f_lgb = train_lgb(X_all, y_all, None, None, n_lgb)
    f_xgb = train_xgb(X_all, y_all, None, None, FEATURES, n_xgb)
    f_cat = None
    if cat_m:
        n_cat = [_best_n(m, N_ESTIMATORS) for m in cat_m]
        f_cat = train_cat(X_all, y_all, None, None, FEATURES, n_cat)

    X_test = pd.DataFrame(X_test_base, columns=base_cols)
    X_test["region_id"] = pd.Categorical(test_ids)
    for f in FEATURES:
        if f not in X_test.columns: X_test[f] = np.float32(0)

    test_preds = (best_w["lgb"] * pred_lgb(f_lgb, X_test) +
                  best_w["xgb"] * pred_num(f_xgb, X_test, FEATURES))
    if f_cat and "cat" in best_w:
        test_preds += best_w["cat"] * pred_num(f_cat, X_test, FEATURES)

    sub = pd.DataFrame({"region_id": test_ids})
    for k in range(5):
        sub[f"pred_week{k+1}"] = test_preds[:, k]
    if SAMPLE_SUB:
        template = pd.read_csv(SAMPLE_SUB)[["region_id"]]
        sub = template.merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col] = sub[col].fillna(0.0)
    sub.to_csv(OUT_PATH, index=False)
    print(f"  Saved: {OUT_PATH.name}  ({len(sub):,} rows × 6 cols)")

    # ── Final summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  RESULTS — kaggle_v29_fwd_seasonal")
    print(f"  {'-'*58}")
    print(f"  {'Features':.<35} {len(FEATURES)}  (v24: 134, +5 rsm_fw)")
    print(f"  {'Val scheme':.<35} 20% holdout  ({int(len(X_va))} regions)")
    print(f"  {'Training windows':.<35} {len(X_tr):,}")
    print(f"  {'Val windows':.<35} {len(X_va):,}")
    print(f"  {'Blend val MAE':.<35} {best_val_mae:.4f}")
    print(f"  {'Blend weights':.<35} "
          f"{' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}")
    print(f"  {'LGB iters (wk1-5)':.<35} {lgb_iters}")
    print(f"  {'-'*58}")
    print(f"  Reference Kaggle scores (public LB):")
    print(f"  {'  Baseline 3 (target)':.<35} 0.8056")
    print(f"  {'  recent_local_8y (best so far)':.<35} 0.8095")
    print(f"  {'  v24 (+seasonal_mean)':.<35} 0.8106")
    print(f"  {'  v22/recent8':.<35} 0.8132")
    print(f"  {'-'*58}")
    print(f"  {'Runtime':.<35} {elapsed(t0)}")
    print("=" * 62)

main()
