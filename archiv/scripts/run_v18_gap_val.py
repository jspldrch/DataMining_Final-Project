"""
run_v18_gap_val.py  —  Gap-Simulation Validation
=================================================
Base: v12/kaggle_v1 (weather features only, no fresh score_lag)
One change: score_lag1/2/3 added with GAP=13 weeks offset

WHY GAP=13?
  In training/validation (current scheme):
    score_lag1 = score from 1 week ago  (autocorr ~0.97 → trivially easy)
  In Kaggle test:
    score_lag1 = score from ~13 weeks ago  (autocorr ~0.65 → much harder)
  → Model learns from fresh scores but is tested on stale ones → val/test MAE mismatch

THIS SCRIPT:
  score_lag1 = sc[t - 13]  (13 weeks old, matching the test scenario)
  Model trains and validates with the same staleness → val MAE should ≈ Kaggle MAE

EXPECTED OUTCOME:
  gap=0  val MAE ≈ 0.03   (unrealistically optimistic, like v15)
  gap=13 val MAE ≈ 0.82?  (realistic; if true, hypothesis confirmed)

  If gap=13 val MAE ≈ Kaggle MAE → validation scheme was the problem, not model design.
  If gap=13 val MAE << Kaggle MAE → test regions are fundamentally different.

TEST PREDICTIONS:
  X_test uses last test-day weather features (from weekly cache).
  score_lag1 for test = last known training score (~13 weeks before prediction point).
  This is naturally consistent with gap=13 training — no special X_test construction needed.

Kaggle-compatible: auto-detects input file paths via _find_npz().
Reuses shared weekly cache (cache_weekly.npz) from v1/v2/v3 if available.
"""
import time
import warnings
from pathlib import Path

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

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(exist_ok=True)


def _find_npz(name: str) -> Path:
    """Find a .npz file anywhere under /kaggle/input/, regardless of dataset slug."""
    import glob as _g
    for slug in ["trainthis", "testset", "drought-data", "train-data", "data", "input"]:
        p = Path(f"/kaggle/input/{slug}/{name}")
        if p.exists():
            return p
    found = sorted(_g.glob(f"/kaggle/input/**/{name}", recursive=True))
    if found:
        print(f"  Found: {found[0]}")
        return Path(found[0])
    p = Path(f"/kaggle/working/{name}")
    if p.exists():
        return p
    avail = sorted(str(x) for x in Path("/kaggle/input/").iterdir()) \
            if Path("/kaggle/input/").exists() else ["(empty)"]
    raise FileNotFoundError(
        f"'{name}' not found in /kaggle/input/\n"
        f"Available directories:\n  " + "\n  ".join(avail)
    )


def _find_sample_sub() -> "Path | None":
    import glob as _g
    for p in [
        "/kaggle/input/sample-submission/sample_submission.csv",
        "/kaggle/input/sample_submission/sample_submission.csv",
        "/kaggle/input/sample-submission/sample_submission.npz",
        "/kaggle/input/sample_submission/sample_submission.npz",
    ]:
        if Path(p).exists():
            return Path(p)
    for pat in ["**/sample_submission.csv", "**/sample_submission.npz"]:
        found = _g.glob(f"/kaggle/input/{pat}", recursive=True)
        if found:
            return Path(sorted(found)[0])
    return None


# Shared weekly cache (same as v1/v2/v3 — reused if available)
WEEKLY_CACHE  = WORK_DIR / "cache_weekly.npz"
# Separate windows cache (gap=13 changes the score_lag values inside windows)
WINDOWS_CACHE = WORK_DIR / "cache_windows_v18.npz"
OUT_PATH      = WORK_DIR / "submission_v18_gap.csv"
SAMPLE_SUB    = _find_sample_sub()

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAP_WEEKS       = 13    # weeks score_lag is shifted back (simulates 91-day test gap)
RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0
WINDOW_STRIDE   = 1
N_ESTIMATORS    = 1000

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
    loss_function="MAE", eval_metric="MAE", random_seed=RANDOM_STATE,
    verbose=False, thread_count=-1,
)


# ── Utilities ─────────────────────────────────────────────────────────────────
def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f}m" if s >= 60 else f"{s:.0f}s"

def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))

def show(label: str, y: np.ndarray, p: np.ndarray) -> None:
    print(f"  {label:<52s}  MAE={mae(y, p):.4f}")

def _best_n(m, default: int) -> int:
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(m, attr, None)
        if v is not None:
            return int(v)
    try:
        return int(m.get_best_iteration())
    except Exception:
        return default


# ── Feature list ──────────────────────────────────────────────────────────────
def build_features() -> list:
    """
    v12/v1 base weather features + gap-lagged score_lag1/2/3.
    score_lag here means score from GAP_WEEKS ago, not the current score.
    """
    f = list(WEATHER_COLS)
    f += [f"{c}_lag{l}" for c in LAG_COLS for l in LAGS]
    f += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS
          for s in ("mean", "std", "max")]
    f += ["month_sin", "month_cos", "day_sin", "day_cos"]
    f += ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
          "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    f.append("regional_mean_score")
    # Gap-lagged score features (the only addition vs v12/v1)
    f += ["score_lag1", "score_lag2", "score_lag3"]
    return f


# ── Gap-score computation ─────────────────────────────────────────────────────
def add_gap_score_lags(weekly: pd.DataFrame, gap: int) -> pd.DataFrame:
    """
    Add score_lag1/2/3 shifted back by `gap` additional weeks.

    gap=0  → score_lag1 = sc[t]       (current score, like v15 — unrealistically fresh)
    gap=13 → score_lag1 = sc[t-13]    (13 weeks old, matches Kaggle test scenario)

    NaN at region start (not enough history) is filled with 0.
    """
    weekly = weekly.sort_values(["region_id", "ordinal"]).copy()
    g = weekly.groupby("region_id")["score"]

    if gap == 0:
        weekly["score_lag1"] = g.transform(lambda x: x).astype(np.float32)
        weekly["score_lag2"] = g.shift(1).astype(np.float32)
        weekly["score_lag3"] = g.shift(2).astype(np.float32)
    else:
        # shift(gap) → score from gap weeks ago
        weekly["score_lag1"] = g.shift(gap).astype(np.float32)
        weekly["score_lag2"] = g.shift(gap + 1).astype(np.float32)
        weekly["score_lag3"] = g.shift(gap + 2).astype(np.float32)

    for col in ["score_lag1", "score_lag2", "score_lag3"]:
        weekly[col] = weekly[col].fillna(0.0).astype(np.float32)
    return weekly


# ── NPZ loading ───────────────────────────────────────────────────────────────
def load_npz(path: Path) -> pd.DataFrame:
    d = np.load(path, allow_pickle=True)
    names = d["region_names"]
    df = pd.DataFrame({
        "region_id": names[d["region_id"]],
        "year":  d["year"].astype(np.int32),
        "month": d["month"].astype(np.int32),
        "day":   d["day"].astype(np.int32),
    })
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]
    for col in WEATHER_COLS:
        if col in d:
            df[col] = d[col].astype(np.float32)
    if "score" in d:
        df["score"] = d["score"].astype(np.float32)
    return df


# ── Feature engineering (per region, daily → weekly) ─────────────────────────
def _region_features(tr: pd.DataFrame, te: pd.DataFrame):
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc = {}
    nc["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    nc["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    nc["day_sin"]   = np.sin(2 * np.pi * panel["day"] / 31).astype(np.float32)
    nc["day_cos"]   = np.cos(2 * np.pi * panel["day"] / 31).astype(np.float32)
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
    pp = panel["prec"].shift(1)
    nc["prec_deficit_90d"] = (
        pp.rolling(90, min_periods=30).mean() -
        pp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    p7  = pp.rolling(7, min_periods=3).mean()
    p30 = pp.rolling(30, min_periods=10).mean()
    nc["prec_trend_30d"] = (
        (p7 - p30) / pp.rolling(30, min_periods=10).std().clip(lower=0.01)
    ).astype(np.float32)
    hp = panel["humidity"].shift(1)
    nc["humidity_deficit_90d"] = (
        hp.rolling(90, min_periods=30).mean() -
        hp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    tp   = panel["tmp"].shift(1)
    anom = (tp.rolling(90, min_periods=30).mean() -
            tp.rolling(365, min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]  = anom
    nc["heat_drought_idx"] = (nc["prec_deficit_90d"] * anom.clip(lower=0)).astype(np.float32)
    dry = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    nc["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)
    panel = pd.concat([panel, pd.DataFrame(nc, index=panel.index)], axis=1)
    n = len(tr)
    return panel.iloc[:n].copy(), panel.iloc[n:].copy()


def _daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    wk = df["ordinal"] // WEEK_BUCKET
    return df.loc[df.groupby(wk, sort=False)["ordinal"].idxmax()].reset_index(drop=True)


# ── Weekly cache (shared with v1/v2/v3 if available) ─────────────────────────
def load_weekly(t0: float):
    if WEEKLY_CACHE.exists():
        print(f"  [cache] weekly: {WEEKLY_CACHE.stat().st_size / 1e6:.0f} MB")
        ck = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base = list(ck["feature_names"])
        weekly = pd.DataFrame(ck["weekly_feats"], columns=base)
        weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        weekly["region_id"] = ck["weekly_region"].astype(str)
        weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
        return weekly, ck["X_test_base"].astype(np.float32), ck["test_region_ids"].astype(str), base

    print(f"  No cache — running feature engineering (~20 min) ... [{elapsed(t0)}]")
    train_path = _find_npz("train.npz")
    test_path  = _find_npz("test.npz")
    print(f"  train: {train_path}")
    print(f"  test:  {test_path}")
    train_raw = load_npz(train_path)
    test_raw  = load_npz(test_path)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)

    regions      = train_raw["region_id"].unique()
    region_means = train_raw.groupby("region_id")["score"].mean()
    tr_by = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    te_by = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id", sort=False)}
    del train_raw, test_raw

    all_tr, all_te = [], []
    for i, region in enumerate(regions, 1):
        tf, ef = _region_features(tr_by[region], te_by.get(region, pd.DataFrame()))
        all_tr.append(tf)
        all_te.append(ef)
        if i % 500 == 0 or i == len(regions):
            print(f"    {i}/{len(regions)}  [{elapsed(t0)}]")

    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te

    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)

    labeled = train_feat[train_feat["score"].notna()].copy()
    weekly  = pd.concat(
        [_daily_to_weekly(g) for _, g in labeled.groupby("region_id", sort=False)],
        ignore_index=True,
    )
    del labeled

    base_cols = [c for c in weekly.columns
                 if c not in ("score", "region_id", "ordinal", "date", "year", "month", "day")]

    # X_test: last day of the test window per region (weather features, no score)
    X_test_df = (test_feat.sort_values(["region_id", "ordinal"])
                 .groupby("region_id", sort=False).tail(1)
                 [["region_id"] + base_cols].reset_index(drop=True))
    test_region_ids = X_test_df["region_id"].values.astype(str)
    X_test_arr = X_test_df[base_cols].to_numpy(np.float32)

    np.savez_compressed(
        WEEKLY_CACHE,
        weekly_feats    = weekly[base_cols].to_numpy(np.float32),
        weekly_scores   = weekly["score"].to_numpy(np.float32),
        weekly_region   = weekly["region_id"].values.astype(str),
        weekly_ordinal  = weekly["ordinal"].to_numpy(np.int32),
        X_test_base     = X_test_arr,
        test_region_ids = test_region_ids,
        feature_names   = np.array(base_cols, dtype=object),
    )
    print(f"  Weekly cache saved: {WEEKLY_CACHE.name}  [{elapsed(t0)}]")
    return weekly, X_test_arr, test_region_ids, base_cols


# ── Sliding windows ───────────────────────────────────────────────────────────
def _build_windows(weekly: pd.DataFrame, skip: set, features: list, stride: int = 1):
    Xp, yp, rp = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip:
            continue
        g  = g.sort_values("ordinal")
        sc = g["score"].to_numpy(np.float32)
        Xn = g[features].to_numpy(np.float32)
        n  = len(g)
        if n < 6:
            continue
        nw = n - 5
        yr = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        idx = list(range(0, nw, stride))
        if (nw - 1) not in idx:
            idx.append(nw - 1)
        Xp.append(Xn[idx])
        yp.append(yr[idx])
        rp.extend([region] * len(idx))
    X = pd.DataFrame(np.vstack(Xp).astype(np.float32), columns=features)
    X["region_id"] = pd.Categorical(rp)
    return X, np.vstack(yp).astype(np.float32)


def _build_val(weekly: pd.DataFrame, val_regions: list, features: list):
    Xp, yp, rp = [], [], []
    for region in val_regions:
        g = weekly.loc[weekly["region_id"] == region].sort_values("ordinal")
        if len(g) < 6:
            continue
        Xp.append(g.iloc[-6][features].to_numpy(np.float32))
        yp.append(g.iloc[-5:]["score"].to_numpy(np.float32))
        rp.append(region)
    X = pd.DataFrame(np.vstack(Xp), columns=features)
    X["region_id"] = pd.Categorical(rp)
    return X, np.vstack(yp)


def load_or_build_windows(weekly: pd.DataFrame, val_regions: set, features: list, t0: float):
    if WINDOWS_CACHE.exists():
        ck = dict(np.load(WINDOWS_CACHE, allow_pickle=True))
        same_feats = list(ck["feature_names"]) == features
        same_val   = set(ck["val_regions"].astype(str).tolist()) == val_regions
        same_gap   = int(ck.get("gap_weeks", np.array([0]))[0]) == GAP_WEEKS
        if same_feats and same_val and same_gap:
            print(f"  [cache] windows: {WINDOWS_CACHE.stat().st_size / 1e6:.0f} MB")
            def _r(prefix):
                X = pd.DataFrame(ck[f"X_{prefix}"], columns=features)
                X["region_id"] = pd.Categorical(ck[f"r_{prefix}"].astype(str).tolist())
                return X, ck[f"y_{prefix}"]
            return *_r("tr"), *_r("va"), *_r("all")
        print("  Windows cache outdated — rebuilding ...")

    print(f"  Building sliding windows (gap={GAP_WEEKS}) ...  [{elapsed(t0)}]")
    X_tr,  y_tr  = _build_windows(weekly, val_regions, features, WINDOW_STRIDE)
    X_va,  y_va  = _build_val(weekly, sorted(val_regions), features)
    X_all, y_all = _build_windows(weekly, set(), features, WINDOW_STRIDE)
    np.savez_compressed(
        WINDOWS_CACHE,
        X_tr  = X_tr[features].to_numpy(np.float32),  y_tr  = y_tr,
        r_tr  = np.array(X_tr["region_id"].astype(str),  dtype=object),
        X_va  = X_va[features].to_numpy(np.float32),  y_va  = y_va,
        r_va  = np.array(X_va["region_id"].astype(str),  dtype=object),
        X_all = X_all[features].to_numpy(np.float32), y_all = y_all,
        r_all = np.array(X_all["region_id"].astype(str), dtype=object),
        val_regions   = np.array(sorted(val_regions), dtype=object),
        feature_names = np.array(features, dtype=object),
        gap_weeks     = np.array([GAP_WEEKS]),
    )
    print(f"  Windows cache saved  [{elapsed(t0)}]")
    return X_tr, y_tr, X_va, y_va, X_all, y_all


# ── Model training ────────────────────────────────────────────────────────────
def train_lgb(X_tr, y_tr, X_va, y_va, n_trees=None):
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or LGB_P["n_estimators"]
        m = lgb.LGBMRegressor(**dict(LGB_P, random_state=RANDOM_STATE + wk, n_estimators=n))
        kw = dict(categorical_feature=["region_id"])
        if X_va is not None:
            kw.update(
                eval_set=[(X_va, y_va[:, wk].ravel())],
                eval_metric="mae",
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
    if not CAT:
        return None
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


def pred_lgb(models, X):
    feat = models[0].booster_.feature_name()
    return np.clip(
        np.column_stack([m.predict(X[feat]) for m in models]), 0, 5
    ).astype(np.float32)


def pred_num(models, X, features):
    Xn = X[features].to_numpy(np.float32)
    return np.clip(
        np.column_stack([m.predict(Xn) for m in models]), 0, 5
    ).astype(np.float32)


def optimize_blend(y_va: np.ndarray, preds: dict):
    names  = list(preds)
    arrays = [preds[n] for n in names]
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
                if c < 0.05:
                    continue
                m = mae(y_va, a * arrays[0] + b * arrays[1] + c * arrays[2])
                if m < best_mae:
                    best_mae, best_w = m, {names[0]: a, names[1]: b, names[2]: c}
    return best_w, best_mae


def print_importance(lgb_models: list) -> None:
    feat  = np.array(lgb_models[0].booster_.feature_name())
    imp   = sum(m.booster_.feature_importance("gain") for m in lgb_models) / len(lgb_models)
    mask  = feat != "region_id"
    feat, imp = feat[mask], imp[mask]
    total = imp.sum()
    order = np.argsort(imp)[::-1]
    print(f"\n{'─' * 62}")
    print(f"  FEATURE IMPORTANCE (LGB Gain, avg weeks 1-5)")
    print(f"  {'Rank':<4}  {'Feature':<36}  {'%':>6}")
    for rank, i in enumerate(order[:20], 1):
        print(f"  {rank:<4d}  {feat[i]:<36}  {100 * imp[i] / total:>5.2f}%")
    lag_mask     = np.array(["score_lag" in n for n in feat])
    weather_mask = ~lag_mask & ~np.array(["region" in n for n in feat])
    print(f"\n  score_lag total: {100 * imp[lag_mask].sum() / total:.1f}%   "
          f"weather total: {100 * imp[weather_mask].sum() / total:.1f}%")
    print(f"{'─' * 62}\n")


# ── Submission ────────────────────────────────────────────────────────────────
def make_submission(test_preds: np.ndarray, test_region_ids: np.ndarray, out: Path):
    sub = pd.DataFrame({"region_id": test_region_ids})
    for k in range(5):
        sub[f"pred_week{k+1}"] = test_preds[:, k]
    if SAMPLE_SUB is not None and SAMPLE_SUB.suffix == ".npz":
        d = np.load(SAMPLE_SUB, allow_pickle=True)
        template = pd.DataFrame({"region_id": d["region_names"][d["region_id"]]})
    elif SAMPLE_SUB is not None:
        template = pd.read_csv(SAMPLE_SUB)[["region_id"]]
    else:
        template = sub[["region_id"]]
    result = template.merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        result[col] = result[col].fillna(0.0)
    result.to_csv(out, index=False)
    print(f"  Submission saved: {out.name}  ({len(result):,} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0       = time.time()
    FEATURES = build_features()

    print("=" * 62)
    print("  run_v18_gap_val  —  Gap-Simulation Validation")
    print(f"  GAP_WEEKS={GAP_WEEKS}  |  features={len(FEATURES)}  |  trees={N_ESTIMATORS}")
    print(f"  Hypothesis: val MAE(gap=13) ≈ Kaggle MAE (~0.82)")
    print(f"              val MAE(gap=0)  ≈ 0.03  (unrealistic, like v15)")
    print("=" * 62)

    # 1. Load weekly features (base weather features, no score_lag yet)
    print(f"\n[1/5] Weekly features  [{elapsed(t0)}]")
    weekly, X_test_base, test_region_ids, base_cols = load_weekly(t0)
    print(f"  {len(weekly):,} rows, {weekly['region_id'].nunique()} regions")

    # 2. Add gap-lagged score features (the only addition vs v12/v1)
    print(f"\n[2/5] Adding gap-lagged score features (gap={GAP_WEEKS})  [{elapsed(t0)}]")
    weekly_gap   = add_gap_score_lags(weekly.copy(), gap=GAP_WEEKS)
    weekly_fresh = add_gap_score_lags(weekly.copy(), gap=0)
    for col in ["score_lag1", "score_lag2", "score_lag3"]:
        for w in (weekly_gap, weekly_fresh):
            if col not in w.columns:
                w[col] = np.float32(0)
    print(f"  score_lag1 mean  gap={GAP_WEEKS}: {weekly_gap['score_lag1'].mean():.3f}"
          f"   gap=0: {weekly_fresh['score_lag1'].mean():.3f}")

    # 3. Sliding windows with gap=GAP_WEEKS score_lag
    print(f"\n[3/5] Sliding windows  [{elapsed(t0)}]")
    rng         = np.random.default_rng(RANDOM_STATE)
    all_regions = sorted(weekly_gap["region_id"].unique())
    val_regions = set(rng.choice(all_regions,
                                 max(1, int(len(all_regions) * VAL_REGION_FRAC)),
                                 replace=False))
    X_tr, y_tr, X_va, y_va, X_all, y_all = load_or_build_windows(
        weekly_gap, val_regions, FEATURES, t0)
    print(f"  train={len(X_tr):,}  val={len(X_va):,}  all={len(X_all):,}")

    # Baselines
    last_score = weekly_gap.sort_values("ordinal").groupby("region_id")["score"].last()
    show("Persistence baseline (last score repeated)",
         y_va, np.column_stack([last_score.reindex(sorted(val_regions)).fillna(0).to_numpy()] * 5))
    show(f"score_lag1 (gap={GAP_WEEKS}) repeated for all 5 weeks",
         y_va, np.column_stack([X_va["score_lag1"].to_numpy()] * 5))

    # 4. Training with gap=GAP_WEEKS score_lag
    print(f"\n[4/5] Training  [{elapsed(t0)}]")
    lgb_m   = train_lgb(X_tr, y_tr, X_va, y_va)
    lgb_val = pred_lgb(lgb_m, X_va)
    show(f"LightGBM  (val gap={GAP_WEEKS})", y_va, lgb_val)
    for wk in range(5):
        feat = lgb_m[wk].booster_.feature_name()
        v    = mae(y_va[:, wk], np.clip(lgb_m[wk].predict(X_va[feat]), 0, 5))
        print(f"    week {wk+1}: best_iter={_best_n(lgb_m[wk], N_ESTIMATORS):4d}  MAE={v:.4f}")

    xgb_m   = train_xgb(X_tr, y_tr, X_va, y_va, FEATURES)
    xgb_val = pred_num(xgb_m, X_va, FEATURES)
    show(f"XGBoost   (val gap={GAP_WEEKS})", y_va, xgb_val)

    preds_val = {"lgb": lgb_val, "xgb": xgb_val}
    cat_m = train_cat(X_tr, y_tr, X_va, y_va, FEATURES)
    if cat_m is not None:
        cat_val = pred_num(cat_m, X_va, FEATURES)
        show(f"CatBoost  (val gap={GAP_WEEKS})", y_va, cat_val)
        preds_val["cat"] = cat_val

    best_w, best_val_mae = optimize_blend(y_va, preds_val)
    w_str = "  ".join(f"{k}={v:.2f}" for k, v in best_w.items())
    print(f"\n  Blend: {w_str}  →  MAE={best_val_mae:.4f}")
    print_importance(lgb_m)

    # ── Comparison: same model, same val regions, but gap=0 score_lag ─────────
    # Shows what the old validation scheme would have reported for this model.
    Xp0, rp0 = [], []
    for region in sorted(val_regions):
        g = weekly_fresh.loc[weekly_fresh["region_id"] == region].sort_values("ordinal")
        if len(g) < 6:
            continue
        Xp0.append(g.iloc[-6][FEATURES].to_numpy(np.float32))
        rp0.append(region)
    X_va_fresh = pd.DataFrame(np.vstack(Xp0), columns=FEATURES)
    X_va_fresh["region_id"] = pd.Categorical(rp0)

    lgb_val0    = pred_lgb(lgb_m, X_va_fresh)
    xgb_val0    = pred_num(xgb_m, X_va_fresh, FEATURES)
    blend_val0  = best_w["lgb"] * lgb_val0 + best_w["xgb"] * xgb_val0
    if cat_m and "cat" in best_w:
        blend_val0 += best_w["cat"] * pred_num(cat_m, X_va_fresh, FEATURES)
    val_mae_fresh = mae(y_va, blend_val0)  # same targets, different X (fresh lags)

    print(f"{'═' * 62}")
    print(f"  VALIDATION COMPARISON (same {len(rp0)} val regions)")
    print(f"{'═' * 62}")
    print(f"  Val MAE  gap={GAP_WEEKS}  (realistic — simulates Kaggle test):  {best_val_mae:.4f}")
    print(f"  Val MAE  gap=0   (unrealistic — like v15, always optimistic): {val_mae_fresh:.4f}")
    print(f"\n  Reference: Kaggle MAE v12 (no score_lag)    = 0.8258")
    print(f"  Reference: Kaggle MAE v15 (fresh score_lag)  = 1.0470")
    print(f"\n  INTERPRETATION:")
    if best_val_mae > 0.70:
        print(f"  [OK]  gap={GAP_WEEKS} val MAE ({best_val_mae:.4f}) is close to the expected Kaggle range.")
        print(f"        Hypothesis confirmed: the validation scheme was the problem.")
        print(f"        This gap-trained model should generalize better than v15/v17.")
    elif best_val_mae > 0.35:
        print(f"  [~]   gap={GAP_WEEKS} val MAE ({best_val_mae:.4f}) is more realistic but below Kaggle.")
        print(f"        Partial confirmation. Test regions may be harder than val regions.")
    else:
        print(f"  [!]   gap={GAP_WEEKS} val MAE ({best_val_mae:.4f}) is still very low.")
        print(f"        The gap alone does not explain the Kaggle gap. Another factor is at play.")
    print(f"{'═' * 62}\n")

    # 5. Final training on all regions + submission
    print(f"[5/5] Final training (all regions)  [{elapsed(t0)}]")
    n_lgb = [_best_n(m, N_ESTIMATORS) for m in lgb_m]
    n_xgb = [_best_n(m, N_ESTIMATORS) for m in xgb_m]
    f_lgb = train_lgb(X_all, y_all, None, None, n_lgb)
    f_xgb = train_xgb(X_all, y_all, None, None, FEATURES, n_xgb)
    f_cat = None
    if cat_m:
        n_cat = [_best_n(m, N_ESTIMATORS) for m in cat_m]
        f_cat = train_cat(X_all, y_all, None, None, FEATURES, n_cat)
    print(f"  Done  [{elapsed(t0)}]")

    # Build test features:
    # - Weather: from weekly cache (last test day per region)
    # - score_lag: last training scores (~13 weeks before prediction point)
    #   → naturally consistent with gap=13 training, no special construction needed
    X_test = pd.DataFrame(X_test_base, columns=base_cols)
    X_test["region_id"] = pd.Categorical(test_region_ids)

    _recent = {
        region: g["score"].tolist()
        for region, g in weekly.sort_values("ordinal").groupby("region_id")
    }

    def _get_lag(region: str, k: int) -> float:
        sc = _recent.get(region, [0.0])
        return float(sc[-k]) if len(sc) >= k else float(sc[0])

    X_test["score_lag1"] = np.array([_get_lag(r, 1) for r in test_region_ids], dtype=np.float32)
    X_test["score_lag2"] = np.array([_get_lag(r, 2) for r in test_region_ids], dtype=np.float32)
    X_test["score_lag3"] = np.array([_get_lag(r, 3) for r in test_region_ids], dtype=np.float32)

    for f in FEATURES:
        if f not in X_test.columns:
            X_test[f] = np.float32(0)

    test_preds = best_w["lgb"] * pred_lgb(f_lgb, X_test) + best_w["xgb"] * pred_num(f_xgb, X_test, FEATURES)
    if f_cat and "cat" in best_w:
        test_preds += best_w["cat"] * pred_num(f_cat, X_test, FEATURES)

    make_submission(test_preds, test_region_ids, OUT_PATH)

    print(f"\n  Val MAE (gap={GAP_WEEKS}, realistic): {best_val_mae:.4f}")
    print(f"  Val MAE (gap=0, optimistic):          {val_mae_fresh:.4f}")
    print(f"  Runtime: {elapsed(t0)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
