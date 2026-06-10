"""
kaggle_v28_vpd_weekly.py  --  RY=8 + Multi-Seed + VPD + Weekly Seasonality
============================================================================
Base:        v27_multiseed (RECENT_YEARS=8, last-window val, 133 features, 5-seed LGB)
Kaggle ref:  recent_local=0.8095  v24(8y+seasonal)=0.8106  v22_5y=0.8132

TWO ADDITIONAL CHANGES vs v27:

1. VPD (Vapor Pressure Deficit) approximation  -- 19 new features
   The physically most motivated drought index missing from the feature set.
   High VPD = atmosphere pulls water from soil/plants aggressively -> drought.
   vpd_approx = e_sat * (1 - humidity/100)
   where e_sat ≈ 6.112 * exp(17.67 * tmp / (tmp + 243.5))  [Magnus formula]
   + rolling mean/std/max for all 6 windows (7,14,30,60,90,180d) = 18 more features

2. regional_week_mean  -- replaces/extends monthly seasonal baseline
   v24 showed regional_seasonal_mean (monthly, 12 values/region) helps at 2.9% gain.
   Weekly resolution (52 values/region) should capture finer seasonal patterns.
   week_of_year = ((month-1)*31 + day) // 7  -> 0..52 in proprietary calendar
   regional_week_mean = avg score per (region_id, week_of_year) from full history

Total features: 133 + 1(vpd) + 18(vpd_rolls) + 1(week_mean) = 153 features
Weekly cache cannot be reused (new features) -> cache_weekly_v28.npz

Val: last window of ALL regions (~2248 points)
NB2 datasets (aktuell):
  trainthis/train.npz
  testset/test.npz
  sample-submission/sample_submission.csv

Output: /kaggle/working/submission_v28_vpd_weekly.csv
"""
from __future__ import annotations
import time
import warnings
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

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR      = Path("/kaggle/working")
WEEKLY_CACHE  = WORK_DIR / "cache_weekly_v28.npz"     # new features -> new cache
WINDOWS_CACHE = WORK_DIR / "cache_windows_v28.npz"
OUT_PATH      = WORK_DIR / "submission_v28_vpd_weekly.csv"

def _find_npz(name: str) -> Path:
    for slug in ["datafiles", "datatrain", "datatest", "traindataset", "testdataset",
                 "trainthis", "testset", "data"]:
        p = Path(f"/kaggle/input/{slug}/{name}")
        if p.exists(): return p
    found = sorted(_g.glob(f"/kaggle/input/**/{name}", recursive=True))
    if found:
        print(f"  Found: {found[0]}")
        return Path(found[0])
    p = Path(f"/kaggle/working/{name}")
    if p.exists(): return p
    avail = sorted(str(x) for x in Path("/kaggle/input/").iterdir()) if Path("/kaggle/input/").exists() else ["(empty)"]
    raise FileNotFoundError(f"'{name}' not found.\nAvailable:\n  " + "\n  ".join(avail))

def _find_sample_sub() -> Path | None:
    for p in ["/kaggle/input/samplesub/sample_submission.csv",
              "/kaggle/input/submissionsample/sample_submission.csv",
              "/kaggle/input/samplesubmission/sample_submission.csv",
              "/kaggle/input/sample-submission/sample_submission.csv"]:
        if Path(p).exists(): return Path(p)
    found = _g.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(sorted(found)[0]) if found else None

SAMPLE_SUB = _find_sample_sub()

# ── Knobs ─────────────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
WEEK_BUCKET      = 7
DRY_THRESHOLD    = 1.0
WINDOW_STRIDE    = 1
N_ESTIMATORS     = 1000
RECENT_YEARS     = 8
ORDINAL_PER_YEAR = 372
SEEDS            = [42, 123, 777, 2024, 31415]

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "humidity", "tmp", "wind"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]


LGB_P = dict(objective="regression", metric="mae", n_estimators=N_ESTIMATORS,
             learning_rate=0.04, num_leaves=127, min_child_samples=60,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
             n_jobs=-1, verbose=-1)
XGB_P = dict(objective="reg:squarederror", n_estimators=N_ESTIMATORS, learning_rate=0.04,
             max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
             reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", n_jobs=-1, verbosity=0)
CAT_P = dict(iterations=N_ESTIMATORS, learning_rate=0.04, depth=6,
             loss_function="MAE", eval_metric="MAE", random_seed=RANDOM_STATE,
             verbose=False, thread_count=-1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def elapsed(t0): s = time.time()-t0; return f"{s/60:.1f}m" if s >= 60 else f"{s:.0f}s"
def mae(y, p):   return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))
def show(n, y, p): print(f"  {n:<52s}  MAE={mae(y,p):.4f}")

def _best_n(m, default):
    for a in ("best_iteration_", "best_iteration"):
        v = getattr(m, a, None)
        if v is not None: return int(v)
    try: return int(m.get_best_iteration())
    except: return default


# ── Feature list (153 features) ───────────────────────────────────────────────
def build_features():
    f = list(WEATHER_COLS)
    f += [f"{c}_lag{l}"      for c in LAG_COLS  for l in LAGS]
    f += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS
          for s in ("mean", "std", "max")]
    f += ["month_sin", "month_cos", "day_sin", "day_cos"]
    f += ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
          "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    f.append("regional_mean_score")
    # *** NEW: VPD features ***
    f.append("vpd_approx")
    f += [f"vpd_roll{w}_{s}" for w in ROLL_WINS for s in ("mean", "std", "max")]
    # *** NEW: weekly seasonality ***
    f.append("regional_week_mean")
    return f


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
        if col in d: df[col] = d[col].astype(np.float32)
    if "score" in d: df["score"] = d["score"].astype(np.float32)
    return df


# ── Feature engineering (per region) ─────────────────────────────────────────
def _region_features(tr: pd.DataFrame, te: pd.DataFrame):
    te = te.copy(); te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc = {}
    nc["month_sin"] = np.sin(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["month_cos"] = np.cos(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["day_sin"]   = np.sin(2*np.pi*panel["day"]/31).astype(np.float32)
    nc["day_cos"]   = np.cos(2*np.pi*panel["day"]/31).astype(np.float32)
    for col in LAG_COLS:
        for lag in LAGS:
            nc[f"{col}_lag{lag}"] = panel[col].shift(lag).astype(np.float32)
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=max(3, w//10))
            nc[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            nc[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            nc[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)
    pp = panel["prec"].shift(1)
    nc["prec_deficit_90d"] = (pp.rolling(90,min_periods=30).mean() -
                               pp.rolling(365,min_periods=60).mean()).astype(np.float32)
    p7 = pp.rolling(7,min_periods=3).mean(); p30 = pp.rolling(30,min_periods=10).mean()
    nc["prec_trend_30d"] = ((p7-p30)/pp.rolling(30,min_periods=10).std().clip(lower=0.01)).astype(np.float32)
    hp = panel["humidity"].shift(1)
    nc["humidity_deficit_90d"] = (hp.rolling(90,min_periods=30).mean() -
                                   hp.rolling(365,min_periods=60).mean()).astype(np.float32)
    tp = panel["tmp"].shift(1)
    anom = (tp.rolling(90,min_periods=30).mean() - tp.rolling(365,min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]  = anom
    nc["heat_drought_idx"] = (nc["prec_deficit_90d"] * anom.clip(lower=0)).astype(np.float32)
    dry = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    nc["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)

    # *** NEW: VPD (Vapor Pressure Deficit) ***
    # Magnus formula: e_sat ≈ 6.112 * exp(17.67 * T / (T + 243.5))  [hPa]
    # VPD = e_sat * (1 - RH/100)  -- positive = drier atmosphere
    e_sat = 6.112 * np.exp(17.67 * panel["tmp"] / (panel["tmp"] + 243.5))
    vpd = (e_sat * (1 - panel["humidity"] / 100)).clip(lower=0)
    nc["vpd_approx"] = vpd.astype(np.float32)
    vpd_prior = vpd.shift(1)
    for w in ROLL_WINS:
        rv = vpd_prior.rolling(w, min_periods=max(3, w//10))
        nc[f"vpd_roll{w}_mean"] = rv.mean().astype(np.float32)
        nc[f"vpd_roll{w}_std"]  = rv.std().astype(np.float32)
        nc[f"vpd_roll{w}_max"]  = rv.max().astype(np.float32)

    panel = pd.concat([panel, pd.DataFrame(nc, index=panel.index)], axis=1)
    n = len(tr)
    return panel.iloc[:n].copy(), panel.iloc[n:].copy()

def _daily_to_weekly(df):
    wk = df["ordinal"] // WEEK_BUCKET
    return df.loc[df.groupby(wk, sort=False)["ordinal"].idxmax()].reset_index(drop=True)


# ── Weekly cache (includes VPD + week_mean) ───────────────────────────────────
def load_weekly(t0):
    if WEEKLY_CACHE.exists():
        print(f"  [Cache] Weekly v28: {WEEKLY_CACHE.stat().st_size/1e6:.0f} MB")
        ck = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base = list(ck["feature_names"])
        weekly = pd.DataFrame(ck["weekly_feats"], columns=base)
        weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        weekly["region_id"] = ck["weekly_region"].astype(str)
        weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
        weekly["month"]     = ck["weekly_month"].astype(np.int32)
        weekly["day"]       = ck["weekly_day"].astype(np.int32)
        x_key = "X_test_base" if "X_test_base" in ck else "X_test"
        return weekly, ck[x_key].astype(np.float32), ck["test_region_ids"].astype(str), base

    print(f"  No cache -- feature engineering (~20 min) ... [{elapsed(t0)}]")
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
        if i % 500 == 0 or i == len(regions): print(f"    {i}/{len(regions)} [{elapsed(t0)}]")
    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te

    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)

    labeled = train_feat[train_feat["score"].notna()].copy()

    # *** NEW: weekly seasonality -- 52 values per region (vs 12 for monthly) ***
    # week_of_year: 0..52 using proprietary calendar (month-1)*31 + day
    labeled["week_of_year"] = ((labeled["month"] - 1) * 31 + labeled["day"]) // 7
    week_means_ser = labeled.groupby(["region_id", "week_of_year"])["score"].mean()
    w_map = week_means_ser.to_dict()
    fallback = region_means.to_dict()

    weekly = pd.concat([_daily_to_weekly(g) for _, g in labeled.groupby("region_id", sort=False)],
                       ignore_index=True)
    del labeled

    weekly["week_of_year"] = ((weekly["month"] - 1) * 31 + weekly["day"]) // 7
    weekly["regional_week_mean"] = np.array(
        [w_map.get((r, w), fallback.get(r, 0.0))
         for r, w in zip(weekly["region_id"], weekly["week_of_year"])],
        dtype=np.float32
    )
    test_feat["week_of_year"] = ((test_feat["month"] - 1) * 31 + test_feat["day"]) // 7
    test_feat["regional_week_mean"] = np.array(
        [w_map.get((r, w), fallback.get(r, 0.0))
         for r, w in zip(test_feat["region_id"], test_feat["week_of_year"])],
        dtype=np.float32
    )

    base_cols = [c for c in weekly.columns
                 if c not in ("score","region_id","ordinal","date","year","month","day","week_of_year")]
    X_test_df = (test_feat.sort_values(["region_id","ordinal"])
                 .groupby("region_id", sort=False).tail(1)
                 [["region_id"] + base_cols].reset_index(drop=True))
    test_ids  = X_test_df["region_id"].values.astype(str)
    X_test    = X_test_df[base_cols].to_numpy(np.float32)

    np.savez_compressed(WEEKLY_CACHE,
        weekly_feats    = weekly[base_cols].to_numpy(np.float32),
        weekly_scores   = weekly["score"].to_numpy(np.float32),
        weekly_region   = weekly["region_id"].values.astype(str),
        weekly_ordinal  = weekly["ordinal"].to_numpy(np.int32),
        weekly_month    = weekly["month"].to_numpy(np.int32),
        weekly_day      = weekly["day"].to_numpy(np.int32),
        X_test_base     = X_test,
        test_region_ids = test_ids,
        feature_names   = np.array(base_cols, dtype=object),
    )
    print(f"  Weekly cache saved [{elapsed(t0)}]")
    return weekly, X_test, test_ids, base_cols


# ── Recent filter ─────────────────────────────────────────────────────────────
def filter_recent_per_region(weekly):
    parts = []
    for _, g in weekly.groupby("region_id", sort=False):
        cutoff = int(g["ordinal"].max()) - RECENT_YEARS * ORDINAL_PER_YEAR
        parts.append(g[g["ordinal"] >= cutoff])
    return pd.concat(parts, ignore_index=True)


# ── Windows: last-window val ──────────────────────────────────────────────────
def build_lastwindow_windows(weekly_recent, features):
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
        Xva.append(Xn[nw-1]); yva.append(yr[nw-1]); rva.append(region)
        idx_all = list(range(0, nw, WINDOW_STRIDE))
        if (nw-1) not in idx_all: idx_all.append(nw-1)
        Xal.append(Xn[idx_all]); yal.append(yr[idx_all]); ral.extend([region]*len(idx_all))
        if nw < 2: continue
        idx = list(range(0, nw-1, WINDOW_STRIDE))
        if (nw-2) not in idx: idx.append(nw-2)
        Xtr.append(Xn[idx]); ytr.append(yr[idx]); rtr.extend([region]*len(idx))
    X_tr  = pd.DataFrame(np.vstack(Xtr).astype(np.float32), columns=features)
    X_tr["region_id"]  = pd.Categorical(rtr)
    X_va  = pd.DataFrame(np.vstack(Xva).astype(np.float32), columns=features)
    X_va["region_id"]  = pd.Categorical(rva)
    X_all = pd.DataFrame(np.vstack(Xal).astype(np.float32), columns=features)
    X_all["region_id"] = pd.Categorical(ral)
    return (X_tr, np.vstack(ytr).astype(np.float32),
            X_va, np.vstack(yva).astype(np.float32),
            X_all, np.vstack(yal).astype(np.float32))

def load_or_build_windows(weekly_recent, features, t0):
    if WINDOWS_CACHE.exists():
        ck = dict(np.load(WINDOWS_CACHE, allow_pickle=True))
        if list(ck["feature_names"]) == features:
            print(f"  [Cache] Windows: {WINDOWS_CACHE.stat().st_size/1e6:.0f} MB")
            def _r(p):
                X = pd.DataFrame(ck[f"X_{p}"], columns=features)
                X["region_id"] = pd.Categorical(ck[f"r_{p}"].astype(str).tolist())
                return X, ck[f"y_{p}"]
            return *_r("tr"), *_r("va"), *_r("all")
        print("  Windows cache outdated -- rebuilding ...")
    print(f"  Building windows (RY={RECENT_YEARS}) ... [{elapsed(t0)}]")
    X_tr, y_tr, X_va, y_va, X_all, y_all = build_lastwindow_windows(weekly_recent, features)
    np.savez_compressed(WINDOWS_CACHE,
        X_tr=X_tr[features].to_numpy(np.float32), y_tr=y_tr,
        r_tr=np.array(X_tr["region_id"].astype(str), dtype=object),
        X_va=X_va[features].to_numpy(np.float32), y_va=y_va,
        r_va=np.array(X_va["region_id"].astype(str), dtype=object),
        X_all=X_all[features].to_numpy(np.float32), y_all=y_all,
        r_all=np.array(X_all["region_id"].astype(str), dtype=object),
        feature_names=np.array(features, dtype=object),
    )
    print(f"  Windows cache saved [{elapsed(t0)}]")
    return X_tr, y_tr, X_va, y_va, X_all, y_all


# ── Multi-seed LGB ────────────────────────────────────────────────────────────
def train_lgb_multiseed(X_tr, y_tr, X_va, y_va, n_trees_per_wk=None):
    all_seed_models = []
    for seed in SEEDS:
        week_models = []
        for wk in range(5):
            n = (n_trees_per_wk[wk] if n_trees_per_wk else None) or LGB_P["n_estimators"]
            m = lgb.LGBMRegressor(**dict(LGB_P, random_state=seed, n_estimators=n))
            kw = dict(categorical_feature=["region_id"])
            if X_va is not None:
                kw.update(eval_set=[(X_va, y_va[:,wk].ravel())], eval_metric="mae",
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            m.fit(X_tr, y_tr[:,wk].ravel(), **kw)
            week_models.append(m)
        all_seed_models.append(week_models)
    return all_seed_models

def pred_lgb_multiseed(all_seed_models, X):
    feat = all_seed_models[0][0].booster_.feature_name()
    preds = [np.column_stack([m.predict(X[feat]) for m in wms]) for wms in all_seed_models]
    return np.clip(np.mean(preds, axis=0), 0, 5).astype(np.float32)

def get_avg_iters(all_seed_models):
    n_wk = len(all_seed_models[0])
    return [int(round(np.mean([_best_n(sm[wk], N_ESTIMATORS) for sm in all_seed_models])))
            for wk in range(n_wk)]

def train_xgb(X_tr, y_tr, X_va, y_va, features, n_trees=None):
    Xn = X_tr[features].to_numpy(np.float32)
    Vn = X_va[features].to_numpy(np.float32) if X_va is not None else None
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or XGB_P["n_estimators"]
        p = dict(XGB_P, random_state=RANDOM_STATE+wk, n_estimators=n)
        kw = {}
        if Vn is not None:
            p["early_stopping_rounds"] = 50
            kw.update(eval_set=[(Vn, y_va[:,wk].ravel())], verbose=False)
        m = xgb.XGBRegressor(**p)
        m.fit(Xn, y_tr[:,wk].ravel(), **kw)
        models.append(m)
    return models

def train_cat(X_tr, y_tr, X_va, y_va, features, n_trees=None):
    if not CAT: return None
    Xn = X_tr[features].to_numpy(np.float32)
    Vn = X_va[features].to_numpy(np.float32) if X_va is not None else None
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or CAT_P["iterations"]
        p = dict(CAT_P, iterations=n, random_seed=RANDOM_STATE+wk)
        kw = {}
        if Vn is not None: kw.update(eval_set=(Vn, y_va[:,wk].ravel()), early_stopping_rounds=50)
        m = CatBoostRegressor(**p)
        m.fit(Xn, y_tr[:,wk].ravel(), **kw)
        models.append(m)
    return models

def pred_num(models, X, features):
    Xn = X[features].to_numpy(np.float32)
    return np.clip(np.column_stack([m.predict(Xn) for m in models]), 0, 5).astype(np.float32)

def blend(y_va, preds: dict):
    names = list(preds); arrays = [preds[n] for n in names]
    alphas = [round(x*0.05,2) for x in range(1,20)]
    best_mae, best_w = 999., {n: 1/len(names) for n in names}
    if len(names) == 2:
        for a in alphas:
            m = mae(y_va, a*arrays[0]+(1-a)*arrays[1])
            if m < best_mae: best_mae, best_w = m, {names[0]:a, names[1]:round(1-a,8)}
    elif len(names) == 3:
        for a in alphas:
            for b in alphas:
                c = round(1-a-b,8)
                if c < 0.05: continue
                m = mae(y_va, a*arrays[0]+b*arrays[1]+c*arrays[2])
                if m < best_mae: best_mae, best_w = m, {names[0]:a,names[1]:b,names[2]:c}
    return best_w, best_mae

def print_importance(lgb_models_seed0, features):
    feat  = np.array(lgb_models_seed0[0].booster_.feature_name())
    imp   = sum(m.booster_.feature_importance("gain") for m in lgb_models_seed0) / 5
    mask  = feat != "region_id"; feat = feat[mask]; imp = imp[mask]
    total = imp.sum(); order = np.argsort(imp)[::-1]
    print(f"\n{'='*60}")
    print(f"  FEATURE IMPORTANCE (LGB Gain, seed={SEEDS[0]}, avg wk1-5)")
    print(f"  {'Rank':<4}  {'Feature':<38}  {'%':>6}")
    for rank, i in enumerate(order[:25], 1):
        tag = ""
        if "vpd" in feat[i]: tag = " (VPD)"
        if feat[i] == "regional_week_mean": tag = " (NEW weekly)"
        print(f"  {rank:<4d}  {feat[i]:<38}  {100*imp[i]/total:>5.2f}%{tag}")
    vpd_imp = imp[[i for i,f in enumerate(feat) if "vpd" in f]].sum()
    wk_imp  = imp[[i for i,f in enumerate(feat) if f == "regional_week_mean"]].sum()
    r_imp   = imp[[i for i,f in enumerate(feat) if "roll" in f and "vpd" not in f]].sum()
    rm_imp  = imp[[i for i,f in enumerate(feat) if f == "regional_mean_score"]].sum()
    d_imp   = imp[[i for i,f in enumerate(feat)
                   if any(k in f for k in ["deficit","trend","anomaly","drought","dry_days"])]].sum()
    print(f"\n  Groups:")
    print(f"    Rolling stats (excl. VPD): {100*r_imp/total:>5.1f}%")
    print(f"    VPD features (NEW):        {100*vpd_imp/total:>5.1f}%")
    print(f"    Drought indices:           {100*d_imp/total:>5.1f}%")
    print(f"    Regional mean (annual):    {100*rm_imp/total:>5.1f}%")
    print(f"    Regional week mean (NEW):  {100*wk_imp/total:>5.1f}%")
    print(f"  Top-10 cumulative:          {100*imp[order[:10]].sum()/total:.1f}%")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    FEATURES = build_features()
    n_vpd = 1 + len(ROLL_WINS) * 3
    print("=" * 60)
    print(f"  kaggle_v28_vpd_weekly  |  Features: {len(FEATURES)}  |  RY={RECENT_YEARS}")
    print(f"  +VPD: {n_vpd} features (1 direct + {len(ROLL_WINS)*3} rolls)")
    print(f"  +regional_week_mean: 52 values/region (vs 12 for monthly)")
    print(f"  LGB: {len(SEEDS)} seeds averaged  |  Val: last-window 2248 pts")
    print(f"  Datasets: trainthis / testset / sample-submission")
    print("=" * 60)

    print(f"\n[1/5] Load weekly features ... [{elapsed(t0)}]")
    weekly, X_test_base, test_ids, base_cols = load_weekly(t0)
    n_regions   = weekly["region_id"].nunique()
    n_weeks_all = len(weekly)
    print(f"  {n_weeks_all:,} rows  |  {n_regions} regions")
    for f in FEATURES:
        if f not in weekly.columns: weekly[f] = np.float32(0)

    print(f"\n[2/5] Recent filter: last {RECENT_YEARS} years ... [{elapsed(t0)}]")
    weekly_recent = filter_recent_per_region(weekly)
    pct = 100 * len(weekly_recent) / n_weeks_all
    print(f"  After filter: {len(weekly_recent):,} rows  ({pct:.0f}% retained)")

    print(f"\n[3/5] Build windows ... [{elapsed(t0)}]")
    X_tr, y_tr, X_va, y_va, X_all, y_all = load_or_build_windows(weekly_recent, FEATURES, t0)
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")

    last_score  = weekly_recent.sort_values("ordinal").groupby("region_id")["score"].last()
    val_regions = X_va["region_id"].astype(str).tolist()
    persist     = np.column_stack([last_score.reindex(val_regions).fillna(0).to_numpy()] * 5)
    show("Persistence baseline", y_va, persist)

    print(f"\n[4/5] Training ({len(SEEDS)} LGB seeds × 5 weeks = {len(SEEDS)*5} models) ... [{elapsed(t0)}]")
    lgb_ms = train_lgb_multiseed(X_tr, y_tr, X_va, y_va)
    lgb_val = pred_lgb_multiseed(lgb_ms, X_va)
    show("LightGBM (5-seed avg)", y_va, lgb_val)
    avg_iters = get_avg_iters(lgb_ms)
    for wk in range(5):
        seed_iters = [_best_n(lgb_ms[si][wk], N_ESTIMATORS) for si in range(len(SEEDS))]
        hit = " ← HIT LIMIT" if max(seed_iters) >= N_ESTIMATORS - 5 else ""
        print(f"    Week {wk+1}: {seed_iters}  avg={avg_iters[wk]}{hit}")
    print(f"  v24(1seed,no_vpd): 784/987/547/1000/943")

    xgb_m = train_xgb(X_tr, y_tr, X_va, y_va, FEATURES)
    xgb_val = pred_num(xgb_m, X_va, FEATURES)
    show("XGBoost", y_va, xgb_val)

    preds_val = {"lgb": lgb_val, "xgb": xgb_val}
    cat_m = train_cat(X_tr, y_tr, X_va, y_va, FEATURES)
    if cat_m:
        cat_val = pred_num(cat_m, X_va, FEATURES)
        show("CatBoost", y_va, cat_val)
        preds_val["cat"] = cat_val

    best_w, best_val_mae = blend(y_va, preds_val)
    print(f"  Blend: {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}  MAE={best_val_mae:.4f}")
    print(f"  v24(8y,1seed,seasonal): blend lgb=0.90 cat=0.05 xgb=0.05  MAE=0.2314")
    print_importance(lgb_ms[0], FEATURES)

    print(f"\n[5/5] Final training (all {len(X_all):,} windows, multi-seed) ... [{elapsed(t0)}]")
    f_lgb = train_lgb_multiseed(X_all, y_all, None, None, avg_iters)
    n_xgb = [_best_n(m, N_ESTIMATORS) for m in xgb_m]
    f_xgb = train_xgb(X_all, y_all, None, None, FEATURES, n_xgb)
    f_cat = None
    if cat_m:
        n_cat = [_best_n(m, N_ESTIMATORS) for m in cat_m]
        f_cat = train_cat(X_all, y_all, None, None, FEATURES, n_cat)

    X_test = pd.DataFrame(X_test_base, columns=base_cols)
    X_test["region_id"] = pd.Categorical(test_ids)
    for f in FEATURES:
        if f not in X_test.columns: X_test[f] = np.float32(0)

    test_preds = best_w["lgb"]*pred_lgb_multiseed(f_lgb,X_test) + best_w["xgb"]*pred_num(f_xgb,X_test,FEATURES)
    if f_cat and "cat" in best_w:
        test_preds += best_w["cat"] * pred_num(f_cat, X_test, FEATURES)

    sub = pd.DataFrame({"region_id": test_ids})
    for k in range(5): sub[f"pred_week{k+1}"] = test_preds[:,k]
    if SAMPLE_SUB:
        template = pd.read_csv(SAMPLE_SUB)[["region_id"]]
        sub = template.merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col] = sub[col].fillna(0.0)
    sub.to_csv(OUT_PATH, index=False)
    print(f"  Submission: {OUT_PATH.name}  ({len(sub):,} rows)")

    print()
    print("=" * 60)
    print(f"  RESULTS -- kaggle_v28_vpd_weekly")
    print(f"  {'RECENT_YEARS':.<42} {RECENT_YEARS}  |  {len(FEATURES)} features")
    print(f"  {'New features':.<42} +VPD({n_vpd}) +week_mean(1)")
    print(f"  {'LGB seeds':.<42} {SEEDS}")
    print(f"  {'Avg iters per week':.<42} {avg_iters}")
    print(f"  {'LGB val MAE (5-seed avg)':.<42} {mae(y_va, lgb_val):.4f}  (v24: 0.2292)")
    print(f"  {'XGBoost val MAE':.<42} {mae(y_va, xgb_val):.4f}")
    if cat_m: print(f"  {'CatBoost val MAE':.<42} {mae(y_va, pred_num(cat_m, X_va, FEATURES)):.4f}")
    print(f"  {'Blend val MAE':.<42} {best_val_mae:.4f}  (v24: 0.2314)")
    print(f"  {'Blend weights':.<42} {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}")
    print(f"  {'-'*56}")
    print(f"  Known Kaggle (to beat):  recent_local=0.8095  v24=0.8106")
    print(f"  {'Runtime':.<42} {elapsed(t0)}")
    print("=" * 60)

main()
