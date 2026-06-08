"""
kaggle_v20_schema_a.py  --  Last-K-Windows Validation (Schema A)
=================================================================
Base: v19 (weather features only, NO score_lag)
Change: val = last K windows of EVERY region (Schema A)
        K=5 -> ~11,240 val points (vs 2,248 in v19)

Dataset names (Notebook 2):
  datafiles/train.npz, datafiles/test.npz
  samplesub/sample_submission.csv

No accelerator needed: LGB/XGB/CatBoost run on CPU.

Output: /kaggle/working/submission_v20_schema_a.csv
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
WEEKLY_CACHE  = WORK_DIR / "cache_weekly.npz"        # shared with v19
WINDOWS_CACHE = WORK_DIR / "cache_windows_v20.npz"   # schema-a specific
OUT_PATH      = WORK_DIR / "submission_v20_schema_a.csv"

def _find_npz(name: str) -> Path:
    # Notebook 2 dataset folder is called "datafiles"
    p = Path(f"/kaggle/input/datafiles/{name}")
    if p.exists(): return p
    # Fallback: search all input folders
    for slug in ["trainthis", "testset", "drought-data", "train-data", "data", "input"]:
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
    # Notebook 2 sample sub folder is called "samplesub"
    for p in ["/kaggle/input/samplesub/sample_submission.csv",
              "/kaggle/input/sample-submission/sample_submission.csv",
              "/kaggle/input/sample_submission/sample_submission.csv"]:
        if Path(p).exists(): return Path(p)
    found = _g.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(sorted(found)[0]) if found else None

SAMPLE_SUB = _find_sample_sub()

# ── Knobs ─────────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0
WINDOW_STRIDE   = 1
N_ESTIMATORS    = 1000
K_VAL_WINDOWS   = 5    # Schema A: last K windows per region as val
                        # 5 * 2248 regions = ~11,240 val points

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
def show(n, y, p): print(f"  {n:<50s}  MAE={mae(y,p):.4f}")

def _best_n(m, default):
    for a in ("best_iteration_", "best_iteration"):
        v = getattr(m, a, None)
        if v is not None: return int(v)
    try: return int(m.get_best_iteration())
    except: return default


# ── Feature list (no score_lag) ───────────────────────────────────────────────
def build_features():
    f = list(WEATHER_COLS)
    f += [f"{c}_lag{l}"      for c in LAG_COLS  for l in LAGS]
    f += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS
          for s in ("mean", "std", "max")]
    f += ["month_sin", "month_cos", "day_sin", "day_cos"]
    f += ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
          "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    f.append("regional_mean_score")
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
    panel = pd.concat([panel, pd.DataFrame(nc, index=panel.index)], axis=1)
    n = len(tr)
    return panel.iloc[:n].copy(), panel.iloc[n:].copy()

def _daily_to_weekly(df):
    wk = df["ordinal"] // WEEK_BUCKET
    return df.loc[df.groupby(wk, sort=False)["ordinal"].idxmax()].reset_index(drop=True)


# ── Weekly feature cache (shared with v19) ────────────────────────────────────
def load_weekly(t0):
    if WEEKLY_CACHE.exists():
        print(f"  [Cache] Weekly: {WEEKLY_CACHE.stat().st_size/1e6:.0f} MB")
        ck = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base = list(ck["feature_names"])
        weekly = pd.DataFrame(ck["weekly_feats"], columns=base)
        weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        weekly["region_id"] = ck["weekly_region"].astype(str)
        weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
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
    weekly  = pd.concat([_daily_to_weekly(g) for _, g in labeled.groupby("region_id", sort=False)],
                        ignore_index=True)
    del labeled
    base_cols = [c for c in weekly.columns
                 if c not in ("score","region_id","ordinal","date","year","month","day")]
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
        X_test_base     = X_test,
        test_region_ids = test_ids,
        feature_names   = np.array(base_cols, dtype=object),
    )
    print(f"  Weekly cache saved [{elapsed(t0)}]")
    return weekly, X_test, test_ids, base_cols


# ── Schema A: Last-K-Windows Validation ──────────────────────────────────────
def build_last_k_windows(weekly, features, k=K_VAL_WINDOWS):
    """
    Schema A: val = last K windows of EVERY region.
    K=5 -> ~11,240 val points (5 x 2248 regions).

    For each region with nw valid windows:
      Val:   windows [nw-K .. nw-1]  (K windows per region)
      Train: windows [0   .. nw-K-1]

    Persistence per val point: sc[i] repeated 5 weeks ahead.
    This correctly uses the score AT the val window, not the last training score.
    """
    Xtr, ytr, rtr = [], [], []
    Xva, yva, rva = [], [], []
    pva = []   # persistence score per val point
    skipped = 0

    for region, g in weekly.groupby("region_id", sort=False):
        g  = g.sort_values("ordinal")
        sc = g["score"].to_numpy(np.float32)
        Xn = g[features].to_numpy(np.float32)
        n  = len(g)
        if n < k + 6:   # need k val windows + 1 train + 5 y-lookahead
            skipped += 1
            continue
        nw = n - 5      # number of valid windows (each needs 5 future weeks for y)
        yr = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]

        # Val: last k windows
        val_start = nw - k
        for i in range(val_start, nw):
            Xva.append(Xn[i])
            yva.append(yr[i])
            rva.append(region)
            pva.append(sc[i])  # score at time i -> persistence prediction

        # Train: all windows before val_start
        if val_start < 1:
            skipped += 1
            continue
        idx = list(range(0, val_start, WINDOW_STRIDE))
        if (val_start - 1) not in idx: idx.append(val_start - 1)
        Xtr.append(Xn[idx]); ytr.append(yr[idx]); rtr.extend([region] * len(idx))

    X_tr = pd.DataFrame(np.vstack(Xtr).astype(np.float32), columns=features)
    X_tr["region_id"] = pd.Categorical(rtr)
    X_va = pd.DataFrame(np.vstack(Xva).astype(np.float32), columns=features)
    X_va["region_id"] = pd.Categorical(rva)
    persist_va = np.column_stack([np.array(pva, np.float32)] * 5)

    if skipped:
        print(f"  Regions skipped (too short for {k} val windows): {skipped}")
    return (X_tr, np.vstack(ytr).astype(np.float32),
            X_va, np.vstack(yva).astype(np.float32),
            persist_va)

def build_all_windows(weekly, features):
    Xp, yp, rp = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        g  = g.sort_values("ordinal"); sc = g["score"].to_numpy(np.float32)
        Xn = g[features].to_numpy(np.float32); n = len(g)
        if n < 6: continue
        nw = n - 5; yr = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        idx = list(range(0, nw, WINDOW_STRIDE))
        if (nw-1) not in idx: idx.append(nw-1)
        Xp.append(Xn[idx]); yp.append(yr[idx]); rp.extend([region]*len(idx))
    X = pd.DataFrame(np.vstack(Xp).astype(np.float32), columns=features)
    X["region_id"] = pd.Categorical(rp)
    return X, np.vstack(yp).astype(np.float32)

def load_or_build_windows(weekly, features, t0):
    if WINDOWS_CACHE.exists():
        ck = dict(np.load(WINDOWS_CACHE, allow_pickle=True))
        if list(ck["feature_names"]) == features and "p_va" in ck:
            print(f"  [Cache] Windows: {WINDOWS_CACHE.stat().st_size/1e6:.0f} MB")
            def _r(p):
                X = pd.DataFrame(ck[f"X_{p}"], columns=features)
                X["region_id"] = pd.Categorical(ck[f"r_{p}"].astype(str).tolist())
                return X, ck[f"y_{p}"]
            return *_r("tr"), *_r("va"), ck["p_va"], *_r("all")
        print("  Windows cache outdated -- rebuilding ...")

    print(f"  Building windows (K={K_VAL_WINDOWS}) ... [{elapsed(t0)}]")
    X_tr, y_tr, X_va, y_va, p_va = build_last_k_windows(weekly, features)
    X_all, y_all = build_all_windows(weekly, features)
    np.savez_compressed(WINDOWS_CACHE,
        X_tr=X_tr[features].to_numpy(np.float32), y_tr=y_tr,
        r_tr=np.array(X_tr["region_id"].astype(str), dtype=object),
        X_va=X_va[features].to_numpy(np.float32), y_va=y_va,
        r_va=np.array(X_va["region_id"].astype(str), dtype=object),
        p_va=p_va,
        X_all=X_all[features].to_numpy(np.float32), y_all=y_all,
        r_all=np.array(X_all["region_id"].astype(str), dtype=object),
        feature_names=np.array(features, dtype=object),
    )
    print(f"  Windows cache saved [{elapsed(t0)}]")
    return X_tr, y_tr, X_va, y_va, p_va, X_all, y_all


# ── Training ──────────────────────────────────────────────────────────────────
def train_lgb(X_tr, y_tr, X_va, y_va, n_trees=None):
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or LGB_P["n_estimators"]
        m = lgb.LGBMRegressor(**dict(LGB_P, random_state=RANDOM_STATE+wk, n_estimators=n))
        kw = dict(categorical_feature=["region_id"])
        if X_va is not None:
            kw.update(eval_set=[(X_va, y_va[:,wk].ravel())], eval_metric="mae",
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        m.fit(X_tr, y_tr[:,wk].ravel(), **kw)
        models.append(m)
    return models

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

def pred_lgb(models, X):
    feat = models[0].booster_.feature_name()
    return np.clip(np.column_stack([m.predict(X[feat]) for m in models]), 0, 5).astype(np.float32)

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

def print_importance(lgb_models):
    feat  = np.array(lgb_models[0].booster_.feature_name())
    imp   = sum(m.booster_.feature_importance("gain") for m in lgb_models) / len(lgb_models)
    mask  = feat != "region_id"; feat = feat[mask]; imp = imp[mask]
    total = imp.sum(); order = np.argsort(imp)[::-1]
    print(f"\n{'='*60}")
    print(f"  FEATURE IMPORTANCE (LGB Gain, avg weeks 1-5)")
    print(f"  {'Rank':<4}  {'Feature':<34}  {'%':>6}")
    for rank, i in enumerate(order[:20], 1):
        print(f"  {rank:<4d}  {feat[i]:<34}  {100*imp[i]/total:>5.2f}%")
    w_imp = imp[[i for i,f in enumerate(feat) if f in WEATHER_COLS]].sum()
    r_imp = imp[[i for i,f in enumerate(feat) if "roll" in f]].sum()
    l_imp = imp[[i for i,f in enumerate(feat) if "_lag" in f]].sum()
    d_imp = imp[[i for i,f in enumerate(feat)
                 if any(k in f for k in ["deficit","trend","anomaly","drought","dry_days"])]].sum()
    rm_imp = imp[[i for i,f in enumerate(feat) if f == "regional_mean_score"]].sum()
    print(f"\n  Groups:")
    print(f"    Weather (direct):  {100*w_imp/total:>5.1f}%")
    print(f"    Rolling stats:     {100*r_imp/total:>5.1f}%")
    print(f"    Lags:              {100*l_imp/total:>5.1f}%")
    print(f"    Drought indices:   {100*d_imp/total:>5.1f}%")
    print(f"    Regional mean:     {100*rm_imp/total:>5.1f}%")
    print(f"  Top-10 cumulative:  {100*imp[order[:10]].sum()/total:.1f}%")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    FEATURES = build_features()
    print("=" * 60)
    print(f"  kaggle_v20_schema_a  |  Features: {len(FEATURES)}  |  no score_lag")
    print(f"  Validation: Schema A -- last {K_VAL_WINDOWS} windows per region")
    print(f"  Expected val points: ~{K_VAL_WINDOWS * 2248:,}  (vs 2,248 in v19)")
    print("=" * 60)

    # 1. Weekly features
    print(f"\n[1/5] Load weekly features ... [{elapsed(t0)}]")
    weekly, X_test_base, test_ids, base_cols = load_weekly(t0)
    n_regions = weekly["region_id"].nunique()
    print(f"  {len(weekly):,} rows, {n_regions} regions")
    for f in FEATURES:
        if f not in weekly.columns: weekly[f] = np.float32(0)

    # 2. Windows
    print(f"\n[2/5] Build windows (Schema A, K={K_VAL_WINDOWS}) ... [{elapsed(t0)}]")
    X_tr, y_tr, X_va, y_va, p_va, X_all, y_all = load_or_build_windows(weekly, FEATURES, t0)
    n_val_regions = X_va["region_id"].nunique()
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")
    print(f"  Val regions: {n_val_regions}  |  Val points per region: {K_VAL_WINDOWS}")
    print(f"  Val % of all windows: {100*len(X_va)/(len(X_all)):.1f}%")

    # Persistence baseline (per-point: sc[i] repeated 5 weeks)
    persist_mae = mae(y_va, p_va)
    show("Persistence baseline (per-point score -> 5 weeks)", y_va, p_va)
    print(f"  Comparison: region-holdout persistence was 0.0321")
    print(f"  -> Higher persistence MAE = val covers more volatile periods")

    # 3. Training
    print(f"\n[3/5] Training ... [{elapsed(t0)}]")
    lgb_m = train_lgb(X_tr, y_tr, X_va, y_va)
    lgb_val = pred_lgb(lgb_m, X_va)
    show("LightGBM", y_va, lgb_val)
    for wk in range(5):
        feat = lgb_m[wk].booster_.feature_name()
        v = mae(y_va[:,wk], np.clip(lgb_m[wk].predict(X_va[feat]),0,5))
        print(f"    Week {wk+1}: iter={_best_n(lgb_m[wk],N_ESTIMATORS):4d}  MAE={v:.4f}")

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
    print(f"  Blend weights: {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}  MAE={best_val_mae:.4f}")
    print_importance(lgb_m)

    # 4. Final training
    print(f"\n[4/5] Final training (all windows) ... [{elapsed(t0)}]")
    n_lgb = [_best_n(m, N_ESTIMATORS) for m in lgb_m]
    n_xgb = [_best_n(m, N_ESTIMATORS) for m in xgb_m]
    f_lgb = train_lgb(X_all, y_all, None, None, n_lgb)
    f_xgb = train_xgb(X_all, y_all, None, None, FEATURES, n_xgb)
    f_cat = None
    if cat_m:
        n_cat = [_best_n(m, N_ESTIMATORS) for m in cat_m]
        f_cat = train_cat(X_all, y_all, None, None, FEATURES, n_cat)

    # 5. Submission
    print(f"\n[5/5] Submission ... [{elapsed(t0)}]")
    X_test = pd.DataFrame(X_test_base, columns=base_cols)
    X_test["region_id"] = pd.Categorical(test_ids)
    for f in FEATURES:
        if f not in X_test.columns: X_test[f] = np.float32(0)

    test_preds = best_w["lgb"]*pred_lgb(f_lgb,X_test) + best_w["xgb"]*pred_num(f_xgb,X_test,FEATURES)
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

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  RESULTS -- kaggle_v20_schema_a (Schema A, K={K_VAL_WINDOWS})")
    print(f"  {'Val points':.<35} {len(X_va):,}")
    print(f"  {'Val regions':.<35} {n_val_regions}")
    print(f"  {'Persistence MAE':.<35} {persist_mae:.4f}")
    print(f"  {'LightGBM val MAE':.<35} {mae(y_va, lgb_val):.4f}")
    print(f"  {'XGBoost val MAE':.<35} {mae(y_va, xgb_val):.4f}")
    if cat_m:
        print(f"  {'CatBoost val MAE':.<35} {mae(y_va, pred_num(cat_m, X_va, FEATURES)):.4f}")
    print(f"  {'Blend val MAE':.<35} {best_val_mae:.4f}")
    print(f"  {'Blend weights':.<35} {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}")
    print(f"  {'-'*56}")
    print(f"  {'v19 val MAE (last 1 window)':.<35} (see v19 run)")
    print(f"  {'Reference v12 Kaggle MAE':.<35} 0.8258")
    print(f"  {'Runtime':.<35} {elapsed(t0)}")
    print("=" * 60)

main()
