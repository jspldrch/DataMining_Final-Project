"""
kaggle_v31_stratified.py  --  v31 + Stratified Holdout Val
===========================================================
Identisch zu v31_final mit EINER Änderung:

  Val-Split: STRATIFIZIERT nach Regional-Dürre-Quartilen
  -------------------------------------------------------
  Problem mit random 20% Holdout: Zufall kann dazu führen dass Val-Regionen
  systematisch weniger/mehr Dürre haben als Trainingsregionen → falsch
  kalibrierte Blend-Gewichte.
  Lösung: Regionen werden nach regional_mean_score in 4 Quartile eingeteilt.
  Aus jedem Quartil werden 20% als Val ausgewählt. Dadurch haben Train und
  Val nahezu identische Score-Verteilung.

  Diagnostik: Script druckt Val/Train mean score — Differenz sollte < 0.02 sein.

ALLE v31 ÄNDERUNGEN ENTHALTEN:
  - SPI features: prec_spi30/90/180, tmp_spi90 (klimatologisch normalisiert)
  - ROLL_COLS: + surf_pre, + dp_tmp (Transformer: surf_pre = #1 Feature)
  - Sample weights: 1.3/1.2/1.1 (reduziert von 2.0/1.5/1.2 — v30 zeigte 2.0 zu aggressiv)
  - rsm_fw_wk{1..5}: seasonal mean für Zielmonat k
  - Multi-Seed LGB: 3 Seeds
  - N_ESTIMATORS=1500, LR=0.03
  - Atomare Cache-Schreibung (kein korrupter Cache bei Abbruch)
  Total: 179 Features

KEIN ACCELERATOR NÖTIG.
Dataset: gleicher Pfad wie v29 (glob findet train.npz/test.npz automatisch)
Output:  /kaggle/working/submission_v31_stratified.csv
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
WEEKLY_CACHE  = WORK_DIR / "cache_weekly_v31s.npz"
WINDOWS_CACHE = WORK_DIR / "cache_windows_v31s.npz"
OUT_PATH      = WORK_DIR / "submission_v31_stratified.csv"

def _find_npz(name: str) -> Path:
    for slug in ["datafinal", "datafiles", "datatrain", "datatest",
                 "traindataset", "testdataset", "data"]:
        p = Path(f"/kaggle/input/{slug}/{name}")
        if p.exists(): return p
    found = sorted(_g.glob(f"/kaggle/input/**/{name}", recursive=True))
    if found: return Path(found[0])
    p = WORK_DIR / name
    if p.exists(): return p
    avail = sorted(str(x) for x in Path("/kaggle/input/").iterdir()) \
            if Path("/kaggle/input/").exists() else ["(none)"]
    raise FileNotFoundError(f"'{name}' not found.\n  " + "\n  ".join(avail))

def _find_sample_sub() -> Path | None:
    for p in [
        "/kaggle/input/datafinal/sample_submission.csv",
        "/kaggle/input/samplesub/sample_submission.csv",
        "/kaggle/input/samplesubmission/sample_submission.csv",
    ]:
        if Path(p).exists(): return Path(p)
    found = _g.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(sorted(found)[0]) if found else None

SAMPLE_SUB = _find_sample_sub()

# ── Knobs ──────────────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
HOLDOUT_FRAC     = 0.20
HOLDOUT_SEED     = 42
WEEK_BUCKET      = 7
DRY_THRESHOLD    = 1.0
WINDOW_STRIDE    = 1
N_ESTIMATORS     = 1500
RECENT_YEARS     = 8
ORDINAL_PER_YEAR = 372
DAYS_PER_MONTH   = 31
SEEDS            = [42, 123, 777]

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "humidity", "tmp", "wind", "surf_pre", "dp_tmp"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]

LGB_P = dict(
    objective="regression", metric="mae", n_estimators=N_ESTIMATORS,
    learning_rate=0.03, num_leaves=127, min_child_samples=60,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, verbose=-1,
)
XGB_P = dict(
    objective="reg:squarederror", n_estimators=N_ESTIMATORS, learning_rate=0.03,
    max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", n_jobs=-1, verbosity=0,
)
CAT_P = dict(
    iterations=N_ESTIMATORS, learning_rate=0.03, depth=6,
    loss_function="MAE", eval_metric="MAE",
    random_seed=RANDOM_STATE, verbose=False, thread_count=-1,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f}m" if s >= 60 else f"{s:.0f}s"
def mae(y, p):
    return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))
def show(name, y, p):
    print(f"  {name:<56s}  MAE={mae(y,p):.4f}")
def _best_n(m, default):
    for a in ("best_iteration_", "best_iteration"):
        v = getattr(m, a, None)
        if v is not None: return int(v)
    try: return int(m.get_best_iteration())
    except: return default

def _save_npz_atomic(path: Path, **arrays):
    """Atomar speichern: erst .tmp, dann umbenennen → kein korrupter Cache bei Abbruch."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)

# ── Sample weights (reduziert — v30 zeigte 2.0 zu aggressiv) ─────────────────
def compute_weights(y_col: np.ndarray) -> np.ndarray:
    w = np.ones(len(y_col), dtype=np.float32)
    w[(y_col >= 1.0) & (y_col < 2.0)] = 1.3
    w[(y_col >= 2.0) & (y_col < 3.0)] = 1.2
    w[y_col >= 3.0]                    = 1.1
    return w

# ── Feature list (179) ─────────────────────────────────────────────────────────
def build_features() -> list[str]:
    f  = list(WEATHER_COLS)
    f += [f"{c}_lag{l}" for c in LAG_COLS for l in LAGS]
    f += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS
          for s in ("mean", "std", "max")]
    f += ["month_sin", "month_cos", "day_sin", "day_cos"]
    f += ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
          "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    f.append("regional_mean_score")
    f.append("regional_seasonal_mean")
    f += [f"rsm_fw_wk{k}" for k in range(1, 6)]
    f += ["prec_spi30", "prec_spi90", "prec_spi180", "tmp_spi90"]
    return f

def _future_month(month: int, day: int, k_weeks: int) -> int:
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

# ── Feature engineering ────────────────────────────────────────────────────────
def _region_features(tr: pd.DataFrame, te: pd.DataFrame):
    te = te.copy(); te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc = {}
    nc["month_sin"] = np.sin(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["month_cos"] = np.cos(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["day_sin"]   = np.sin(2*np.pi*panel["day"]  /31).astype(np.float32)
    nc["day_cos"]   = np.cos(2*np.pi*panel["day"]  /31).astype(np.float32)
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
    pp  = panel["prec"].shift(1)
    nc["prec_deficit_90d"]    = (pp.rolling(90, min_periods=30).mean() -
                                  pp.rolling(365,min_periods=60).mean()).astype(np.float32)
    p7  = pp.rolling(7, min_periods=3).mean()
    p30 = pp.rolling(30,min_periods=10).mean()
    nc["prec_trend_30d"]      = ((p7-p30)/pp.rolling(30,min_periods=10).std().clip(lower=0.01)).astype(np.float32)
    hp  = panel["humidity"].shift(1)
    nc["humidity_deficit_90d"]= (hp.rolling(90, min_periods=30).mean() -
                                  hp.rolling(365,min_periods=60).mean()).astype(np.float32)
    tp   = panel["tmp"].shift(1)
    anom = (tp.rolling(90,min_periods=30).mean() -
            tp.rolling(365,min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]     = anom
    nc["heat_drought_idx"]    = (nc["prec_deficit_90d"]*anom.clip(lower=0)).astype(np.float32)
    dry = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    nc["dry_days_14d"]        = dry.rolling(14,min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"]        = dry.rolling(30,min_periods=7).sum().astype(np.float32)
    panel = pd.concat([panel, pd.DataFrame(nc, index=panel.index)], axis=1)
    n = len(tr)
    return panel.iloc[:n].copy(), panel.iloc[n:].copy()

def _daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    wk = df["ordinal"] // WEEK_BUCKET
    return df.loc[df.groupby(wk, sort=False)["ordinal"].idxmax()].reset_index(drop=True)

# ── SPI: per-region × per-month klimatologische Normalisierung ────────────────
def _compute_spi_stats(weekly: pd.DataFrame) -> pd.DataFrame:
    stats = weekly.groupby(["region_id", "month"]).agg(
        p30m  = ("prec_roll30_mean",  "mean"),
        p30s  = ("prec_roll30_mean",  "std"),
        p90m  = ("prec_roll90_mean",  "mean"),
        p90s  = ("prec_roll90_mean",  "std"),
        p180m = ("prec_roll180_mean", "mean"),
        p180s = ("prec_roll180_mean", "std"),
        t90m  = ("tmp_roll90_mean",   "mean"),
        t90s  = ("tmp_roll90_mean",   "std"),
    ).reset_index()
    for col in ["p30s", "p90s", "p180s", "t90s"]:
        stats[col] = stats[col].clip(lower=0.1)
    return stats

def _apply_spi(df: pd.DataFrame, spi_stats: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(spi_stats, on=["region_id", "month"], how="left")
    df["prec_spi30"]  = ((m["prec_roll30_mean"]  - m["p30m"])  / m["p30s"]).astype(np.float32)
    df["prec_spi90"]  = ((m["prec_roll90_mean"]  - m["p90m"])  / m["p90s"]).astype(np.float32)
    df["prec_spi180"] = ((m["prec_roll180_mean"] - m["p180m"]) / m["p180s"]).astype(np.float32)
    df["tmp_spi90"]   = ((m["tmp_roll90_mean"]   - m["t90m"])  / m["t90s"]).astype(np.float32)
    return df

# ── Weekly cache ───────────────────────────────────────────────────────────────
def load_weekly(t0: float):
    if WEEKLY_CACHE.exists():
        print(f"  [Cache] Weekly v31s: {WEEKLY_CACHE.stat().st_size/1e6:.0f} MB")
        ck   = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base = list(ck["feature_names"])
        weekly = pd.DataFrame(ck["weekly_feats"], columns=base)
        weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        weekly["region_id"] = ck["weekly_region"].astype(str)
        weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
        return weekly, ck["X_test_base"].astype(np.float32), ck["test_region_ids"].astype(str), base

    print(f"  No cache — full feature engineering (~30 min) ... [{elapsed(t0)}]")
    train_raw = load_npz(_find_npz("train.npz"))
    test_raw  = load_npz(_find_npz("test.npz"))
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)

    regions      = train_raw["region_id"].unique()
    region_means = train_raw.groupby("region_id")["score"].mean()
    tr_by = {r: g.reset_index(drop=True) for r,g in train_raw.groupby("region_id", sort=False)}
    te_by = {r: g.reset_index(drop=True) for r,g in test_raw.groupby("region_id",  sort=False)}
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

    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)

    labeled = train_feat[train_feat["score"].notna()].copy()
    s_ser   = labeled.groupby(["region_id","month"])["score"].mean()
    s_map   = s_ser.to_dict()
    fallback= region_means.to_dict()

    labeled["regional_seasonal_mean"] = np.array(
        [s_map.get((r,int(m)), fallback.get(r,0.0))
         for r,m in zip(labeled["region_id"], labeled["month"])],
        dtype=np.float32)

    weekly = pd.concat(
        [_daily_to_weekly(g) for _,g in labeled.groupby("region_id", sort=False)],
        ignore_index=True)
    del labeled

    weekly = weekly.sort_values(["region_id","ordinal"]).reset_index(drop=True)

    # Forward seasonal features
    fw_bufs = {f"rsm_fw_wk{k}": np.zeros(len(weekly), dtype=np.float32) for k in range(1,6)}
    for region, g in weekly.groupby("region_id", sort=True):
        idx    = g.index.tolist()
        months = g["month"].tolist()
        n      = len(months)
        for k in range(1,6):
            col = f"rsm_fw_wk{k}"
            for i in range(n):
                m_fwd = months[min(i+k, n-1)]
                fw_bufs[col][idx[i]] = s_map.get((region,int(m_fwd)), fallback.get(region,0.0))
    for k in range(1,6):
        weekly[f"rsm_fw_wk{k}"] = fw_bufs[f"rsm_fw_wk{k}"]
    del fw_bufs

    # SPI features
    print(f"  Computing SPI stats ... [{elapsed(t0)}]")
    spi_stats = _compute_spi_stats(weekly)
    weekly    = _apply_spi(weekly, spi_stats)

    base_cols = [c for c in weekly.columns
                 if c not in ("score","region_id","ordinal","date","year","month","day")]

    # Test: last-weekly-row per region
    test_parts = []
    for region, g in test_feat.groupby("region_id", sort=False):
        g        = g.sort_values("ordinal")
        last_ord = int(g["ordinal"].max())
        bucket   = last_ord // WEEK_BUCKET
        mask     = g["ordinal"] // WEEK_BUCKET == bucket
        row      = g.loc[[g.loc[mask,"ordinal"].idxmax()]]
        test_parts.append(row)
    X_test_df = pd.concat(test_parts, ignore_index=True)

    X_test_df["regional_seasonal_mean"] = np.array(
        [s_map.get((r,int(m)), fallback.get(r,0.0))
         for r,m in zip(X_test_df["region_id"], X_test_df["month"])],
        dtype=np.float32)
    for k in range(1,6):
        X_test_df[f"rsm_fw_wk{k}"] = np.array([
            s_map.get((r,_future_month(int(m),int(d),k)), fallback.get(r,0.0))
            for r,m,d in zip(X_test_df["region_id"],X_test_df["month"],X_test_df["day"])
        ], dtype=np.float32)
    X_test_df = _apply_spi(X_test_df, spi_stats)

    test_ids = X_test_df["region_id"].values.astype(str)
    X_test   = X_test_df[base_cols].to_numpy(np.float32)

    _save_npz_atomic(WEEKLY_CACHE,
        weekly_feats    = weekly[base_cols].to_numpy(np.float32),
        weekly_scores   = weekly["score"].to_numpy(np.float32),
        weekly_region   = weekly["region_id"].values.astype(str),
        weekly_ordinal  = weekly["ordinal"].to_numpy(np.int32),
        X_test_base     = X_test,
        test_region_ids = test_ids,
        feature_names   = np.array(base_cols, dtype=object),
    )
    print(f"  Weekly cache v31s saved (atomic) [{elapsed(t0)}]")
    return weekly, X_test, test_ids, base_cols

# ── Recent filter ──────────────────────────────────────────────────────────────
def filter_recent_per_region(weekly: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in weekly.groupby("region_id", sort=False):
        cutoff = int(g["ordinal"].max()) - RECENT_YEARS * ORDINAL_PER_YEAR
        parts.append(g[g["ordinal"] >= cutoff])
    return pd.concat(parts, ignore_index=True)

# ── Val: STRATIFIZIERTER Holdout nach Dürre-Quartilen ────────────────────────
def build_stratified_holdout_windows(weekly_recent: pd.DataFrame, features: list[str]):
    """
    Teilt Regionen in 4 Quartile nach regional_mean_score auf.
    Aus jedem Quartil werden 20% als Val ausgewählt.
    → Train und Val haben nahezu identische Score-Verteilung.
    → Kein Bias durch Zufall (z.B. alle Dürre-Regionen im Val).
    """
    rng         = np.random.default_rng(HOLDOUT_SEED)
    all_regions = np.array(weekly_recent["region_id"].unique())
    mean_scores = weekly_recent.groupby("region_id")["score"].mean()
    scores_arr  = mean_scores.reindex(all_regions).fillna(0).values

    # Quartile: Q0 = geringe Dürre, Q3 = starke Dürre
    quartiles = pd.qcut(scores_arr, q=4, labels=False, duplicates="drop")
    n_q = int(quartiles.max()) + 1

    val_list = []
    for q in range(n_q):
        q_regions = all_regions[quartiles == q]
        n_val_q   = max(1, int(len(q_regions) * HOLDOUT_FRAC))
        val_list.extend(rng.choice(q_regions, n_val_q, replace=False).tolist())
    val_set = set(val_list)

    # Diagnostik: Score-Verteilung Val vs Train
    train_set  = set(all_regions) - val_set
    val_mean   = weekly_recent[weekly_recent["region_id"].isin(val_set)]["score"].mean()
    train_mean = weekly_recent[weekly_recent["region_id"].isin(train_set)]["score"].mean()
    diff = abs(val_mean - train_mean)
    status = "✓ gut" if diff < 0.02 else "⚠ prüfen" if diff < 0.05 else "✗ schlechter Split"
    print(f"  Stratifizierter Split: {len(train_set)} train / {len(val_set)} val Regionen")
    print(f"  Val mean score: {val_mean:.4f}  |  Train mean score: {train_mean:.4f}  "
          f"|  Diff: {diff:.4f}  {status}")
    for q in range(n_q):
        q_regions = all_regions[quartiles == q]
        q_val = [r for r in q_regions if r in val_set]
        q_mean = mean_scores.reindex(q_regions).mean()
        print(f"  Q{q}: {len(q_regions)} Regionen (mean={q_mean:.3f}), {len(q_val)} als Val")

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

        idx_all = list(range(0, nw, WINDOW_STRIDE))
        if (nw-1) not in idx_all: idx_all.append(nw-1)
        Xal.append(Xn[idx_all]); yal.append(yr[idx_all]); ral.extend([region]*len(idx_all))

        if region in val_set:
            Xva.append(Xn[nw-1]); yva.append(yr[nw-1]); rva.append(region)
        else:
            idx = list(range(0, nw, WINDOW_STRIDE))
            if (nw-1) not in idx: idx.append(nw-1)
            Xtr.append(Xn[idx]); ytr.append(yr[idx]); rtr.extend([region]*len(idx))

    def _mk(Xs, ys, rs):
        X = pd.DataFrame(np.vstack(Xs).astype(np.float32), columns=features)
        X["region_id"] = pd.Categorical(rs)
        return X, np.vstack(ys).astype(np.float32)

    X_tr, y_tr = _mk(Xtr, ytr, rtr)
    X_va, y_va = _mk(Xva, yva, rva)
    X_all,y_all= _mk(Xal, yal, ral)
    return X_tr,y_tr, X_va,y_va, X_all,y_all

def load_or_build_windows(weekly_recent, features, t0):
    if WINDOWS_CACHE.exists():
        ck = dict(np.load(WINDOWS_CACHE, allow_pickle=True))
        if list(ck["feature_names"]) == features:
            print(f"  [Cache] Windows v31s: {WINDOWS_CACHE.stat().st_size/1e6:.0f} MB")
            def _r(p):
                X = pd.DataFrame(ck[f"X_{p}"], columns=features)
                X["region_id"] = pd.Categorical(ck[f"r_{p}"].astype(str).tolist())
                return X, ck[f"y_{p}"]
            return *_r("tr"), *_r("va"), *_r("all")
        print("  Windows cache outdated — rebuilding ...")

    print(f"  Building windows (stratified holdout, recent {RECENT_YEARS}y) [{elapsed(t0)}]")
    X_tr,y_tr,X_va,y_va,X_all,y_all = build_stratified_holdout_windows(weekly_recent, features)
    _save_npz_atomic(WINDOWS_CACHE,
        X_tr  = X_tr[features].to_numpy(np.float32),  y_tr  = y_tr,
        r_tr  = np.array(X_tr["region_id"].astype(str),  dtype=object),
        X_va  = X_va[features].to_numpy(np.float32),  y_va  = y_va,
        r_va  = np.array(X_va["region_id"].astype(str),  dtype=object),
        X_all = X_all[features].to_numpy(np.float32), y_all = y_all,
        r_all = np.array(X_all["region_id"].astype(str), dtype=object),
        feature_names = np.array(features, dtype=object),
    )
    print(f"  Windows cache v31s saved (atomic) [{elapsed(t0)}]")
    return X_tr,y_tr,X_va,y_va,X_all,y_all

# ── Multi-Seed LGB ─────────────────────────────────────────────────────────────
def train_lgb_multiseed(X_tr, y_tr, X_va, y_va, n_trees_per_wk=None, weighted=True):
    all_seed_models = []
    for seed in SEEDS:
        wk_models = []
        for wk in range(5):
            n  = (n_trees_per_wk[wk] if n_trees_per_wk else None) or LGB_P["n_estimators"]
            m  = lgb.LGBMRegressor(**dict(LGB_P, random_state=seed, n_estimators=n))
            kw = dict(categorical_feature=["region_id"])
            if X_va is not None:
                kw.update(eval_set=[(X_va, y_va[:,wk].ravel())], eval_metric="mae",
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            if weighted:
                kw["sample_weight"] = compute_weights(y_tr[:,wk].ravel())
            m.fit(X_tr, y_tr[:,wk].ravel(), **kw)
            wk_models.append(m)
        all_seed_models.append(wk_models)
    return all_seed_models

def pred_lgb_multiseed(all_seed_models, X) -> np.ndarray:
    feat  = all_seed_models[0][0].booster_.feature_name()
    preds = [np.column_stack([m.predict(X[feat]) for m in wms]) for wms in all_seed_models]
    return np.clip(np.mean(preds, axis=0), 0, 5).astype(np.float32)

def get_avg_iters(all_seed_models) -> list[int]:
    return [int(round(np.mean([_best_n(sm[wk], N_ESTIMATORS) for sm in all_seed_models])))
            for wk in range(5)]

def train_xgb(X_tr, y_tr, X_va, y_va, features, n_trees=None, weighted=True):
    Xn = X_tr[features].to_numpy(np.float32)
    Vn = X_va[features].to_numpy(np.float32) if X_va is not None else None
    models = []
    for wk in range(5):
        n  = (n_trees[wk] if n_trees else None) or XGB_P["n_estimators"]
        p  = dict(XGB_P, random_state=RANDOM_STATE+wk, n_estimators=n)
        kw = {}
        if Vn is not None:
            p["early_stopping_rounds"] = 50
            kw.update(eval_set=[(Vn, y_va[:,wk].ravel())], verbose=False)
        if weighted: kw["sample_weight"] = compute_weights(y_tr[:,wk].ravel())
        m = xgb.XGBRegressor(**p)
        m.fit(Xn, y_tr[:,wk].ravel(), **kw)
        models.append(m)
    return models

def train_cat(X_tr, y_tr, X_va, y_va, features, n_trees=None, weighted=True):
    if not CAT: return None
    Xn = X_tr[features].to_numpy(np.float32)
    Vn = X_va[features].to_numpy(np.float32) if X_va is not None else None
    models = []
    for wk in range(5):
        n  = (n_trees[wk] if n_trees else None) or CAT_P["iterations"]
        p  = dict(CAT_P, iterations=n, random_seed=RANDOM_STATE+wk)
        kw = {}
        if Vn is not None: kw.update(eval_set=(Vn, y_va[:,wk].ravel()), early_stopping_rounds=50)
        if weighted: kw["sample_weight"] = compute_weights(y_tr[:,wk].ravel())
        m = CatBoostRegressor(**p)
        m.fit(Xn, y_tr[:,wk].ravel(), **kw)
        models.append(m)
    return models

def pred_num(models, X, features) -> np.ndarray:
    Xn = X[features].to_numpy(np.float32)
    return np.clip(np.column_stack([m.predict(Xn) for m in models]), 0, 5).astype(np.float32)

def blend(y_va, preds: dict):
    names = list(preds); arrays = [preds[n] for n in names]
    alphas = [round(x*0.05,2) for x in range(1,20)]
    best_mae, best_w = 999., {n:1/len(names) for n in names}
    if len(names) == 2:
        for a in alphas:
            m = mae(y_va, a*arrays[0]+(1-a)*arrays[1])
            if m < best_mae: best_mae,best_w = m,{names[0]:a,names[1]:round(1-a,8)}
    elif len(names) == 3:
        for a in alphas:
            for b in alphas:
                c = round(1-a-b,8)
                if c < 0.05: continue
                m = mae(y_va, a*arrays[0]+b*arrays[1]+c*arrays[2])
                if m < best_mae: best_mae,best_w = m,{names[0]:a,names[1]:b,names[2]:c}
    return best_w, best_mae

def print_importance(all_seed_models, features):
    feat = np.array(all_seed_models[0][0].booster_.feature_name())
    imp  = sum(m.booster_.feature_importance("gain")
               for wms in all_seed_models for m in wms) / (len(all_seed_models) * 5)
    mask  = feat != "region_id"
    feat  = feat[mask]; imp = imp[mask]
    total = imp.sum(); order = np.argsort(imp)[::-1]
    print(f"\n{'='*66}")
    print(f"  FEATURE IMPORTANCE (LGB Gain, avg seeds/weeks, top 25)")
    for rank, i in enumerate(order[:25], 1):
        tag = " ◄ SPI" if "spi" in feat[i] else \
              " ◄ NEW" if ("surf_pre_roll" in feat[i] or "dp_tmp_roll" in feat[i]) else \
              " ◄ FW"  if feat[i].startswith("rsm_fw") else ""
        print(f"  {rank:<4d}  {feat[i]:<42}  {100*imp[i]/total:>5.2f}%{tag}")
    groups = {
        "Rolling (prec/hum/tmp/wind)": ["roll" in f and not any(x in f for x in ["surf_pre","dp_tmp"]) for f in feat],
        "Rolling surf_pre ◄":         ["surf_pre_roll" in f for f in feat],
        "Rolling dp_tmp ◄":           ["dp_tmp_roll"   in f for f in feat],
        "SPI (klimatol. norm.) ◄":    ["spi"           in f for f in feat],
        "Lags":                       ["_lag"          in f for f in feat],
        "Drought indices":            [any(k in f for k in ["deficit","trend","anomaly","drought","dry_days"]) for f in feat],
        "Weather raw":                [f in WEATHER_COLS  for f in feat],
        "Seasonal (current+forward)": [("seasonal" in f or "rsm_fw" in f) for f in feat],
        "Regional mean":              [f == "regional_mean_score" for f in feat],
    }
    print(f"\n  Groups:")
    for gname, mask_list in groups.items():
        g_imp = imp[[i for i,v in enumerate(mask_list) if v]].sum()
        print(f"    {gname:<36}  {100*g_imp/total:>5.1f}%")
    print(f"  Top-10 cumul.: {100*imp[order[:10]].sum()/total:.1f}%")
    print(f"{'='*66}\n")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0       = time.time()
    FEATURES = build_features()
    print("=" * 66)
    print(f"  kaggle_v31_stratified  |  {len(FEATURES)} features  |  CPU only")
    print(f"  Seeds: {SEEDS}  |  N_EST={N_ESTIMATORS}  |  LR={LGB_P['learning_rate']}")
    print(f"  Val: stratifiziert nach Dürre-Quartilen (4 Gruppen × 20%)")
    print(f"  SPI + surf_pre rolling + rsm_fw + sample weights (1.3/1.2/1.1)")
    print(f"  Dataset: {_find_npz('train.npz').parent}")
    print("=" * 66)

    print(f"\n[1/5] Weekly features ... [{elapsed(t0)}]")
    weekly, X_test_base, test_ids, base_cols = load_weekly(t0)
    n_regions   = weekly["region_id"].nunique()
    n_weeks_all = len(weekly)
    print(f"  {n_weeks_all:,} weekly rows  |  {n_regions} regions  |  {len(FEATURES)} features")
    for f in FEATURES:
        if f not in weekly.columns: weekly[f] = np.float32(0)

    print(f"\n[2/5] Recent filter: last {RECENT_YEARS} years ... [{elapsed(t0)}]")
    weekly_recent = filter_recent_per_region(weekly)
    n_recent = len(weekly_recent)
    print(f"  {n_weeks_all:,} → {n_recent:,}  ({100*n_recent/n_weeks_all:.0f}% retained)")

    print(f"\n[3/5] Build windows (stratified holdout) ... [{elapsed(t0)}]")
    X_tr,y_tr,X_va,y_va,X_all,y_all = load_or_build_windows(weekly_recent, FEATURES, t0)
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")
    last_score  = weekly_recent.sort_values("ordinal").groupby("region_id")["score"].last()
    val_regions = X_va["region_id"].astype(str).tolist()
    persist     = np.column_stack([last_score.reindex(val_regions).fillna(0).to_numpy()]*5)
    show("Persistence (last score × 5)", y_va, persist)

    print(f"\n[4/5] Training ({len(SEEDS)} seeds × 5 weeks = {len(SEEDS)*5} LGB models) [{elapsed(t0)}]")
    lgb_ms  = train_lgb_multiseed(X_tr, y_tr, X_va, y_va, weighted=True)
    lgb_val = pred_lgb_multiseed(lgb_ms, X_va)
    show(f"LightGBM ({len(SEEDS)}-seed, weighted)", y_va, lgb_val)
    avg_iters = get_avg_iters(lgb_ms)
    for wk in range(5):
        seed_iters = [_best_n(lgb_ms[si][wk], N_ESTIMATORS) for si in range(len(SEEDS))]
        hit = "  ← HIT LIMIT" if max(seed_iters) >= N_ESTIMATORS-5 else ""
        print(f"    Week {wk+1}: {seed_iters}  avg={avg_iters[wk]}{hit}")

    # Sanity: weighted vs unweighted
    lgb_ms_uw  = train_lgb_multiseed(X_tr, y_tr, X_va, y_va, weighted=False)
    lgb_val_uw = pred_lgb_multiseed(lgb_ms_uw, X_va)
    show("LightGBM (unweighted, sanity)", y_va, lgb_val_uw)
    use_weighted = mae(y_va, lgb_val) <= mae(y_va, lgb_val_uw) + 0.008
    if use_weighted:
        print("  ✓ Weights OK (Δ ≤ 0.008) → verwende weighted")
    else:
        print("  ✗ Weights schaden → fallback auf unweighted")
        lgb_ms, lgb_val = lgb_ms_uw, lgb_val_uw

    xgb_m   = train_xgb(X_tr, y_tr, X_va, y_va, FEATURES, weighted=use_weighted)
    xgb_val = pred_num(xgb_m, X_va, FEATURES)
    show("XGBoost", y_va, xgb_val)

    preds_val = {"lgb": lgb_val, "xgb": xgb_val}
    cat_m = train_cat(X_tr, y_tr, X_va, y_va, FEATURES, weighted=use_weighted)
    if cat_m:
        cat_val = pred_num(cat_m, X_va, FEATURES)
        show("CatBoost", y_va, cat_val)
        preds_val["cat"] = cat_val

    best_w, best_val_mae = blend(y_va, preds_val)
    print(f"  Blend: {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}  MAE={best_val_mae:.4f}")
    print_importance(lgb_ms, FEATURES)

    print(f"\n[5/5] Final training ({len(X_all):,} windows, alle Regionen) ... [{elapsed(t0)}]")
    f_lgb = train_lgb_multiseed(X_all, y_all, None, None, avg_iters, weighted=use_weighted)
    n_xgb = [_best_n(m, N_ESTIMATORS) for m in xgb_m]
    f_xgb = train_xgb(X_all, y_all, None, None, FEATURES, n_xgb, weighted=use_weighted)
    f_cat = None
    if cat_m:
        n_cat = [_best_n(m, N_ESTIMATORS) for m in cat_m]
        f_cat = train_cat(X_all, y_all, None, None, FEATURES, n_cat, weighted=use_weighted)

    X_test = pd.DataFrame(X_test_base, columns=base_cols)
    X_test["region_id"] = pd.Categorical(test_ids)
    for f in FEATURES:
        if f not in X_test.columns: X_test[f] = np.float32(0)

    test_preds = (best_w["lgb"] * pred_lgb_multiseed(f_lgb, X_test) +
                  best_w["xgb"] * pred_num(f_xgb, X_test, FEATURES))
    if f_cat and "cat" in best_w:
        test_preds += best_w["cat"] * pred_num(f_cat, X_test, FEATURES)

    sub = pd.DataFrame({"region_id": test_ids})
    for k in range(5): sub[f"pred_week{k+1}"] = test_preds[:,k]
    if SAMPLE_SUB:
        sub = pd.read_csv(SAMPLE_SUB)[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col] = sub[col].fillna(0.0)
    sub.to_csv(OUT_PATH, index=False)
    print(f"  Saved: {OUT_PATH.name}  ({len(sub):,} rows)")

    print()
    print("=" * 66)
    print(f"  RESULTS — kaggle_v31_stratified")
    print(f"  {'-'*62}")
    print(f"  {'Features':.<40} {len(FEATURES)}")
    print(f"  {'Val-Split':.<40} stratifiziert (4 Dürre-Quartile)")
    print(f"  {'Seeds / N_EST / LR':.<40} {SEEDS} / {N_ESTIMATORS} / {LGB_P['learning_rate']}")
    print(f"  {'Sample weights':.<40} {use_weighted} (1.3/1.2/1.1)")
    print(f"  {'Avg iters (wk1-5)':.<40} {avg_iters}")
    print(f"  {'Val regions':.<40} {len(X_va)}")
    print(f"  {'Blend val MAE':.<40} {best_val_mae:.4f}")
    print(f"  {'Blend':.<40} {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}")
    print(f"  {'-'*62}")
    print(f"  Kaggle Ziel: < 0.8056 (Baseline 3) | best bisher: 0.8095")
    print(f"  {'Runtime':.<40} {elapsed(t0)}")
    print("=" * 66)

main()
