"""
run_v15.py  –  Drought Severity Prediction v15

Neu vs v12:
  + score_lag1/2/3       – fehlt in v12, Autokorrelation 0.966, Scout Rank 1/2/5
  + seasonal deviations  – prec/humidity/tmp vs Region-Monats-Mittelwert, Scout Rank 6/9/18
  + Bereinigte Features  – kein *_roll_max, kein wind_roll_*, kein dry_days (Scout: Noise)

Modelle:
  LightGBM / XGBoost / CatBoost — je 5 Modelle (eines pro Vorhersage-Woche)
  Warum 5 statt 1: Woche 1 (Autokorr. 0.97) braucht anderen Feature-Mix als
  Woche 5 (Autokorr. 0.58). Jedes Modell kann sich spezialisieren.
  CatBoost = Gradient Boosting von Yandex, findet andere Muster → Ensemble-Diversität.

Jahre: RECENT_YEARS filtert PER REGION (letzte N Jahre des jeweiligen 13-Jahres-Fensters).
       Default None = alle 13 Jahre nutzen.

Checkpoints:
  data/precomputed/_checkpoint_weekly.npz      – Feature Engineering, ~20 Min, wiederverwendet
  data/precomputed/_checkpoint_v15_windows.npz – Sliding Windows, ~3 Min, neu wenn Features ändern

Usage:
    python scripts/run_v15.py
Output: outputs/submission_v15.csv
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
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
CACHE_DIR     = DATA_DIR / "precomputed"
OUT_DIR       = ROOT / "outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV     = DATA_DIR / "train.csv"
TEST_CSV      = DATA_DIR / "test.csv"
SAMPLE_SUB    = DATA_DIR / "sample_submission.csv"
OUT_PATH      = OUT_DIR / "submission_v15.csv"
WEEKLY_CACHE  = CACHE_DIR / "_checkpoint_weekly.npz"
WINDOWS_CACHE = CACHE_DIR / "_checkpoint_v15b_windows.npz"

# ─── Knobs ────────────────────────────────────────────────────────────────────
QUICK_MODE      = False   # True ~15 Min (kein Submission), False ~45 Min (voll)
RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0

# Per-Region-Recency: letzte N Jahre des jeweiligen 13-Jahres-Fensters jeder Region
# None = alle 13 Jahre; 5 = letzte 5 Jahre; 8 = letzte 8 Jahre
RECENT_YEARS    = None

WINDOW_STRIDE = 4 if QUICK_MODE else 1
N_ESTIMATORS  = 500 if QUICK_MODE else 1000

# ─── Feature-Konfiguration ────────────────────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS     = [1, 3, 7, 14, 21]

# Rolling: NUR prec/humidity/tmp, NUR mean+std
# wind_roll_* weg (Scout Bottom), *_roll_max weg (Scout Bottom)
ROLL_COLS  = ["prec", "humidity", "tmp"]
ROLL_WINS  = [7, 14, 30, 60, 90, 180]
ROLL_STATS = ["mean", "std"]

NUM_FEATURES: list[str] = []


def build_feature_list() -> list[str]:
    feats = list(WEATHER_COLS)
    feats += [f"{c}_lag{l}" for c in LAG_COLS for l in LAGS]
    feats += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS for s in ROLL_STATS]
    # week_sin/cos Gain ~13 → raus; month/day sin/cos behalten
    feats += ["month_sin", "month_cos", "day_sin", "day_cos"]
    feats += [
        "prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
        "tmp_anomaly_90d", "heat_drought_idx",
        # dry_days_14d/30d: Scout Bottom 15 → raus
    ]
    feats.append("regional_mean_score")
    # Neu v15
    feats += ["score_lag1", "score_lag2", "score_lag3", "score_lag4", "score_lag5"]
    feats += ["prec_seasonal_dev", "humidity_seasonal_dev", "tmp_seasonal_dev"]
    return feats


# ─── Modell-Parameter ─────────────────────────────────────────────────────────
LGB_PARAMS = dict(
    objective="regression", metric="mae", n_estimators=N_ESTIMATORS,
    learning_rate=0.04, num_leaves=127, min_child_samples=60,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, verbose=-1,
)
XGB_PARAMS = dict(
    objective="reg:squarederror", n_estimators=N_ESTIMATORS, learning_rate=0.04,
    max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", n_jobs=-1, verbosity=0,
)
CAT_PARAMS = dict(
    iterations=N_ESTIMATORS, learning_rate=0.04, depth=6,
    loss_function="MAE", eval_metric="MAE",
    random_seed=RANDOM_STATE, verbose=False, thread_count=-1,
)


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────
def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} Min." if s >= 60 else f"{s:.0f}s"

def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))

def show_mae(name: str, y: np.ndarray, p: np.ndarray) -> None:
    print(f"  {name:<50s}  MAE = {mae(y, p):.4f}")

def _parse_dates(df: pd.DataFrame) -> None:
    p = df["date"].str.split("-", expand=True)
    df["year"] = p[0].astype(np.int32); df["month"] = p[1].astype(np.int32)
    df["day"]  = p[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]

def _best_n(m, default: int) -> int:
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(m, attr, None)
        if v is not None: return int(v)
    try: return int(m.get_best_iteration())
    except: return default


# ─── Feature Engineering (läuft nur ohne Weekly-Cache) ────────────────────────
def _region_features(tr: pd.DataFrame, te: pd.DataFrame):
    te = te.copy(); te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc: dict = {}
    nc["month_sin"] = np.sin(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["month_cos"] = np.cos(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["day_sin"]   = np.sin(2*np.pi*panel["day"]/31).astype(np.float32)
    nc["day_cos"]   = np.cos(2*np.pi*panel["day"]/31).astype(np.float32)
    woy = (panel["ordinal"] // 7) % 52
    nc["week_sin"] = np.sin(2*np.pi*woy/52).astype(np.float32)  # im Cache, in v15 nicht genutzt
    nc["week_cos"] = np.cos(2*np.pi*woy/52).astype(np.float32)
    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS: nc[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)
    for col in ["prec", "humidity", "tmp", "wind"]:  # wind im Cache für Rückwärtskompatibilität
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=max(3, w//10))
            nc[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            nc[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            nc[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)
    pp = panel["prec"].shift(1)
    nc["prec_deficit_90d"] = (
        pp.rolling(90, min_periods=30).mean() - pp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    p7 = pp.rolling(7, min_periods=3).mean(); p30 = pp.rolling(30, min_periods=10).mean()
    nc["prec_trend_30d"] = ((p7 - p30) / pp.rolling(30, min_periods=10).std().clip(lower=0.01)).astype(np.float32)
    hp = panel["humidity"].shift(1)
    nc["humidity_deficit_90d"] = (
        hp.rolling(90, min_periods=30).mean() - hp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    tp = panel["tmp"].shift(1)
    anom = (tp.rolling(90, min_periods=30).mean() - tp.rolling(365, min_periods=60).mean()).astype(np.float32)
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


# ─── Neue V15-Features (brauchen Score-Spalte, daher nach Cache-Load) ─────────
def add_v15_features(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.sort_values(["region_id", "ordinal"]).copy()

    # Score-Lags: lag1 = aktueller Score, lag2+ = Vorwochen
    g = weekly.groupby("region_id")["score"]
    weekly["score_lag1"] = g.transform(lambda x: x).astype(np.float32)
    for k, col in enumerate(["score_lag2", "score_lag3", "score_lag4", "score_lag5"], 1):
        weekly[col] = g.shift(k).astype(np.float32)
    # NaN am Anfang jeder Region mit nächst-bekanntem Wert füllen
    for prev, cur in zip(
        ["score_lag1", "score_lag2", "score_lag3", "score_lag4"],
        ["score_lag2", "score_lag3", "score_lag4", "score_lag5"],
    ):
        weekly[cur].fillna(weekly[prev], inplace=True)

    # Saisonale Abweichung: aktueller Wert minus historischer Region-Monats-Mittelwert
    # Fix: Monat 12 → 12*31=372 → ordinal%372=day → //31=0 (falsch). Korrektur: 0→12
    m = (weekly["ordinal"] % 372) // 31
    weekly["_month"] = m.where(m > 0, 12).astype(np.int8)
    for col in ["prec", "humidity", "tmp"]:
        if col in weekly.columns:
            norm = weekly.groupby(["region_id", "_month"])[col].transform("mean")
            weekly[f"{col}_seasonal_dev"] = (weekly[col] - norm).astype(np.float32)
    weekly.drop(columns=["_month"], inplace=True)
    return weekly


# ─── Checkpoint 1: Wöchentliche Daten ─────────────────────────────────────────
def load_weekly() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Lädt aus Cache oder berechnet neu + speichert Cache."""
    if WEEKLY_CACHE.exists():
        print(f"   Weekly-Cache: {WEEKLY_CACHE.name}  ({WEEKLY_CACHE.stat().st_size/1e6:.0f} MB)")
        ck = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base = list(ck["feature_names"])
        weekly = pd.DataFrame(ck["weekly_feats"], columns=base)
        weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        weekly["region_id"] = ck["weekly_region"].astype(str)
        weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
        return weekly, ck["X_test"].astype(np.float32), ck["test_region_ids"].astype(str)

    # Kein Cache → volle Berechnung
    print("   Kein Cache — Feature Engineering (~20 Min) ...")
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw  = pd.read_csv(TEST_CSV,  dtype=dtypes)
    _parse_dates(train_raw); _parse_dates(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    region_means = train_raw.groupby("region_id")["score"].mean()
    tr_by = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    te_by = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw
    all_tr, all_te = [], []
    for i, region in enumerate(regions, 1):
        tf, ef = _region_features(tr_by[region], te_by.get(region, pd.DataFrame()))
        all_tr.append(tf); all_te.append(ef)
        if i % 500 == 0 or i == len(regions):
            print(f"   Region {i}/{len(regions)}")
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
    X_test_df = (
        test_feat.sort_values(["region_id", "ordinal"])
        .groupby("region_id", sort=False).tail(1)
        [["region_id"] + base_cols].reset_index(drop=True)
    )
    test_region_ids = X_test_df["region_id"].values.astype(str)
    X_test_arr = X_test_df[base_cols].to_numpy(np.float32)
    np.savez_compressed(WEEKLY_CACHE,
        weekly_feats   = weekly[base_cols].to_numpy(np.float32),
        weekly_scores  = weekly["score"].to_numpy(np.float32),
        weekly_region  = weekly["region_id"].values.astype(str),
        weekly_ordinal = weekly["ordinal"].to_numpy(np.int32),
        X_test         = X_test_arr,
        test_region_ids= test_region_ids,
        feature_names  = np.array(base_cols, dtype=object),
    )
    print(f"   Weekly-Cache gespeichert: {WEEKLY_CACHE.name}")
    return weekly, X_test_arr, test_region_ids


# ─── Checkpoint 2: Sliding Windows ───────────────────────────────────────────
def _per_region_cutoff(g: pd.DataFrame) -> pd.DataFrame:
    """Filtert auf letzte RECENT_YEARS Jahre dieser Region (per-Region, nicht global)."""
    if RECENT_YEARS is None: return g
    cutoff = int(g["ordinal"].max()) - int(RECENT_YEARS * 372)  # 372 Ordinal-Einheiten = 1 Jahr
    return g[g["ordinal"] >= cutoff]

def _build_windows(weekly, skip_regions, features, stride=1):
    Xp, yp, rp = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions: continue
        g = _per_region_cutoff(g.sort_values("ordinal"))
        sc = g["score"].to_numpy(np.float32)
        Xn = g[features].to_numpy(np.float32)
        n = len(g)
        if n < 6: continue
        nw = n - 5
        yr = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        idx = list(range(0, nw, stride))
        if (nw - 1) not in idx: idx.append(nw - 1)
        Xp.append(Xn[idx]); yp.append(yr[idx]); rp.extend([region] * len(idx))
    X = pd.DataFrame(np.vstack(Xp).astype(np.float32), columns=features)
    X["region_id"] = pd.Categorical(rp)
    return X, np.vstack(yp).astype(np.float32)

def _build_val(weekly, val_regions, features):
    Xp, yp, rp = [], [], []
    for region in val_regions:
        g = weekly.loc[weekly["region_id"] == region].sort_values("ordinal")
        if len(g) < 6: continue
        Xp.append(g.iloc[-6][features].to_numpy(np.float32))
        yp.append(g.iloc[-5:]["score"].to_numpy(np.float32))
        rp.append(region)
    X = pd.DataFrame(np.vstack(Xp), columns=features)
    X["region_id"] = pd.Categorical(rp)
    return X, np.vstack(yp)

def load_or_build_windows(weekly, val_regions, features, t0):
    """Lädt Windows-Cache oder berechnet neu. Invalidiert wenn Features/Val-Split ändern."""
    if WINDOWS_CACHE.exists():
        ck = dict(np.load(WINDOWS_CACHE, allow_pickle=True))
        same_feats = list(ck["feature_names"]) == features
        same_val   = set(ck["val_regions"].astype(str).tolist()) == val_regions
        if same_feats and same_val:
            print(f"   Windows-Cache: {WINDOWS_CACHE.name}  ({WINDOWS_CACHE.stat().st_size/1e6:.0f} MB)")
            def _rebuild(prefix):
                X = pd.DataFrame(ck[f"X_{prefix}"], columns=features)
                X["region_id"] = pd.Categorical(ck[f"r_{prefix}"].astype(str).tolist())
                return X, ck[f"y_{prefix}"]
            X_tr, y_tr = _rebuild("tr")
            X_va, y_va = _rebuild("va")
            X_all, y_all = _rebuild("all")
            return X_tr, y_tr, X_va, y_va, X_all, y_all
        reason = "Feature-Liste" if not same_feats else "Val-Regionen"
        print(f"   {reason} geändert — Windows neu berechnen ...")

    print(f"   Berechne Sliding Windows ...  [{elapsed(t0)}]")
    X_tr,  y_tr  = _build_windows(weekly, val_regions, features, WINDOW_STRIDE)
    X_va,  y_va  = _build_val(weekly, sorted(val_regions), features)
    X_all, y_all = _build_windows(weekly, set(), features, WINDOW_STRIDE)
    np.savez_compressed(WINDOWS_CACHE,
        X_tr  = X_tr[features].to_numpy(np.float32),  y_tr  = y_tr,
        r_tr  = np.array(X_tr["region_id"].astype(str), dtype=object),
        X_va  = X_va[features].to_numpy(np.float32),  y_va  = y_va,
        r_va  = np.array(X_va["region_id"].astype(str), dtype=object),
        X_all = X_all[features].to_numpy(np.float32), y_all = y_all,
        r_all = np.array(X_all["region_id"].astype(str), dtype=object),
        val_regions  = np.array(sorted(val_regions), dtype=object),
        feature_names= np.array(features, dtype=object),
    )
    print(f"   Windows-Cache gespeichert: {WINDOWS_CACHE.name}  [{elapsed(t0)}]")
    return X_tr, y_tr, X_va, y_va, X_all, y_all


# ─── Modell-Training ──────────────────────────────────────────────────────────
def _train_lgb(X_tr, y_tr, X_va, y_va, n_trees=None):
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or LGB_PARAMS["n_estimators"]
        p = dict(LGB_PARAMS, random_state=RANDOM_STATE + wk, n_estimators=n)
        m = lgb.LGBMRegressor(**p)
        kw: dict = dict(categorical_feature=["region_id"])
        if X_va is not None:
            kw["eval_set"] = [(X_va, y_va[:, wk].ravel())]
            kw["eval_metric"] = "mae"
            kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        m.fit(X_tr, y_tr[:, wk].ravel(), **kw)
        models.append(m)
    return models

def _train_xgb(X_tr, y_tr, X_va, y_va, features, n_trees=None):
    Xn = X_tr[features].to_numpy(np.float32)
    Vn = X_va[features].to_numpy(np.float32) if X_va is not None else None
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or XGB_PARAMS["n_estimators"]
        p = dict(XGB_PARAMS, random_state=RANDOM_STATE + wk, n_estimators=n)
        kw: dict = {}
        if Vn is not None:
            p["early_stopping_rounds"] = 50
            kw["eval_set"] = [(Vn, y_va[:, wk].ravel())]
            kw["verbose"] = False
        m = xgb.XGBRegressor(**p)
        m.fit(Xn, y_tr[:, wk].ravel(), **kw)
        models.append(m)
    return models

def _train_cat(X_tr, y_tr, X_va, y_va, features, n_trees=None):
    if not CATBOOST_AVAILABLE: return None
    Xn = X_tr[features].to_numpy(np.float32)
    Vn = X_va[features].to_numpy(np.float32) if X_va is not None else None
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or CAT_PARAMS["iterations"]
        p = dict(CAT_PARAMS, iterations=n, random_seed=RANDOM_STATE + wk)
        kw: dict = {}
        if Vn is not None:
            kw["eval_set"] = (Vn, y_va[:, wk].ravel())
            kw["early_stopping_rounds"] = 50
        m = CatBoostRegressor(**p)
        m.fit(Xn, y_tr[:, wk].ravel(), **kw)
        models.append(m)
    return models

def _pred_lgb(models, X):
    feat = models[0].booster_.feature_name()  # exakt die Spalten aus dem Training
    return np.clip(np.column_stack([m.predict(X[feat]) for m in models]), 0, 5).astype(np.float32)

def _pred_num(models, X, features):
    Xn = X[features].to_numpy(np.float32)
    return np.clip(np.column_stack([m.predict(Xn) for m in models]), 0, 5).astype(np.float32)

def optimize_blend(y_va, preds: dict) -> tuple[dict, float]:
    names = list(preds.keys()); arrays = [preds[n] for n in names]
    alphas = [round(x * 0.05, 2) for x in range(1, 20)]
    best_mae, best_w = 999.0, {n: 1/len(names) for n in names}
    if len(names) == 2:
        for a in alphas:
            b = round(1 - a, 8)
            m = mae(y_va, a*arrays[0] + b*arrays[1])
            if m < best_mae: best_mae, best_w = m, {names[0]: a, names[1]: b}
    elif len(names) == 3:
        for a in alphas:
            for b in alphas:
                c = round(1 - a - b, 8)
                if c < 0.05: continue
                m = mae(y_va, a*arrays[0] + b*arrays[1] + c*arrays[2])
                if m < best_mae: best_mae, best_w = m, {names[0]: a, names[1]: b, names[2]: c}
    return best_w, best_mae


# ─── Feature Importance (LightGBM, Mittelwert über alle 5 Wochen-Modelle) ─────
def print_feature_importance(lgb_models: list, top_n: int = 40) -> None:
    feat_names = np.array(lgb_models[0].booster_.feature_name())
    importance  = np.zeros(len(feat_names))
    for m in lgb_models:
        importance += m.booster_.feature_importance(importance_type="gain")
    importance /= len(lgb_models)
    mask = feat_names != "region_id"
    feat_names = feat_names[mask]; importance = importance[mask]
    total = importance.sum(); order = np.argsort(importance)[::-1]
    print(f"\n{'─'*64}")
    print(f"  FEATURE IMPORTANCE  (LightGBM Gain, Ø Woche 1-5)")
    print(f"{'─'*64}")
    print(f"  {'Rank':<5}  {'Feature':<36}  {'Gain':>10}  {'%':>6}")
    for rank, i in enumerate(order[:top_n], 1):
        print(f"  {rank:<5d}  {feat_names[i]:<36}  {importance[i]:>10.0f}  {100*importance[i]/total:>5.2f}%")
    print(f"\n  BOTTOM 10 (Noise-Kandidaten):")
    for i in order[-10:]:
        print(f"  {'':5}  {feat_names[i]:<36}  {importance[i]:>10.0f}  {100*importance[i]/total:>5.2f}%")
    print(f"\n  Top-10 kumulativ: {100*importance[order[:10]].sum()/total:.1f}% des Gains")
    print(f"  Features mit < 0.01% Gain: {(importance/total < 0.0001).sum()} Stück")
    print(f"{'─'*64}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 64)
    print("  Drought Severity Prediction  —  run_v15.py")
    print(f"  Mode: {'QUICK' if QUICK_MODE else 'FULL'}  |  stride={WINDOW_STRIDE}  trees={N_ESTIMATORS}")
    print(f"  CatBoost: {'ON' if CATBOOST_AVAILABLE else 'OFF'}  |  RECENT_YEARS={RECENT_YEARS}")
    print(f"  Features: {len(NUM_FEATURES)}  (v12 hatte ~138)")
    print("=" * 64)

    # ── 1. Wöchentliche Daten ─────────────────────────────────────────────────
    print(f"\n[1/5] Wöchentliche Daten laden ...")
    weekly, X_test_base, test_region_ids = load_weekly()
    print(f"   {len(weekly):,} Rows, {weekly['region_id'].nunique()} Regionen  |  [{elapsed(t0)}]")

    # ── 2. V15-Features ───────────────────────────────────────────────────────
    print(f"\n[2/5] V15-Features berechnen (Score-Lags, Seasonal Dev) ...")
    weekly = add_v15_features(weekly)
    last_score = weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    for f in NUM_FEATURES:
        if f not in weekly.columns: weekly[f] = np.float32(0)
    print(f"   Done  |  [{elapsed(t0)}]")

    # ── 3. Sliding Windows ────────────────────────────────────────────────────
    print(f"\n[3/5] Sliding Windows ...")
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg     = sorted(weekly["region_id"].unique())
    val_regions = set(rng.choice(all_reg, max(1, int(len(all_reg)*VAL_REGION_FRAC)), replace=False))
    X_tr, y_tr, X_va, y_va, X_all, y_all = load_or_build_windows(
        weekly, val_regions, NUM_FEATURES, t0
    )
    print(f"   Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")

    # Baselines
    persist_va = np.column_stack(
        [last_score.reindex(sorted(val_regions)).fillna(0).to_numpy()] * 5
    )
    show_mae("Persistence-Baseline (letzter Score wiederholt)", y_va, persist_va)
    show_mae("score_lag1 wiederholt", y_va,
             np.column_stack([X_va["score_lag1"].to_numpy()] * 5))

    # ── 4. Training ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Training  |  [{elapsed(t0)}]")

    print("  LightGBM (5 Modelle, je 1 pro Vorhersage-Woche) ...")
    lgb_models = _train_lgb(X_tr, y_tr, X_va, y_va)
    lgb_val    = _pred_lgb(lgb_models, X_va)
    show_mae("LightGBM (gesamt)", y_va, lgb_val)
    for wk in range(5):
        n = _best_n(lgb_models[wk], N_ESTIMATORS)
        v = mae(y_va[:, wk], np.clip(lgb_models[wk].predict(X_va), 0, 5))
        print(f"    Woche {wk+1}: best_iter={n:4d}  MAE={v:.4f}")

    print("  XGBoost (5 Modelle) ...")
    xgb_models = _train_xgb(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
    xgb_val    = _pred_num(xgb_models, X_va, NUM_FEATURES)
    show_mae("XGBoost (gesamt)", y_va, xgb_val)

    cat_val, cat_models = None, None
    if CATBOOST_AVAILABLE:
        print("  CatBoost (5 Modelle) ...")
        cat_models = _train_cat(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
        cat_val    = _pred_num(cat_models, X_va, NUM_FEATURES)
        show_mae("CatBoost (gesamt)", y_va, cat_val)

    preds_val = {"lgb": lgb_val, "xgb": xgb_val}
    if cat_val is not None: preds_val["cat"] = cat_val
    best_w, best_val_mae = optimize_blend(y_va, preds_val)
    w_str = "  ".join(f"{k.upper()}={v:.2f}" for k, v in best_w.items())
    print(f"\n  Blend: {w_str}  →  MAE={best_val_mae:.4f}")

    # Feature Importance (LightGBM, Ø aller 5 Modelle)
    print_feature_importance(lgb_models, top_n=40)

    if QUICK_MODE:
        print("\n  QUICK_MODE=True — kein Final-Training, keine Submission.")
        print(f"  Gesamtlaufzeit: {elapsed(t0)}\n")
        return

    # ── 5. Final Training + Submission ───────────────────────────────────────
    print(f"\n[5/5] Final Training (alle Regionen)  |  [{elapsed(t0)}]")
    n_lgb = [_best_n(m, N_ESTIMATORS) for m in lgb_models]
    n_xgb = [_best_n(m, N_ESTIMATORS) for m in xgb_models]
    final_lgb = _train_lgb(X_all, y_all, None, None, n_lgb)
    final_xgb = _train_xgb(X_all, y_all, None, None, NUM_FEATURES, n_xgb)
    final_cat = None
    if CATBOOST_AVAILABLE and cat_models:
        n_cat     = [_best_n(m, N_ESTIMATORS) for m in cat_models]
        final_cat = _train_cat(X_all, y_all, None, None, NUM_FEATURES, n_cat)
    print(f"   Done  |  [{elapsed(t0)}]")

    # Test-Features aufbauen
    cache_base_cols = list(dict(np.load(WEEKLY_CACHE, allow_pickle=True))["feature_names"])
    X_test = pd.DataFrame(X_test_base, columns=cache_base_cols)
    X_test["region_id"] = pd.Categorical(test_region_ids)

    # Score-Lags: echte letzte Trainingswerte pro Region (nicht Kopie von lag1)
    _recent = {
        region: g["score"].tolist()
        for region, g in weekly.sort_values("ordinal").groupby("region_id")
    }
    def _get_lag(region: str, k: int) -> float:
        sc = _recent.get(region, [0.0])
        return float(sc[-k]) if len(sc) >= k else float(sc[0])
    for k, col in enumerate(["score_lag1","score_lag2","score_lag3","score_lag4","score_lag5"], 1):
        X_test[col] = np.array([_get_lag(r, k) for r in test_region_ids], dtype=np.float32)

    # Seasonal Dev: letzte bekannte saisonale Abweichung aus wöchentlichen Trainingsdaten
    last_w = weekly.sort_values("ordinal").groupby("region_id").last()
    for col in ["prec_seasonal_dev", "humidity_seasonal_dev", "tmp_seasonal_dev"]:
        if col in last_w.columns:
            X_test[col] = X_test["region_id"].map(last_w[col]).astype(np.float32).fillna(0)
        else:
            X_test[col] = np.float32(0)

    # Fehlende Features auf 0 (Fallback)
    for f in NUM_FEATURES:
        if f not in X_test.columns: X_test[f] = np.float32(0)

    lgb_test   = _pred_lgb(final_lgb, X_test)
    xgb_test   = _pred_num(final_xgb, X_test, NUM_FEATURES)
    test_preds = best_w["lgb"] * lgb_test + best_w["xgb"] * xgb_test
    if final_cat is not None and "cat" in best_w:
        test_preds += best_w["cat"] * _pred_num(final_cat, X_test, NUM_FEATURES)

    sub = pd.DataFrame({"region_id": test_region_ids})
    for k in range(5): sub[f"pred_week{k+1}"] = test_preds[:, k]
    template = pd.read_csv(SAMPLE_SUB)
    sub = template[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]: sub[col] = sub[col].fillna(0.0)
    sub.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*64}")
    print(f"  Submission: {OUT_PATH}  ({len(sub):,} Rows)")
    print(f"  Val MAE: {best_val_mae:.4f}  (lokale Val ≠ Kaggle-Score)")
    print(f"  Gesamtlaufzeit: {elapsed(t0)}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
