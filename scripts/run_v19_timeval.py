"""
run_v19_timeval.py  —  Time-based Validation  (lokale Version)
==============================================================
Basis: v12/v1 (nur Wetter-Features, KEIN score_lag)
Änderung: anderes Validierungs-Schema — Zeit-Schnitt statt Region-Holdout

Dieses Script beantwortet eine Frage:
  Gibt das neue Val-Schema (Zeit-Schnitt) eine Val-MAE,
  die näher an der echten Kaggle-MAE (~0.82) liegt?

=========================================================
ERKLÄRUNG: Was hier genau passiert
=========================================================

1. VAL-TARGETS (y) — vollständig rausgehalten
   ─────────────────────────────────────────
   Das Modell sieht in einem Trainings-Fenster:
     X[t] = Wetter-Features bei Woche t
     y[t] = Scores der Wochen t+1, t+2, t+3, t+4, t+5

   Beim Zeit-Schnitt gilt:
     Cutoff = letzter Ordinal - VAL_WEEKS Wochen
     Train-Fenster: alle mit X-PunktVOR dem Cutoff
     Val-Fenster:   X = Woche direkt vor Cutoff,
                    y = Scores der 5 Wochen NACH Cutoff

   Die y-Werte der Val-Fenster kommen aus echten Trainingsdaten
   (die Zukunft dieser Wochen IST bekannt), aber das Modell
   hat sie während des Trainings nie als Ziel gesehen.

   ┌────────────────────────────────────────────────┐
   │  Zeit →                                        │
   │  [=====Train-Fenster=====][  X  ][──y──]       │
   │                            ↑                   │
   │                          Cutoff                │
   │  Modell trainiert NUR auf dem linken Teil.     │
   │  Die 5 y-Wochen rechts sieht es nie im Train.  │
   └────────────────────────────────────────────────┘


2. SCORE ALS FEATURE — komplett ausgeschlossen
   ────────────────────────────────────────────
   v15/v17/v18 haben score_lag1 als Feature → Modell "sieht" den
   aktuellen Dürre-Zustand.

   v19 (wie v12/v1): KEIN score_lag.
   Das Modell weiß nur: aktuelle Wetterbedingungen, saisonale
   Anomalien, historische Muster. Der Score-Wert (0-5)
   kommt weder als Feature noch sonst in X vor.

   → Das Modell muss Dürre aus Wetter-Signalen lernen,
     nicht aus dem aktuellen Dürre-Zustand ableiten.


3. ALLE REGIONEN — Zeit-Schnitt statt Region-Holdout
   ───────────────────────────────────────────────────
   Bisher (Region-Holdout):
     20% der Regionen werden komplett aus Training entfernt.
     Val: letztes Window jeder dieser 449 Regionen.
     → Diese "letzten Wochen" sind zufällig gewählt,
       können stabil oder volatil sein.
     → Persistence-Baseline = 0.03 (Val-Regionen sind am Ende
       einer ruhigen Phase → unrealistisch leicht).

   Jetzt (Zeit-Schnitt):
     ALLE 2248 Regionen sind im Training bis zum Cutoff.
     Val: jede Region hat EINEN Val-Punkt am selben Cutoff.
     → 2248 Val-Punkte statt 449
     → Mischt stabile UND volatile Regionen
     → Testet: kann das Modell für ALLE Regionen gleichzeitig
       die nächsten 5 Wochen vorhersagen?

   ┌────────────────────────────────────────────────┐
   │  Region 1: [===train===][X][y y y y y]         │
   │  Region 2: [===train===][X][y y y y y]         │
   │  Region 3: [===train===][X][y y y y y]         │
   │  ...alle 2248 Regionen am gleichen Cutoff...   │
   └────────────────────────────────────────────────┘

=========================================================
Was wir erwarten:
  Persistence-Baseline > 0.03  → Val realistischer als bisher
  Model-MAE ≈ 0.82             → Val-Schema passt zum Kaggle-Test
  Model-MAE << 0.82            → Kaggle-Testdaten grundlegend anders

Checkpoints:
  data/precomputed/_checkpoint_weekly.npz       geteilt mit v15/v17
  data/precomputed/_checkpoint_v19_windows.npz  eigener Cache

Usage:  python scripts/run_v19_timeval.py
Output: outputs/submission_v19_timeval.csv
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

# ── Lokale Pfade ──────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CACHE_DIR = DATA_DIR / "precomputed"
OUT_DIR   = ROOT / "outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV  = DATA_DIR / "train.csv"
TEST_CSV   = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
OUT_PATH   = OUT_DIR  / "submission_v19_timeval.csv"

# Wöchentlicher Feature-Cache (geteilt mit v15/v17 — 495 MB, ~20 Min. Berechnung)
WEEKLY_CACHE  = CACHE_DIR / "_checkpoint_weekly.npz"
# Eigener Windows-Cache für Zeit-Schnitt
WINDOWS_CACHE = CACHE_DIR / "_checkpoint_v19_windows.npz"

# ── Knobs ─────────────────────────────────────────────────────────────────────
# Wie viele Wochen vor Trainings-Ende als Validation?
# 8 Wochen = ~56 Tage: genug Zeitabstand um Generalisierung zu testen
VAL_WEEKS     = 8

RANDOM_STATE  = 42
WEEK_BUCKET   = 7
DRY_THRESHOLD = 1.0
WINDOW_STRIDE = 1
N_ESTIMATORS  = 1000

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "humidity", "tmp", "wind"]   # wind im lokalen Cache vorhanden
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


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} Min." if s >= 60 else f"{s:.0f}s"

def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))

def show(label: str, y: np.ndarray, p: np.ndarray) -> None:
    print(f"  {label:<52s}  MAE = {mae(y, p):.4f}")

def _best_n(m, default: int) -> int:
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(m, attr, None)
        if v is not None:
            return int(v)
    try:
        return int(m.get_best_iteration())
    except Exception:
        return default

def _parse_dates(df: pd.DataFrame) -> None:
    p = df["date"].str.split("-", expand=True)
    df["year"]    = p[0].astype(np.int32)
    df["month"]   = p[1].astype(np.int32)
    df["day"]     = p[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]


# ── Feature-Liste (kein score_lag) ────────────────────────────────────────────
def build_features() -> list[str]:
    """
    Identisch zu v12/v1: nur Wetter-Features.
    score_lag ist NICHT enthalten — Score als Variable komplett ausgeschlossen.
    """
    f = list(WEATHER_COLS)
    f += [f"{c}_lag{l}"        for c in LAG_COLS  for l in LAGS]
    f += [f"{c}_roll{w}_{s}"   for c in ROLL_COLS for w in ROLL_WINS
          for s in ("mean", "std", "max")]
    f += ["month_sin", "month_cos", "day_sin", "day_cos"]
    f += ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
          "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    f.append("regional_mean_score")
    return f


# ── Feature Engineering (nur ohne Cache) ─────────────────────────────────────
def _region_features(tr: pd.DataFrame, te: pd.DataFrame):
    te = te.copy(); te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc: dict = {}
    nc["month_sin"] = np.sin(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["month_cos"] = np.cos(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["day_sin"]   = np.sin(2*np.pi*panel["day"]/31).astype(np.float32)
    nc["day_cos"]   = np.cos(2*np.pi*panel["day"]/31).astype(np.float32)
    woy = (panel["ordinal"] // 7) % 52
    nc["week_sin"]  = np.sin(2*np.pi*woy/52).astype(np.float32)
    nc["week_cos"]  = np.cos(2*np.pi*woy/52).astype(np.float32)
    for col in LAG_COLS:
        for lag in LAGS:
            nc[f"{col}_lag{lag}"] = panel[col].shift(lag).astype(np.float32)
    for col in ["prec", "humidity", "tmp", "wind"]:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=max(3, w//10))
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


# ── Wöchentlicher Cache ───────────────────────────────────────────────────────
def load_weekly(t0: float):
    """
    Lädt aus bestehendem Cache (_checkpoint_weekly.npz, geteilt mit v15/v17).
    Cache-Key: 'X_test' (lokaler Key, anders als Kaggle-Version 'X_test_base').
    """
    if WEEKLY_CACHE.exists():
        print(f"   Cache: {WEEKLY_CACHE.name}  ({WEEKLY_CACHE.stat().st_size/1e6:.0f} MB)")
        ck   = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base = list(ck["feature_names"])
        weekly = pd.DataFrame(ck["weekly_feats"], columns=base)
        weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        weekly["region_id"] = ck["weekly_region"].astype(str)
        weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
        # Lokaler Cache nutzt 'X_test', Kaggle-Cache nutzt 'X_test_base'
        x_key = "X_test" if "X_test" in ck else "X_test_base"
        return weekly, ck[x_key].astype(np.float32), ck["test_region_ids"].astype(str), base

    print(f"   Kein Cache — Feature Engineering (~20 Min) ...  [{elapsed(t0)}]")
    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw  = pd.read_csv(TEST_CSV)
    _parse_dates(train_raw); _parse_dates(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    for df in (train_raw, test_raw):
        for col in WEATHER_COLS:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

    regions      = train_raw["region_id"].unique()
    region_means = train_raw.groupby("region_id")["score"].mean()
    tr_by = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    te_by = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id", sort=False)}
    del train_raw, test_raw

    all_tr, all_te = [], []
    for i, region in enumerate(regions, 1):
        tf, ef = _region_features(tr_by[region], te_by.get(region, pd.DataFrame()))
        all_tr.append(tf); all_te.append(ef)
        if i % 500 == 0 or i == len(regions):
            print(f"   Region {i}/{len(regions)}  [{elapsed(t0)}]")

    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te

    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)

    labeled   = train_feat[train_feat["score"].notna()].copy()
    weekly    = pd.concat(
        [_daily_to_weekly(g) for _, g in labeled.groupby("region_id", sort=False)],
        ignore_index=True,
    )
    del labeled
    base_cols = [c for c in weekly.columns
                 if c not in ("score", "region_id", "ordinal", "date", "year", "month", "day")]
    X_test_df = (test_feat.sort_values(["region_id", "ordinal"])
                 .groupby("region_id", sort=False).tail(1)
                 [["region_id"] + base_cols].reset_index(drop=True))
    test_ids  = X_test_df["region_id"].values.astype(str)
    X_test_arr = X_test_df[base_cols].to_numpy(np.float32)

    np.savez_compressed(WEEKLY_CACHE,
        weekly_feats    = weekly[base_cols].to_numpy(np.float32),
        weekly_scores   = weekly["score"].to_numpy(np.float32),
        weekly_region   = weekly["region_id"].values.astype(str),
        weekly_ordinal  = weekly["ordinal"].to_numpy(np.int32),
        X_test          = X_test_arr,
        test_region_ids = test_ids,
        feature_names   = np.array(base_cols, dtype=object),
    )
    print(f"   Cache gespeichert  [{elapsed(t0)}]")
    return weekly, X_test_arr, test_ids, base_cols


# ── Zeit-basierte Sliding Windows ─────────────────────────────────────────────
def build_timeval_windows(weekly: pd.DataFrame, features: list, val_weeks: int, t0: float):
    """
    Erstellt Trainings- und Val-Fenster mit gemeinsamem Zeit-Schnitt.

    Cutoff = max(ordinal) - val_weeks * WEEK_BUCKET
    Für jede Region:
      Train: alle Fenster mit X-Punkt-Ordinal < cutoff
      Val:   letztes Fenster vor cutoff  (X = Woche t, y = Wochen t+1..t+5)

    Die Val-y-Werte sind echte Trainingsdaten-Scores — nie während
    des Trainings als Ziel verwendet (komplett rausgehalten).
    """
    max_ord = int(weekly["ordinal"].max())
    cutoff  = max_ord - val_weeks * WEEK_BUCKET

    print(f"   Cutoff-Ordinal: {cutoff}  |  Max: {max_ord}")
    print(f"   Val-Fenster: letzte {val_weeks} Wochen aller Regionen")

    Xtr, ytr, rtr = [], [], []
    Xva, yva, rva = [], [], []
    skipped = 0

    for region, g in weekly.groupby("region_id", sort=False):
        g    = g.sort_values("ordinal")
        sc   = g["score"].to_numpy(np.float32)
        Xn   = g[features].to_numpy(np.float32)
        ords = g["ordinal"].to_numpy(np.int32)
        n    = len(g)
        if n < 7:  # mindestens 7 Wochen: 1 Val-X + 5 Val-y + 1 Train
            skipped += 1
            continue

        # Letzter Index i mit ords[i] <= cutoff UND i+5 < n (y braucht 5 Folge-Wochen)
        val_i = None
        for i in range(n - 6, -1, -1):
            if ords[i] <= cutoff:
                val_i = i
                break

        if val_i is None or val_i < 1:
            skipped += 1
            continue

        # Val-Fenster: X = Woche val_i, y = Wochen val_i+1 .. val_i+5
        Xva.append(Xn[val_i])
        yva.append(sc[val_i + 1: val_i + 6])
        rva.append(region)

        # Train-Fenster: alle Fenster mit X-Punkt-Index < val_i
        nw   = n - 5
        yr   = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        stop = min(val_i, nw)
        if stop < 1:
            continue
        idx = list(range(0, stop, WINDOW_STRIDE))
        if (stop - 1) not in idx:
            idx.append(stop - 1)
        Xtr.append(Xn[idx]); ytr.append(yr[idx]); rtr.extend([region] * len(idx))

    X_tr = pd.DataFrame(np.vstack(Xtr).astype(np.float32), columns=features)
    X_tr["region_id"] = pd.Categorical(rtr)
    X_va = pd.DataFrame(np.vstack(Xva).astype(np.float32), columns=features)
    X_va["region_id"] = pd.Categorical(rva)

    print(f"   Regionen mit Val-Punkt: {len(rva):,}  (übersprungen: {skipped})")
    return X_tr, np.vstack(ytr), X_va, np.vstack(yva), cutoff


def build_all_windows(weekly: pd.DataFrame, features: list):
    """Alle Fenster — für Final-Training ohne Holdout."""
    Xp, yp, rp = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        g  = g.sort_values("ordinal")
        sc = g["score"].to_numpy(np.float32)
        Xn = g[features].to_numpy(np.float32)
        n  = len(g)
        if n < 6: continue
        nw  = n - 5
        yr  = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        idx = list(range(0, nw, WINDOW_STRIDE))
        if (nw - 1) not in idx: idx.append(nw - 1)
        Xp.append(Xn[idx]); yp.append(yr[idx]); rp.extend([region] * len(idx))
    X = pd.DataFrame(np.vstack(Xp).astype(np.float32), columns=features)
    X["region_id"] = pd.Categorical(rp)
    return X, np.vstack(yp).astype(np.float32)


def load_or_build_windows(weekly, features, val_weeks, t0):
    if WINDOWS_CACHE.exists():
        ck = dict(np.load(WINDOWS_CACHE, allow_pickle=True))
        ok = (list(ck["feature_names"]) == features and
              int(ck.get("val_weeks", np.array([0]))[0]) == val_weeks)
        if ok:
            print(f"   Cache: {WINDOWS_CACHE.name}  ({WINDOWS_CACHE.stat().st_size/1e6:.0f} MB)")
            def _r(p):
                X = pd.DataFrame(ck[f"X_{p}"], columns=features)
                X["region_id"] = pd.Categorical(ck[f"r_{p}"].astype(str).tolist())
                return X, ck[f"y_{p}"]
            return *_r("tr"), *_r("va"), int(ck["cutoff"][0]), *_r("all")
        print("   Windows-Cache veraltet — neu berechnen ...")

    print(f"   Berechne Zeit-Fenster (val_weeks={val_weeks}) ...  [{elapsed(t0)}]")
    X_tr, y_tr, X_va, y_va, cutoff = build_timeval_windows(weekly, features, val_weeks, t0)
    X_all, y_all = build_all_windows(weekly, features)

    np.savez_compressed(WINDOWS_CACHE,
        X_tr  = X_tr[features].to_numpy(np.float32),  y_tr  = y_tr,
        r_tr  = np.array(X_tr["region_id"].astype(str), dtype=object),
        X_va  = X_va[features].to_numpy(np.float32),  y_va  = y_va,
        r_va  = np.array(X_va["region_id"].astype(str), dtype=object),
        X_all = X_all[features].to_numpy(np.float32), y_all = y_all,
        r_all = np.array(X_all["region_id"].astype(str), dtype=object),
        feature_names = np.array(features, dtype=object),
        val_weeks     = np.array([val_weeks]),
        cutoff        = np.array([cutoff]),
    )
    print(f"   Windows-Cache gespeichert  [{elapsed(t0)}]")
    return X_tr, y_tr, X_va, y_va, cutoff, X_all, y_all


# ── Training ──────────────────────────────────────────────────────────────────
def train_lgb(X_tr, y_tr, X_va, y_va, n_trees=None):
    models = []
    for wk in range(5):
        n = (n_trees[wk] if n_trees else None) or LGB_P["n_estimators"]
        m = lgb.LGBMRegressor(**dict(LGB_P, random_state=RANDOM_STATE + wk, n_estimators=n))
        kw = dict(categorical_feature=["region_id"])
        if X_va is not None:
            kw.update(eval_set=[(X_va, y_va[:, wk].ravel())], eval_metric="mae",
                      callbacks=[lgb.early_stopping(50, verbose=False)])
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
    if not CATBOOST_AVAILABLE: return None
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
    return np.clip(np.column_stack([m.predict(X[feat]) for m in models]), 0, 5).astype(np.float32)

def pred_num(models, X, features):
    Xn = X[features].to_numpy(np.float32)
    return np.clip(np.column_stack([m.predict(Xn) for m in models]), 0, 5).astype(np.float32)

def optimize_blend(y_va, preds: dict):
    names  = list(preds)
    arrays = [preds[n] for n in names]
    alphas = [round(x * 0.05, 2) for x in range(1, 20)]
    best_mae, best_w = 999.0, {n: 1/len(names) for n in names}
    if len(names) == 2:
        for a in alphas:
            m = mae(y_va, a*arrays[0] + (1-a)*arrays[1])
            if m < best_mae:
                best_mae, best_w = m, {names[0]: a, names[1]: round(1-a, 8)}
    elif len(names) == 3:
        for a in alphas:
            for b in alphas:
                c = round(1 - a - b, 8)
                if c < 0.05: continue
                m = mae(y_va, a*arrays[0] + b*arrays[1] + c*arrays[2])
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
    print(f"\n{'─'*62}")
    print(f"  FEATURE IMPORTANCE  (LGB Gain Ø, Woche 1-5)")
    print(f"  {'Rang':<4}  {'Feature':<36}  {'%':>6}")
    for rank, i in enumerate(order[:15], 1):
        print(f"  {rank:<4d}  {feat[i]:<36}  {100*imp[i]/total:>5.2f}%")
    print(f"  Top-10 kumulativ: {100*imp[order[:10]].sum()/total:.1f}%")
    print(f"{'─'*62}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0       = time.time()
    FEATURES = build_features()

    print("=" * 62)
    print("  run_v19_timeval  —  Zeit-basierte Validierung")
    print(f"  VAL_WEEKS={VAL_WEEKS}  |  Features={len(FEATURES)}  |  kein score_lag")
    print("=" * 62)
    print(f"""
  3 Kernunterschiede zum bisherigen Val-Schema:

  1. Val-Targets (y): Die Scores der 5 Wochen NACH dem Cutoff
     sieht das Modell im Training NIE. Komplett rausgehalten.

  2. Score als Feature: v19 basiert auf v1 (wie v12).
     score_lag ist NICHT in X. Das Modell kennt den aktuellen
     Dürre-Zustand nicht — nur Wetter-Signale.

  3. Alle 2248 Regionen statt 20% Holdout:
     Jede Region hat EINEN Val-Punkt am gleichen Zeit-Cutoff.
     → Mischt stabile + volatile Regionen (realistischer)
     → Bisheriger Region-Holdout hatte Persistence-MAE = 0.03
       (nur stabile Phasen im Val) → zu optimistisch
""")

    # 1. Weekly Features
    print(f"[1/5] Wöchentliche Features laden ...  [{elapsed(t0)}]")
    weekly, X_test_raw, test_ids, base_cols = load_weekly(t0)
    n_regions = weekly["region_id"].nunique()
    print(f"   {len(weekly):,} Zeilen, {n_regions} Regionen  [{elapsed(t0)}]")

    # Sicherstellen dass alle Features vorhanden sind
    for f in FEATURES:
        if f not in weekly.columns:
            weekly[f] = np.float32(0)

    # 2. Zeit-basierte Sliding Windows
    print(f"\n[2/5] Zeit-basierte Fenster (val_weeks={VAL_WEEKS}) ...  [{elapsed(t0)}]")
    X_tr, y_tr, X_va, y_va, cutoff, X_all, y_all = load_or_build_windows(
        weekly, FEATURES, VAL_WEEKS, t0)
    print(f"   Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")

    # Persistence-Baseline (letzter bekannter Score wiederholt)
    # Entscheidend: ist dieser MAE >> 0.03 (Region-Holdout)?
    last_score_at_cutoff = (
        weekly.loc[weekly["ordinal"] <= cutoff]
        .sort_values("ordinal")
        .groupby("region_id")["score"].last()
    )
    val_region_list = X_va["region_id"].astype(str).tolist()
    persist_pred = np.column_stack([
        last_score_at_cutoff.reindex(val_region_list).fillna(0).to_numpy()
    ] * 5)
    persist_mae = mae(y_va, persist_pred)
    show("Persistence-Baseline (letzter Score wiederholt)", y_va, persist_pred)
    for wk in range(5):
        p = mae(y_va[:, wk], persist_pred[:, wk])
        print(f"    Woche {wk+1} Persistence-MAE: {p:.4f}")

    # 3. Training
    print(f"\n[3/5] Training  [{elapsed(t0)}]")
    lgb_m   = train_lgb(X_tr, y_tr, X_va, y_va)
    lgb_val = pred_lgb(lgb_m, X_va)
    show("LightGBM (Zeit-Val)", y_va, lgb_val)
    for wk in range(5):
        feat = lgb_m[wk].booster_.feature_name()
        v    = mae(y_va[:, wk], np.clip(lgb_m[wk].predict(X_va[feat]), 0, 5))
        print(f"    Woche {wk+1}: best_iter={_best_n(lgb_m[wk], N_ESTIMATORS):4d}  MAE={v:.4f}")

    xgb_m   = train_xgb(X_tr, y_tr, X_va, y_va, FEATURES)
    xgb_val = pred_num(xgb_m, X_va, FEATURES)
    show("XGBoost  (Zeit-Val)", y_va, xgb_val)

    preds_val = {"lgb": lgb_val, "xgb": xgb_val}
    cat_m = train_cat(X_tr, y_tr, X_va, y_va, FEATURES)
    if cat_m is not None:
        cat_val = pred_num(cat_m, X_va, FEATURES)
        show("CatBoost (Zeit-Val)", y_va, cat_val)
        preds_val["cat"] = cat_val

    best_w, best_val_mae = optimize_blend(y_va, preds_val)
    w_str = "  ".join(f"{k}={v:.2f}" for k, v in best_w.items())
    print(f"\n  Blend: {w_str}  →  MAE = {best_val_mae:.4f}")
    print_importance(lgb_m)

    # ── Ergebnis-Auswertung ───────────────────────────────────────────────────
    print(f"{'═'*62}")
    print(f"  ERGEBNIS: Zeit-basierte Validierung  (val_weeks={VAL_WEEKS})")
    print(f"{'═'*62}")
    print(f"  Val-Regionen:          {len(X_va):,}  (alle Regionen am Cutoff)")
    print(f"  Persistence-MAE:       {persist_mae:.4f}")
    print(f"  Modell-MAE (Blend):    {best_val_mae:.4f}")
    print(f"\n  Referenz: Kaggle-MAE v12 = 0.8258")
    print(f"  Referenz: Region-Holdout Persistence = 0.0321  ← zum Vergleich")
    print(f"\n  INTERPRETATION:")

    if persist_mae > 0.30:
        print(f"  [+] Persistence-MAE {persist_mae:.4f} >> 0.03:")
        print(f"      Zeit-Val erfasst echte Score-Variabilität.")
        print(f"      Val-Schema ist deutlich realistischer als Region-Holdout.")
    elif persist_mae > 0.10:
        print(f"  [~] Persistence-MAE {persist_mae:.4f} > 0.03 aber nicht sehr groß:")
        print(f"      Leichte Verbesserung; Scores sind am Cutoff noch teilweise stabil.")
    else:
        print(f"  [-] Persistence-MAE {persist_mae:.4f} ≈ 0.03:")
        print(f"      Auch am Zeit-Cutoff sind die Scores sehr stabil.")
        print(f"      Die Trainings-Endphase ist generell eine ruhige Periode.")

    if best_val_mae > 0.60:
        print(f"\n  [OK] Modell-MAE {best_val_mae:.4f} liegt im Kaggle-Bereich!")
        print(f"       Zeit-Val ist ein zuverlässiges Optimierungs-Signal.")
    elif best_val_mae > 0.30:
        print(f"\n  [~]  Modell-MAE {best_val_mae:.4f}: realistischer als bisher,")
        print(f"       aber noch unter Kaggle-MAE. Testdaten könnten schwieriger sein.")
    else:
        print(f"\n  [!]  Modell-MAE {best_val_mae:.4f}: immer noch sehr niedrig.")
        print(f"       Kaggle-Testdaten kommen aus einer anderen Verteilung.")
    print(f"{'═'*62}\n")

    # 4. Final-Training
    print(f"[4/5] Final-Training (alle Regionen)  [{elapsed(t0)}]")
    n_lgb = [_best_n(m, N_ESTIMATORS) for m in lgb_m]
    n_xgb = [_best_n(m, N_ESTIMATORS) for m in xgb_m]
    f_lgb = train_lgb(X_all, y_all, None, None, n_lgb)
    f_xgb = train_xgb(X_all, y_all, None, None, FEATURES, n_xgb)
    f_cat = None
    if cat_m:
        n_cat = [_best_n(m, N_ESTIMATORS) for m in cat_m]
        f_cat = train_cat(X_all, y_all, None, None, FEATURES, n_cat)
    print(f"   Fertig  [{elapsed(t0)}]")

    # 5. Test-Prediction
    print(f"\n[5/5] Submission  [{elapsed(t0)}]")
    X_test = pd.DataFrame(X_test_raw, columns=base_cols)
    X_test["region_id"] = pd.Categorical(test_ids)
    for f in FEATURES:
        if f not in X_test.columns: X_test[f] = np.float32(0)

    test_preds = (best_w["lgb"] * pred_lgb(f_lgb, X_test) +
                  best_w["xgb"] * pred_num(f_xgb, X_test, FEATURES))
    if f_cat and "cat" in best_w:
        test_preds += best_w["cat"] * pred_num(f_cat, X_test, FEATURES)

    sub = pd.read_csv(SAMPLE_SUB)[["region_id"]]
    pred_df = pd.DataFrame({"region_id": test_ids})
    for k in range(5): pred_df[f"pred_week{k+1}"] = test_preds[:, k]
    sub = sub.merge(pred_df, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col] = sub[col].fillna(0.0)
    sub.to_csv(OUT_PATH, index=False)
    print(f"   Submission: {OUT_PATH.name}  ({len(sub):,} Zeilen)")

    print(f"\n  Zeit-Val MAE:       {best_val_mae:.4f}")
    print(f"  Persistence-MAE:    {persist_mae:.4f}  (Region-Holdout war: 0.0321)")
    print(f"  Laufzeit: {elapsed(t0)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
