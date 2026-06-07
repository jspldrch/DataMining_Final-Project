"""
run_recent_local.py  —  Nur letzte N Jahre, kein Cache
=======================================================
Identisch zu run_v1_local.py, eine Änderung:
  Trainingsfenster werden NUR aus den letzten RECENT_YEARS Jahren gebaut.

Feature Engineering läuft noch auf allen Daten (Rolling-Stats brauchen
volle Geschichte). Nur die Sliding-Windows werden gefiltert.

Frage: Schaden alte Klimadaten? Generalisieren neuere Muster besser?

Output: outputs/submission_recent_local.csv

Vergleich:
  run_v1_local.py   → alle Jahre, kein score_lag
  run_recent_local.py → letzte N Jahre, kein score_lag  ← dieser Script
  run_v18 (lokal)   → alle Jahre, score_lag gap=13
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
    CAT = True
except ImportError:
    CAT = False

warnings.filterwarnings("ignore")

# ── Lokale Pfade ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
OUT_DIR    = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV  = DATA_DIR / "train.csv"
TEST_CSV   = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
OUT_PATH   = OUT_DIR  / "submission_recent_local.csv"

# ── Knobs ─────────────────────────────────────────────────────────────────────
RECENT_YEARS    = 8        # nur Trainings-Fenster aus den letzten N Jahren
RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0
WINDOW_STRIDE   = 1
N_ESTIMATORS    = 1000

# Ordinal-Kalender: year*372 + month*31 + day
# 1 Jahr ≈ 372 Ordinal-Einheiten
ORDINAL_PER_YEAR = 372

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


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} Min." if s >= 60 else f"{s:.0f}s"

def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))

def show(label: str, y: np.ndarray, p: np.ndarray) -> None:
    print(f"  {label:<48s}  MAE = {mae(y, p):.4f}")

def _best_n(m, default: int) -> int:
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(m, attr, None)
        if v is not None:
            return int(v)
    try:
        return int(m.get_best_iteration())
    except Exception:
        return default


# ── Feature-Liste ─────────────────────────────────────────────────────────────
def build_features() -> list[str]:
    f = list(WEATHER_COLS)
    f += [f"{c}_lag{l}"      for c in LAG_COLS  for l in LAGS]
    f += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS
          for s in ("mean", "std", "max")]
    f += ["month_sin", "month_cos", "day_sin", "day_cos"]
    f += ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
          "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    f.append("regional_mean_score")
    return f


# ── CSV laden ─────────────────────────────────────────────────────────────────
def _parse_dates(df: pd.DataFrame) -> None:
    p = df["date"].str.split("-", expand=True)
    df["year"]    = p[0].astype(np.int32)
    df["month"]   = p[1].astype(np.int32)
    df["day"]     = p[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]

def load_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"   Lese {TRAIN_CSV.name} ...")
    train = pd.read_csv(TRAIN_CSV)
    print(f"   Lese {TEST_CSV.name} ...")
    test  = pd.read_csv(TEST_CSV)
    _parse_dates(train)
    _parse_dates(test)
    train["score"] = pd.to_numeric(train["score"], errors="coerce").astype(np.float32)
    for df in (train, test):
        for col in WEATHER_COLS:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
    return train, test


# ── Feature Engineering (pro Region) ─────────────────────────────────────────
def _region_features(tr: pd.DataFrame, te: pd.DataFrame):
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc: dict = {}

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

    pp = panel["prec"].shift(1)
    nc["prec_deficit_90d"] = (
        pp.rolling(90, min_periods=30).mean() -
        pp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    p7  = pp.rolling(7,  min_periods=3).mean()
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
    anom = (
        tp.rolling(90, min_periods=30).mean() -
        tp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
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


# ── Feature Engineering über alle Regionen ───────────────────────────────────
def compute_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame, t0: float):
    region_means = train_raw.groupby("region_id")["score"].mean()
    regions = train_raw["region_id"].unique()
    tr_by   = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    te_by   = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw

    all_tr, all_te = [], []
    for i, region in enumerate(regions, 1):
        tf, ef = _region_features(tr_by[region], te_by.get(region, pd.DataFrame()))
        all_tr.append(tf)
        all_te.append(ef)
        if i % 500 == 0 or i == len(regions):
            print(f"   Region {i}/{len(regions)}  [{elapsed(t0)}]")

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

    X_test_df = (test_feat.sort_values(["region_id", "ordinal"])
                 .groupby("region_id", sort=False).tail(1)
                 [["region_id"] + base_cols].reset_index(drop=True))
    test_ids  = X_test_df["region_id"].values.astype(str)
    X_test    = X_test_df[base_cols].to_numpy(np.float32)

    return weekly, X_test, test_ids, base_cols


# ── Sliding Windows ───────────────────────────────────────────────────────────
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
        nw  = n - 5
        yr  = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        idx = list(range(0, nw, stride))
        if (nw - 1) not in idx:
            idx.append(nw - 1)
        Xp.append(Xn[idx]); yp.append(yr[idx]); rp.extend([region] * len(idx))
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

def print_importance(lgb_models: list, label: str = "") -> None:
    feat  = np.array(lgb_models[0].booster_.feature_name())
    imp   = sum(m.booster_.feature_importance("gain") for m in lgb_models) / len(lgb_models)
    mask  = feat != "region_id"
    feat, imp = feat[mask], imp[mask]
    total = imp.sum()
    order = np.argsort(imp)[::-1]

    # Gruppen-Summen
    weather_imp = imp[[i for i, f in enumerate(feat) if f in WEATHER_COLS]].sum()
    roll_imp    = imp[[i for i, f in enumerate(feat) if "roll" in f]].sum()
    lag_imp     = imp[[i for i, f in enumerate(feat) if "_lag" in f]].sum()
    drought_imp = imp[[i for i, f in enumerate(feat)
                       if any(k in f for k in ["deficit", "trend", "anomaly", "drought", "dry_days"])]].sum()
    region_imp  = imp[[i for i, f in enumerate(feat) if f == "regional_mean_score"]].sum()

    title = f"FEATURE IMPORTANCE  {label}(LGB Gain Ø, Woche 1-5)"
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"  {'Rang':<4}  {'Feature':<36}  {'%':>6}")
    for rank, i in enumerate(order[:20], 1):
        print(f"  {rank:<4d}  {feat[i]:<36}  {100*imp[i]/total:>5.2f}%")
    print(f"\n  Gruppen:")
    print(f"    Wetter (direkt):   {100*weather_imp/total:>5.1f}%")
    print(f"    Rolling Stats:     {100*roll_imp/total:>5.1f}%")
    print(f"    Lags:              {100*lag_imp/total:>5.1f}%")
    print(f"    Dürre-Indices:     {100*drought_imp/total:>5.1f}%")
    print(f"    regional_mean:     {100*region_imp/total:>5.1f}%")
    print(f"  Top-10 kumulativ:    {100*imp[order[:10]].sum()/total:.1f}%")
    print(f"{'─'*62}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0       = time.time()
    FEATURES = build_features()

    print("=" * 62)
    print(f"  run_recent_local  |  RECENT_YEARS={RECENT_YEARS}  |  kein score_lag")
    print(f"  Kein Cache — alles wird neu berechnet")
    print("=" * 62)

    # 1. CSV laden
    print(f"\n[1/5] CSV laden ...  [{elapsed(t0)}]")
    train_raw, test_raw = load_csv()
    print(f"   Train: {len(train_raw):,} Zeilen  |  Test: {len(test_raw):,} Zeilen")

    # 2. Feature Engineering (auf allen Daten — Rolling-Stats brauchen volle Geschichte)
    print(f"\n[2/5] Feature Engineering (auf allen Daten) ...  [{elapsed(t0)}]")
    weekly, X_test, test_ids, base_cols = compute_features(train_raw, test_raw, t0)
    del train_raw, test_raw
    print(f"   Weekly (alle Jahre): {len(weekly):,} Zeilen, "
          f"{weekly['region_id'].nunique()} Regionen  [{elapsed(t0)}]")

    for f in FEATURES:
        if f not in weekly.columns:
            weekly[f] = np.float32(0)

    # 3. RECENT-Filter: pro Region die letzten RECENT_YEARS Jahre behalten
    # Globaler Cutoff würde Regionen mit frühen Zeitperioden komplett ausschließen
    # (verschiedene Regionen decken verschiedene Zeitspannen ab).
    # Deshalb: jede Region behält nur ihre eigenen letzten RECENT_YEARS Jahre.
    def filter_recent_per_region(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for _, g in df.groupby("region_id", sort=False):
            cutoff = int(g["ordinal"].max()) - RECENT_YEARS * ORDINAL_PER_YEAR
            parts.append(g[g["ordinal"] >= cutoff])
        return pd.concat(parts, ignore_index=True)

    weekly_all    = weekly.copy()
    weekly_recent = filter_recent_per_region(weekly)

    n_recent = len(weekly_recent)
    n_total  = len(weekly)
    pct      = 100 * n_recent / n_total
    print(f"\n   Recent-Filter: letzte {RECENT_YEARS} Jahre pro Region (individueller Cutoff)")
    print(f"   Behalte {n_recent:,} / {n_total:,} wöchentliche Punkte ({pct:.0f}%)")

    # 4. Sliding Windows (aus recent-gefiltertem weekly)
    print(f"\n[3/5] Sliding Windows (nur letzte {RECENT_YEARS} Jahre) ...  [{elapsed(t0)}]")
    rng         = np.random.default_rng(RANDOM_STATE)
    all_reg     = sorted(weekly_recent["region_id"].unique())
    val_regions = set(rng.choice(all_reg, max(1, int(len(all_reg) * VAL_REGION_FRAC)), replace=False))

    X_tr,  y_tr  = _build_windows(weekly_recent, val_regions, FEATURES, WINDOW_STRIDE)
    X_va,  y_va  = _build_val(weekly_recent, sorted(val_regions), FEATURES)
    X_all, y_all = _build_windows(weekly_recent, set(), FEATURES, WINDOW_STRIDE)
    print(f"   Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")

    last_score = weekly_recent.sort_values("ordinal").groupby("region_id")["score"].last()
    show("Persistence-Baseline", y_va,
         np.column_stack([last_score.reindex(sorted(val_regions)).fillna(0).to_numpy()] * 5))

    # 5. Training
    print(f"\n[4/5] Training ...  [{elapsed(t0)}]")
    lgb_m   = train_lgb(X_tr, y_tr, X_va, y_va)
    lgb_val = pred_lgb(lgb_m, X_va)
    show("LightGBM", y_va, lgb_val)
    for wk in range(5):
        feat = lgb_m[wk].booster_.feature_name()
        v    = mae(y_va[:, wk], np.clip(lgb_m[wk].predict(X_va[feat]), 0, 5))
        print(f"    Woche {wk+1}: iter={_best_n(lgb_m[wk], N_ESTIMATORS):4d}  MAE={v:.4f}")

    xgb_m   = train_xgb(X_tr, y_tr, X_va, y_va, FEATURES)
    xgb_val = pred_num(xgb_m, X_va, FEATURES)
    show("XGBoost", y_va, xgb_val)

    preds_val = {"lgb": lgb_val, "xgb": xgb_val}
    cat_m = train_cat(X_tr, y_tr, X_va, y_va, FEATURES)
    if cat_m is not None:
        cat_val = pred_num(cat_m, X_va, FEATURES)
        show("CatBoost", y_va, cat_val)
        preds_val["cat"] = cat_val

    best_w, best_val_mae = optimize_blend(y_va, preds_val)
    w_str = "  ".join(f"{k}={v:.2f}" for k, v in best_w.items())
    print(f"\n  Blend: {w_str}  →  MAE = {best_val_mae:.4f}")

    print_importance(lgb_m, label=f"(recent={RECENT_YEARS}J) ")

    # 6. Final Training (auf recent-gefiltertem weekly)
    print(f"[5a/5] Final Training ...  [{elapsed(t0)}]")
    n_lgb = [_best_n(m, N_ESTIMATORS) for m in lgb_m]
    n_xgb = [_best_n(m, N_ESTIMATORS) for m in xgb_m]
    f_lgb = train_lgb(X_all, y_all, None, None, n_lgb)
    f_xgb = train_xgb(X_all, y_all, None, None, FEATURES, n_xgb)
    f_cat = None
    if cat_m:
        n_cat = [_best_n(m, N_ESTIMATORS) for m in cat_m]
        f_cat = train_cat(X_all, y_all, None, None, FEATURES, n_cat)
    print(f"   Fertig  [{elapsed(t0)}]")

    # 7. Submission
    print(f"\n[5b/5] Submission bauen ...  [{elapsed(t0)}]")
    X_test_df = pd.DataFrame(X_test, columns=base_cols)
    X_test_df["region_id"] = pd.Categorical(test_ids)
    for f in FEATURES:
        if f not in X_test_df.columns:
            X_test_df[f] = np.float32(0)

    test_preds = (best_w["lgb"] * pred_lgb(f_lgb, X_test_df) +
                  best_w["xgb"] * pred_num(f_xgb, X_test_df, FEATURES))
    if f_cat and "cat" in best_w:
        test_preds += best_w["cat"] * pred_num(f_cat, X_test_df, FEATURES)

    pred_df = pd.DataFrame({"region_id": test_ids})
    for k in range(5):
        pred_df[f"pred_week{k+1}"] = test_preds[:, k]

    template = pd.read_csv(SAMPLE_SUB)[["region_id"]]
    out = template.merge(pred_df, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        out[col] = out[col].fillna(0.0)
    out.to_csv(OUT_PATH, index=False)
    print(f"   Submission: {OUT_PATH.name}  ({len(out):,} Zeilen)")

    print(f"\n{'═'*62}")
    print(f"  ERGEBNIS  (RECENT_YEARS={RECENT_YEARS})")
    print(f"  Val MAE:           {best_val_mae:.4f}")
    print(f"  Trainings-Punkte:  {len(X_tr):,}  ({pct:.0f}% der vollen History)")
    print(f"  Referenz v12:      Kaggle 0.8258  (alle Jahre, kein score_lag)")
    print(f"  Laufzeit:          {elapsed(t0)}")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
