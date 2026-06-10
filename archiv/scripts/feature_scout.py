"""
feature_scout.py  –  Feature Importance in ~2-3 Min

Ziel: schnell herausfinden welche Features beitragen / Noise sind.
Kein Final-Training, keine Submission, kein Ensemble.

Wie:
  1. Lädt cache (_checkpoint_weekly.npz) — instant
  2. Baut Sliding Windows mit stride=16 — sehr wenige Samples, sehr schnell
  3. Trainiert EIN LightGBM (Ziel: Mittelwert Woche 1-3) mit 300 Bäumen
  4. Gibt Feature Importances aus, fertig

Kandidaten zum Testen (einfach aus EXTRA_FEATURES ein-/auskommentieren):
  - score_lag1/2/3         (autocorr 0.966 — erwartet: #1)
  - score_mean_4w          (mittlerer Score letzte 4 Wochen)
  - score_trend_4w         (Trend: wird Dürre besser/schlechter?)
  - score_streak           (Wochen in Folge mit Score > 0)
  - global_year_norm       (relative Epoche der Region, 0-1)
  - prec_weeks_dry         (Wochen ohne Niederschlag)
  - Einzelne Features auskommentieren um Noise zu entfernen

Usage:
    python scripts/feature_scout.py
Laufzeit: ~2-3 Min (mit Cache), ~25 Min (ohne Cache, erstellt Cache)
"""
from __future__ import annotations
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CACHE_DIR = DATA_DIR / "precomputed"
CACHE_DIR.mkdir(exist_ok=True)

WEEKLY_CACHE = CACHE_DIR / "_checkpoint_weekly.npz"
TRAIN_CSV    = DATA_DIR / "train.csv"
TEST_CSV     = DATA_DIR / "test.csv"

# ─── Knobs ────────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0
STRIDE          = 16    # Weniger Samples → schneller. Für finale Runs: 1
N_TREES         = 300   # Für Importance reicht das. Finale Runs: 1000
TARGET_WEEKS    = [0, 1, 2]  # Mittelwert dieser Wochen als Ziel (0=Woche1)

# ─── Feature-Kandidaten — ein-/auskommentieren zum Testen ─────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

# Basis-Features aus v12 — aus diesem Pool weglassen was laut Importance nichts bringt
USE_LAGS   = True   # tmp_range/max/tmp/prec/wind/surf_pre/humidity lags 1,3,7,14,21
USE_ROLLS  = True   # prec/wind/tmp/humidity rolling 7,14,30,60,90,180 (mean/std/max)
USE_CAL    = True   # month_sin/cos, day_sin/cos  (week_sin/cos laut Scout: Gain ~14 → weglassen)
USE_DROUGHT= True   # prec_deficit, prec_trend, humidity_deficit, tmp_anomaly, heat_drought, dry_days
USE_REGION = True   # regional_mean_score

# Neue Feature-Kandidaten
USE_SCORE_LAGS  = True   # score_lag1/2/3 — dominiert laut v13
USE_SCORE_ROLL  = True   # score_mean_4w, score_trend_4w, score_streak
USE_YEAR_FEAT   = True   # global_year_norm (Epoche der Region — spätere Epochen = mehr Dürre)
USE_PREC_DRY_WK  = True   # prec_weeks_dry (Wochen ohne Niederschlag)
USE_SEASONAL_DEV = True   # prec/humidity/tmp Abweichung vom Region-Monats-Mittelwert

# Noise-Kandidaten aus v13 — können raus:
DROP_WEEK_SINCOS = True  # week_sin/cos hatten Gain ~14 → definitiv raus
DROP_WIND_ROLL_MAX = True  # wind_roll*/max hatte sehr niedrige Importance

LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp", "humidity"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]


def build_feature_list() -> list[str]:
    feats: list[str] = list(WEATHER_COLS)

    if USE_LAGS:
        feats += [f"{c}_lag{l}" for c in LAG_COLS for l in LAGS]

    if USE_ROLLS:
        for col in ROLL_COLS:
            for w in ROLL_WINS:
                for stat in ("mean", "std", "max"):
                    name = f"{col}_roll{w}_{stat}"
                    if DROP_WIND_ROLL_MAX and col == "wind" and stat == "max":
                        continue
                    feats.append(name)

    if USE_CAL:
        feats += ["month_sin", "month_cos", "day_sin", "day_cos"]
        if not DROP_WEEK_SINCOS:
            feats += ["week_sin", "week_cos"]

    if USE_DROUGHT:
        feats += [
            "prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
            "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d",
        ]

    if USE_REGION:
        feats.append("regional_mean_score")

    # Neue Features (werden nach Cache-Load berechnet)
    if USE_SCORE_LAGS:
        feats += ["score_lag1", "score_lag2", "score_lag3"]
    if USE_SCORE_ROLL:
        feats += ["score_mean_4w", "score_trend_4w", "score_streak"]
    if USE_YEAR_FEAT:
        feats.append("global_year_norm")
    if USE_PREC_DRY_WK:
        feats.append("prec_weeks_dry")
    if USE_SEASONAL_DEV:
        feats += ["prec_seasonal_dev", "humidity_seasonal_dev", "tmp_seasonal_dev"]

    return feats


def elapsed(t0): return f"{(time.time()-t0)/60:.1f}m" if time.time()-t0 >= 60 else f"{time.time()-t0:.0f}s"

def mae(y_true, y_pred): return float(np.mean(np.abs(np.clip(y_pred, 0, 5) - y_true)))


def _parse_dates(df):
    p = df["date"].str.split("-", expand=True)
    df["year"]    = p[0].astype(np.int32)
    df["month"]   = p[1].astype(np.int32)
    df["day"]     = p[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]


def daily_to_weekly(df):
    wk = df["ordinal"] // WEEK_BUCKET
    return df.loc[df.groupby(wk, sort=False)["ordinal"].idxmax()].reset_index(drop=True)


def add_new_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Alle neuen Features die nicht im Cache sind."""
    weekly = weekly.sort_values(["region_id", "ordinal"]).copy()

    if USE_SCORE_LAGS or USE_SCORE_ROLL:
        g = weekly.groupby("region_id")["score"]
        if USE_SCORE_LAGS:
            weekly["score_lag1"] = g.transform(lambda x: x).astype(np.float32)
            weekly["score_lag2"] = g.shift(1).astype(np.float32)
            weekly["score_lag3"] = g.shift(2).astype(np.float32)
            weekly["score_lag2"] = weekly["score_lag2"].fillna(weekly["score_lag1"])
            weekly["score_lag3"] = weekly["score_lag3"].fillna(weekly["score_lag2"])

        if USE_SCORE_ROLL:
            # 4-Wochen-Mittelwert (nur vergangene Scores, kein Leakage)
            weekly["score_mean_4w"] = (
                weekly.groupby("region_id")["score"]
                .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
                .fillna(0).astype(np.float32)
            )
            # Trend: score jetzt minus score vor 4 Wochen (positiv = Dürre schlimmer)
            weekly["score_trend_4w"] = (
                weekly.groupby("region_id")["score"]
                .transform(lambda s: s.shift(1) - s.shift(4))
                .fillna(0).astype(np.float32)
            )
            # Streak: Wochen in Folge mit Score > 0
            weekly["score_streak"] = (
                weekly.groupby("region_id")["score"]
                .transform(lambda s: s.shift(1).gt(0).rolling(12, min_periods=1).sum())
                .fillna(0).astype(np.float32)
            )

    if USE_YEAR_FEAT:
        # Normiertes Jahr (0 = früheste Epoche im Dataset, 1 = späteste)
        global_min = weekly["ordinal"].min()
        global_max = weekly["ordinal"].max()
        weekly["global_year_norm"] = (
            (weekly["ordinal"] - global_min) / max(global_max - global_min, 1)
        ).astype(np.float32)

    if USE_PREC_DRY_WK:
        # Wochen in Folge ohne Niederschlag (prec < 1)
        if "prec" in weekly.columns:
            weekly["prec_weeks_dry"] = (
                weekly.groupby("region_id")["prec"]
                .transform(lambda s: s.shift(1).lt(DRY_THRESHOLD).rolling(12, min_periods=1).sum())
                .fillna(0).astype(np.float32)
            )

    if USE_SEASONAL_DEV:
        # Monat aus ordinal rekonstruieren (ordinal = year*372 + month*31 + day)
        weekly["_month"] = ((weekly["ordinal"] % 372) // 31).clip(0, 11)
        for col in ["prec", "humidity", "tmp"]:
            if col in weekly.columns:
                norm = weekly.groupby(["region_id", "_month"])[col].transform("mean")
                weekly[f"{col}_seasonal_dev"] = (weekly[col] - norm).astype(np.float32)
        weekly.drop(columns=["_month"], inplace=True)

    return weekly


def build_windows(weekly, skip_regions, features, stride=1):
    Xp, yp, rp = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions: continue
        g = g.sort_values("ordinal")
        sc = g["score"].to_numpy(np.float32)
        Xn = g[features].to_numpy(np.float32)
        n  = len(g)
        if n < 6: continue
        nw = n - 5
        # Ziel: Mittelwert der Target-Wochen
        yr_all = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        yr = yr_all[:, TARGET_WEEKS].mean(axis=1)
        idx = list(range(0, nw, stride))
        if (nw-1) not in idx: idx.append(nw-1)
        Xp.append(Xn[idx]); yp.append(yr[idx]); rp.extend([region]*len(idx))
    X = pd.DataFrame(np.vstack(Xp).astype(np.float32), columns=features)
    X["region_id"] = pd.Categorical(rp)
    return X, np.concatenate(yp).astype(np.float32)


def build_val(weekly, val_regions, features):
    Xp, yp, rp = [], [], []
    for region in val_regions:
        g = weekly.loc[weekly["region_id"]==region].sort_values("ordinal")
        if len(g) < 6: continue
        Xp.append(g.iloc[-6][features].to_numpy(np.float32))
        yp.append(g.iloc[-5:]["score"].to_numpy(np.float32).mean())
        rp.append(region)
    X = pd.DataFrame(np.vstack(Xp), columns=features)
    X["region_id"] = pd.Categorical(rp)
    return X, np.array(yp, np.float32)


def load_or_compute_weekly() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Lädt weekly data aus Cache oder berechnet neu + speichert Cache."""
    if WEEKLY_CACHE.exists():
        print(f"  Cache gefunden: {WEEKLY_CACHE.name}  ({WEEKLY_CACHE.stat().st_size/1e6:.0f} MB)")
        ck = dict(np.load(WEEKLY_CACHE, allow_pickle=True))
        base_feats = list(ck["feature_names"])
        weekly = pd.DataFrame(ck["weekly_feats"], columns=base_feats)
        weekly["score"]     = ck["weekly_scores"].astype(np.float32)
        weekly["region_id"] = ck["weekly_region"].astype(str)
        weekly["ordinal"]   = ck["weekly_ordinal"].astype(np.int32)
        X_test_arr      = ck["X_test"].astype(np.float32)
        test_region_ids = ck["test_region_ids"].astype(str)
        return weekly, X_test_arr, test_region_ids

    # Kein Cache — berechne neu (dauert ~20 Min)
    print("  Kein Cache — Feature Engineering (~20 Min) ...")
    from run_v13_diagnostic import (
        compute_region_features, WEATHER_COLS as WC,
        LAG_COLS as LC, LAGS as LG, ROLL_COLS as RC, ROLL_WINS as RW,
    )
    dtypes = {c: np.float32 for c in WC}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw  = pd.read_csv(TEST_CSV,  dtype=dtypes)
    _parse_dates(train_raw); _parse_dates(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    region_means = train_raw.groupby("region_id")["score"].mean()
    tr_by_r = {r: g.reset_index(drop=True) for r,g in train_raw.groupby("region_id",sort=False)}
    te_by_r = {r: g.reset_index(drop=True) for r,g in test_raw.groupby("region_id",sort=False)}
    del train_raw, test_raw
    all_tr, all_te = [], []
    for i, region in enumerate(regions, 1):
        tf, ef = compute_region_features(tr_by_r[region], te_by_r.get(region, pd.DataFrame()))
        all_tr.append(tf); all_te.append(ef)
        if i % 500 == 0: print(f"  Region {i}/{len(regions)}")
    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te
    base_feats = [f for f in build_feature_list()
                  if f not in ("score_lag1","score_lag2","score_lag3",
                               "score_mean_4w","score_trend_4w","score_streak",
                               "global_year_norm","prec_weeks_dry")]
    base_feats_in_data = [f for f in base_feats if f in train_feat.columns]
    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)
    labeled = train_feat[train_feat["score"].notna()].copy()
    weekly  = pd.concat([daily_to_weekly(g) for _,g in labeled.groupby("region_id",sort=False)],
                        ignore_index=True)
    del labeled
    X_test_df = (test_feat.sort_values(["region_id","ordinal"])
                 .groupby("region_id",sort=False).tail(1)
                 [["region_id"]+base_feats_in_data].reset_index(drop=True))
    test_region_ids = X_test_df["region_id"].values.astype(str)
    X_test_arr      = X_test_df[base_feats_in_data].to_numpy(np.float32)
    np.savez_compressed(WEEKLY_CACHE,
        weekly_feats   = weekly[base_feats_in_data].to_numpy(np.float32),
        weekly_scores  = weekly["score"].to_numpy(np.float32),
        weekly_region  = weekly["region_id"].values.astype(str),
        weekly_ordinal = weekly["ordinal"].to_numpy(np.int32),
        X_test         = X_test_arr, test_region_ids=test_region_ids,
        feature_names  = np.array(base_feats_in_data, dtype=object),
    )
    print(f"  Cache gespeichert ({WEEKLY_CACHE.stat().st_size/1e6:.0f} MB)")
    return weekly, X_test_arr, test_region_ids


def main():
    t0 = time.time()
    features = build_feature_list()
    print("=" * 62)
    print("  Feature Scout  (kein Final-Training, keine Submission)")
    print(f"  Features: {len(features)}  |  stride={STRIDE}  trees={N_TREES}")
    print(f"  Ziel: Woche {[w+1 for w in TARGET_WEEKS]} Mittelwert")
    print("=" * 62)

    # 1. Lade oder berechne weekly data
    print(f"\n[1/3] Lade Daten ...")
    weekly, _, _ = load_or_compute_weekly()

    # 2. Neue Features berechnen (Score-Lags, Trend, etc.)
    print(f"  Berechne neue Features ...  [{elapsed(t0)}]")
    weekly = add_new_features(weekly)

    # Fehlende Features mit 0 füllen
    for f in features:
        if f not in weekly.columns:
            print(f"  WARNUNG: Feature '{f}' fehlt — wird als 0 gesetzt")
            weekly[f] = np.float32(0)

    # 3. Train/Val Split
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg = sorted(weekly["region_id"].unique())
    n_val   = max(1, int(len(all_reg) * VAL_REGION_FRAC))
    val_regions = set(rng.choice(all_reg, size=n_val, replace=False))

    print(f"\n[2/3] Sliding Windows (stride={STRIDE}) ...  [{elapsed(t0)}]")
    X_tr, y_tr = build_windows(weekly, val_regions, features, stride=STRIDE)
    X_va, y_va = build_val(weekly, sorted(val_regions), features)
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}")

    # Persistence Baseline
    last_score = weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    p_baseline = last_score.reindex(sorted(val_regions)).fillna(0).to_numpy()
    print(f"  Persistence-Baseline  MAE = {mae(y_va, p_baseline):.4f}")

    # 4. EIN LightGBM trainieren
    print(f"\n[3/3] Trainiere LightGBM ({N_TREES} Bäume) ...  [{elapsed(t0)}]")
    m = lgb.LGBMRegressor(
        objective="regression", metric="mae",
        n_estimators=N_TREES, learning_rate=0.05,
        num_leaves=63, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1,
        random_state=RANDOM_STATE,
    )
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(30, verbose=False)],
        categorical_feature=["region_id"],
    )
    val_mae = mae(y_va, np.clip(m.predict(X_va), 0, 5))
    print(f"  LightGBM val MAE = {val_mae:.4f}  (best_iter={m.best_iteration_})")

    # 5. Feature Importance
    feat_names = np.array(m.booster_.feature_name())
    importance  = m.booster_.feature_importance(importance_type="gain").astype(float)
    mask       = feat_names != "region_id"
    feat_names  = feat_names[mask]
    importance  = importance[mask]
    total       = importance.sum()
    order       = np.argsort(importance)[::-1]

    print(f"\n{'─'*62}")
    print(f"  FEATURE IMPORTANCE  (LightGBM Gain, 1 Modell, Ziel=Woche1-3)")
    print(f"{'─'*62}")
    print(f"  {'Rank':<5}  {'Feature':<36}  {'Gain':>10}  {'%':>6}")
    print(f"  {'----':<5}  {'-------':<36}  {'----':>10}  {'--':>6}")
    for rank, i in enumerate(order[:50], 1):
        pct = 100 * importance[i] / total
        print(f"  {rank:<5d}  {feat_names[i]:<36}  {importance[i]:>10.0f}  {pct:>5.2f}%")

    print(f"\n  BOTTOM 15 (Noise-Kandidaten):")
    for i in order[-15:]:
        pct = 100 * importance[i] / total
        print(f"  {'':5}  {feat_names[i]:<36}  {importance[i]:>10.0f}  {pct:>5.2f}%")

    print(f"\n{'─'*62}")
    print(f"  Top-10 kumulativ: {100*importance[order[:10]].sum()/total:.1f}% des Gains")
    print(f"  Features mit < 0.01% Gain: {(importance/total < 0.0001).sum()} Stück → Noise")
    print(f"  Gesamtlaufzeit: {elapsed(t0)}")
    print(f"{'─'*62}\n")


if __name__ == "__main__":
    main()
