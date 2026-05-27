"""
run_v3.py  –  Standalone Training + Submission (kein Colab nötig)

Starte aus dem Projekt-Root-Ordner mit:
    python scripts/run_v3.py

Laufzeit auf 16 GB RAM / 8 Kerne: ~12-20 Minuten
Output: outputs/submission_v3.csv

Verbesserungen gegenueber 04_modeling.ipynb (MAE 0.8727):
  1. Laengere Rolling-Fenster  - 30 / 60 / 90 Tage statt nur 7 / 14
  2. Niederschlags-Defizit     - 90-Tage-Mittel minus 365-Tage-Baseline
  3. Prec-Trend                - normalisierter Unterschied 7d vs. 30d-Mittel
  4. Score-Lags                - letzte bekannte Wochenwerte aus Trainingsdaten
  5. LGB + XGB Ensemble        - 50/50-Blend zweier GBDT-Implementierungen
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

# ─── Pfade ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB = ROOT / "resources" / "sample_submission.csv"
OUT_PATH = OUT_DIR / "submission_v3.csv"

# ─── Modus ───────────────────────────────────────────────────────────────────
# QUICK_MODE = True  -> ~15 Minuten, gut zum Testen der Pipeline
# QUICK_MODE = False -> ~60 Minuten, vollstaendiges Training fuer Kaggle
QUICK_MODE = True

# ─── Hyperparameter ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET = 7

# Im Quick-Mode: nur jedes N-te Fenster pro Region verwenden
# 4 = 4x weniger Samples, MAE-Unterschied minimal (Fenster sind stark korreliert)
WINDOW_STRIDE = 1 if not QUICK_MODE else 4
# Im Quick-Mode: weniger Baeume (Early Stopping stoppt sowieso frueh)
N_ESTIMATORS = 900 if not QUICK_MODE else 400

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre"]
LAGS = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp"]
ROLL_WINS = [7, 14, 30, 60, 90]   # Verbesserung 1: 30/60/90 neu
SCORE_LAGS = []  # score lags removed: at test time all equal same forward-filled value

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=60,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    max_depth=6,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    verbosity=0,
)

NUM_FEATURES: list[str] = []  # wird in build_feature_list() gefuellt


# ─── Feature-Engineering fuer eine Region ─────────────────────────────────────

def _parse_dates_inplace(df: pd.DataFrame) -> None:
    parts = df["date"].str.split("-", expand=True)
    df["year"] = parts[0].astype(np.int32)
    df["month"] = parts[1].astype(np.int32)
    df["day"] = parts[2].astype(np.int32)
    df["ordinal"] = (
        df["year"] * 372
        + df["month"] * 31
        + df["day"]
    )


def build_feature_list() -> list[str]:
    lag_names = [f"{c}_lag{lag}" for c in LAG_COLS for lag in LAGS]
    roll_names = [
        f"{col}_roll{w}_{stat}"
        for col in ROLL_COLS
        for w in ROLL_WINS
        for stat in ("mean", "std", "max")
    ]
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos"]
    drought = ["prec_deficit_90d", "prec_trend_30d"]
    return WEATHER_COLS + lag_names + roll_names + calendar + drought


def compute_region_features(
    tr: pd.DataFrame,
    te: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kombiniert train+test einer Region und berechnet alle Features auf dem
    gemeinsamen Panel. So fliessen Rolling-Fenster und Score-Lags sauber
    aus dem Training in den Test-Zeitraum (kein Look-ahead).
    """
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)

    # Kalender
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    panel["day_sin"] = np.sin(2 * np.pi * panel["day"] / 31).astype(np.float32)
    panel["day_cos"] = np.cos(2 * np.pi * panel["day"] / 31).astype(np.float32)

    # Lag-Features
    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS:
            panel[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)

    # Rolling-Features (shift(1) verhindert Look-ahead)
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=3)
            panel[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            panel[f"{col}_roll{w}_std"] = r.std().astype(np.float32)
            panel[f"{col}_roll{w}_max"] = r.max().astype(np.float32)

    # Verbesserung 2: Niederschlags-Defizit
    # Negativer Wert = trockener als Jahresdurchschnitt -> Duerreindikator
    prec_prior = panel["prec"].shift(1)
    panel["prec_deficit_90d"] = (
        prec_prior.rolling(90, min_periods=30).mean()
        - prec_prior.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    # Verbesserung 3: Niederschlags-Trend (wird es trockener?)
    # Negativ = kurzfristig weniger Regen als langfristiger Schnitt
    p7 = prec_prior.rolling(7, min_periods=3).mean()
    p30 = prec_prior.rolling(30, min_periods=10).mean()
    p30_std = prec_prior.rolling(30, min_periods=10).std().clip(lower=0.01)
    panel["prec_trend_30d"] = ((p7 - p30) / p30_std).astype(np.float32)

    n_tr = len(tr)
    return panel.iloc[:n_tr].copy(), panel.iloc[n_tr:].copy()


# ─── Datensatz-Aufbau ─────────────────────────────────────────────────────────

def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    week = df["ordinal"] // WEEK_BUCKET
    idx = df.groupby(week, sort=False)["ordinal"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_sliding_windows(
    weekly: pd.DataFrame,
    skip_regions: set,
    num_features: list[str],
    stride: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sliding-Window: Feature bei Woche i -> Scores i+1..i+5.
    stride > 1: nur jedes N-te Fenster (schneller, kaum MAE-Unterschied).
    Das letzte Fenster wird immer behalten (aktuellste Information).
    """
    X_parts, y_parts, region_parts = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions:
            continue
        g = g.sort_values("ordinal")
        scores = g["score"].to_numpy(dtype=np.float32)
        X_num = g[num_features].to_numpy(dtype=np.float32)
        n = len(g)
        if n < 6:
            continue
        n_win = n - 5
        y_reg = np.lib.stride_tricks.sliding_window_view(scores[1:], 5)[:n_win]
        # Stride: jedes stride-te Fenster + immer das letzte
        idx = list(range(0, n_win, stride))
        if (n_win - 1) not in idx:
            idx.append(n_win - 1)
        X_parts.append(X_num[idx])
        y_parts.append(y_reg[idx])
        region_parts.extend([region] * len(idx))

    X_df = pd.DataFrame(np.vstack(X_parts).astype(np.float32), columns=num_features)
    X_df["region_id"] = pd.Categorical(region_parts)
    return X_df, np.vstack(y_parts).astype(np.float32)


def build_val_samples(
    weekly: pd.DataFrame,
    val_regions: list,
    num_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
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


# ─── Modell-Hilfsfunktionen ───────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(y_pred, 0, 5) - y_true)))


def show_mae(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"  {name:<48s}  MAE = {mae(y_true, y_pred):.4f}")


def train_lgb_models(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
    n_trees_per_week: list[int] | None = None,
) -> list[lgb.LGBMRegressor]:
    models = []
    for week in range(5):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or LGB_PARAMS["n_estimators"]
        p = dict(LGB_PARAMS, random_state=RANDOM_STATE + week, n_estimators=n)
        m = lgb.LGBMRegressor(**p)
        fit_kw: dict = dict(categorical_feature=["region_id"])
        if X_va is not None:
            fit_kw["eval_set"] = [(X_va, y_va[:, week].ravel())]
            fit_kw["eval_metric"] = "mae"
            fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        m.fit(X_tr, y_tr[:, week].ravel(), **fit_kw)
        models.append(m)
    return models


def train_xgb_models(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
    num_features: list[str],
    n_trees_per_week: list[int] | None = None,
) -> list[xgb.XGBRegressor]:
    X_tr_n = X_tr[num_features].to_numpy(dtype=np.float32)
    X_va_n = X_va[num_features].to_numpy(dtype=np.float32) if X_va is not None else None
    models = []
    for week in range(5):
        n = (n_trees_per_week[week] if n_trees_per_week else None) or XGB_PARAMS["n_estimators"]
        p = dict(XGB_PARAMS, random_state=RANDOM_STATE + week, n_estimators=n)
        fit_kw: dict = {}
        if X_va_n is not None:
            p["early_stopping_rounds"] = 50
            fit_kw["eval_set"] = [(X_va_n, y_va[:, week].ravel())]
            fit_kw["verbose"] = False
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr_n, y_tr[:, week].ravel(), **fit_kw)
        models.append(m)
    return models


def predict_lgb(models: list, X: pd.DataFrame) -> np.ndarray:
    return np.clip(
        np.column_stack([m.predict(X) for m in models]), 0.0, 5.0
    ).astype(np.float32)


def predict_xgb(models: list, X: pd.DataFrame, num_features: list[str]) -> np.ndarray:
    X_n = X[num_features].to_numpy(dtype=np.float32)
    return np.clip(
        np.column_stack([m.predict(X_n) for m in models]), 0.0, 5.0
    ).astype(np.float32)


# ─── Hauptpipeline ───────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 62)
    print("  Natural Disaster Severity Prediction  -  run_v3.py")
    mode_label = "QUICK (~15 min)" if QUICK_MODE else "FULL (~60 min)"
    print(f"  Mode: {mode_label}  |  stride={WINDOW_STRIDE}  estimators={N_ESTIMATORS}")
    print("=" * 62)

    # 1. Daten laden
    print("\n[1/6] Lade CSV-Dateien ...")
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw = pd.read_csv(TEST_CSV, dtype=dtypes)
    _parse_dates_inplace(train_raw)
    _parse_dates_inplace(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    print(f"   Train: {len(train_raw):>10,} Zeilen  |  Test: {len(test_raw):>8,} Zeilen")
    print(f"   Regionen: {len(regions)}  |  Features: {len(NUM_FEATURES)}")
    print(f"   Zeit: {time.time()-t0:.1f}s")

    # 2. Feature Engineering pro Region
    print("\n[2/6] Feature Engineering (pro Region) ...")
    all_tr_feat, all_te_feat = [], []
    n = len(regions)
    train_by_region = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    test_by_region  = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw
    for i, region in enumerate(regions, 1):
        if i % 500 == 0 or i == n:
            print(f"   Region {i}/{n}  |  {time.time()-t0:.1f}s")
        tr = train_by_region[region]
        te = test_by_region.get(region, pd.DataFrame())
        tr_f, te_f = compute_region_features(tr, te)
        all_tr_feat.append(tr_f)
        all_te_feat.append(te_f)

    train_feat = pd.concat(all_tr_feat, ignore_index=True)
    test_feat = pd.concat(all_te_feat, ignore_index=True)
    del all_tr_feat, all_te_feat
    print(f"   Fertig  |  {time.time()-t0:.1f}s")

    # 3. Woechentliche Aggregation
    print("\n[3/6] Woechentliche Aggregation ...")
    labeled = train_feat[train_feat["score"].notna()].copy()
    weekly_parts = []
    for region, g in labeled.groupby("region_id", sort=False):
        weekly_parts.append(daily_to_weekly(g))
    train_weekly = pd.concat(weekly_parts, ignore_index=True)
    del labeled
    weeks_per_region = int(len(train_weekly) / len(regions))
    print(f"   {len(train_weekly):,} Wochen-Zeilen  (~{weeks_per_region}/Region)")

    # 4. Train/Val-Split
    print("\n[4/6] Train/Validierung aufbauen ...")
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg = sorted(train_weekly["region_id"].unique())
    n_val = max(1, int(len(all_reg) * VAL_REGION_FRAC))
    val_regions = set(rng.choice(all_reg, size=n_val, replace=False))

    X_tr, y_tr = build_sliding_windows(train_weekly, val_regions, NUM_FEATURES, stride=WINDOW_STRIDE)
    X_va, y_va = build_val_samples(train_weekly, sorted(val_regions), NUM_FEATURES)
    print(f"   Train-Fenster: {len(X_tr):,}  |  Val-Regionen: {len(val_regions)}")

    # Persistence-Baseline
    last_score = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    persist_va = np.column_stack([
        last_score.reindex(sorted(val_regions)).fillna(0).to_numpy() for _ in range(5)
    ])
    show_mae("Persistence-Baseline", y_va, persist_va)

    # 5. Modell-Training
    print("\n[5/6] Training LightGBM ...")
    lgb_models = train_lgb_models(X_tr, y_tr, X_va, y_va)
    lgb_val = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (Validierung)", y_va, lgb_val)

    print("\n       Training XGBoost ...")
    xgb_models = train_xgb_models(X_tr, y_tr, X_va, y_va, NUM_FEATURES)
    xgb_val = predict_xgb(xgb_models, X_va, NUM_FEATURES)
    show_mae("XGBoost (Validierung)", y_va, xgb_val)

    print("\n  Blend-Optimierung:")
    best_alpha, best_mae_val = 0.5, 999.0
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = alpha * lgb_val + (1 - alpha) * xgb_val
        m_val = mae(y_va, blend)
        marker = "  <- best" if m_val < best_mae_val else ""
        print(f"   alpha={alpha:.1f} LGB + {1-alpha:.1f} XGB   MAE = {m_val:.4f}{marker}")
        if m_val < best_mae_val:
            best_mae_val, best_alpha = m_val, alpha

    show_mae(f"Ensemble (alpha={best_alpha} LGB, {1-best_alpha:.1f} XGB)",
             y_va, best_alpha * lgb_val + (1 - best_alpha) * xgb_val)

    # Finales Training auf allen Daten
    print("\n  Finales Training (alle Regionen) ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE)
    n_lgb_trees = [int(getattr(m, "best_iteration_", None) or LGB_PARAMS["n_estimators"])
                   for m in lgb_models]
    n_xgb_trees = [int(getattr(m, "best_iteration", None) or XGB_PARAMS["n_estimators"])
                   for m in xgb_models]

    final_lgb = train_lgb_models(X_all, y_all, None, None, n_lgb_trees)
    final_xgb = train_xgb_models(X_all, y_all, None, None, NUM_FEATURES, n_xgb_trees)
    print(f"   Fertig  |  {time.time()-t0:.1f}s")

    # 6. Test-Vorhersagen und Submission
    print("\n[6/6] Test-Vorhersagen ...")
    X_test = (
        test_feat.sort_values(["region_id", "ordinal"])
        .groupby("region_id", sort=False)
        .tail(1)[["region_id"] + NUM_FEATURES]
        .reset_index(drop=True)
    )
    X_test["region_id"] = X_test["region_id"].astype("category")

    lgb_test = predict_lgb(final_lgb, X_test)
    xgb_test = predict_xgb(final_xgb, X_test, NUM_FEATURES)
    test_preds = best_alpha * lgb_test + (1 - best_alpha) * xgb_test

    sub = pd.DataFrame({"region_id": X_test["region_id"].values})
    for k in range(5):
        sub[f"pred_week{k+1}"] = test_preds[:, k]

    template = pd.read_csv(SAMPLE_SUB)
    sub = template[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col] = sub[col].fillna(0.0)

    sub.to_csv(OUT_PATH, index=False)

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*62}")
    print(f"  Gespeichert: {OUT_PATH}")
    print(f"  Zeilen: {len(sub):,}  |  Gesamtzeit: {total_min:.1f} Min.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
