"""
precompute.py  –  Lokal einmal ausführen, dann alles auf Kaggle hochladen.

Was dieses Script macht:
  1. Feature Engineering (langsam, ~15 Min.)
  2. Weekly Aggregation + Sliding Windows (~10 Min.)
  3. LGB trainieren → Feature Importances extrahieren
  4. Feature Selection: behalte nur die wichtigsten Features (Standard: Top 80)
  5. Alles als NPZ speichern → auf Kaggle hochladen

Was auf Kaggle dann nur noch passiert:
  → kaggle_train.py laden + trainieren in ~5-10 Min. (ohne GPU) oder ~3 Min. (mit GPU)

Gespeicherte Dateien (data/precomputed/):
  features_full.npz      – alle 135 Features: X_train, X_val, X_all, X_test
  features_reduced.npz   – Top-K Features nach LGB Importance
  targets.npz            – y_train, y_val, y_all
  sequences.npz          – DL Wettersequenzen (26w × 14) für train/val/test
  meta.npz               – Feature-Namen, Val-Regionen, Normierung, Importances

Usage:
    python scripts/precompute.py
    → dann data/precomputed/ als Kaggle Dataset hochladen
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = DATA_DIR / "precomputed"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0
WINDOW_STRIDE   = 1
SEQ_LEN         = 26     # Wochen Rückblick für Transformer
TOP_K_FEATURES  = 80     # Behalte nur die wichtigsten K Features (0 = alle)

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp", "humidity"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]

LGB_PARAMS = dict(
    objective="regression", metric="mae",
    n_estimators=500, learning_rate=0.04, num_leaves=127,
    min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1,
)


def elapsed(t0):
    s = time.time() - t0
    return f"{s/60:.1f} Min." if s >= 60 else f"{s:.0f}s"


def _mb(path):
    return f"{path.stat().st_size/1e6:.1f} MB"


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(name: str, **arrays) -> None:
    """Save intermediate result to avoid recomputing on crash."""
    path = OUT_DIR / f"_checkpoint_{name}.npz"
    np.savez_compressed(path, **arrays)
    print(f"   checkpoint saved: {path.name}  ({_mb(path)})")


def load_checkpoint(name: str) -> dict | None:
    """Load checkpoint if it exists."""
    path = OUT_DIR / f"_checkpoint_{name}.npz"
    if path.exists():
        print(f"   checkpoint found: {path.name} — skipping recompute")
        return dict(np.load(path, allow_pickle=True))
    return None


# ─── Feature engineering ──────────────────────────────────────────────────────

def build_feature_list() -> list[str]:
    lags  = [f"{c}_lag{l}"      for c in LAG_COLS  for l in LAGS]
    rolls = [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS for s in ("mean","std","max")]
    cal   = ["month_sin","month_cos","day_sin","day_cos","week_sin","week_cos"]
    drt   = ["prec_deficit_90d","prec_trend_30d","humidity_deficit_90d",
             "tmp_anomaly_90d","heat_drought_idx","dry_days_14d","dry_days_30d"]
    return WEATHER_COLS + lags + rolls + cal + drt + ["regional_mean_score"]


def _parse_dates(df):
    p = df["date"].str.split("-", expand=True)
    df["year"]    = p[0].astype(np.int32)
    df["month"]   = p[1].astype(np.int32)
    df["day"]     = p[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]


def region_features(tr, te):
    te = te.copy(); te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc = {}
    nc["month_sin"] = np.sin(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["month_cos"] = np.cos(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["day_sin"]   = np.sin(2*np.pi*panel["day"]/31).astype(np.float32)
    nc["day_cos"]   = np.cos(2*np.pi*panel["day"]/31).astype(np.float32)
    woy = (panel["ordinal"]//7)%52
    nc["week_sin"]  = np.sin(2*np.pi*woy/52).astype(np.float32)
    nc["week_cos"]  = np.cos(2*np.pi*woy/52).astype(np.float32)
    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS:
            nc[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=max(3, w//10))
            nc[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            nc[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            nc[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)
    pp = panel["prec"].shift(1)
    nc["prec_deficit_90d"] = (pp.rolling(90,min_periods=30).mean()-pp.rolling(365,min_periods=60).mean()).astype(np.float32)
    p7=pp.rolling(7,min_periods=3).mean(); p30=pp.rolling(30,min_periods=10).mean()
    nc["prec_trend_30d"] = ((p7-p30)/pp.rolling(30,min_periods=10).std().clip(lower=0.01)).astype(np.float32)
    hp=panel["humidity"].shift(1)
    nc["humidity_deficit_90d"]=(hp.rolling(90,min_periods=30).mean()-hp.rolling(365,min_periods=60).mean()).astype(np.float32)
    tp=panel["tmp"].shift(1)
    anom=(tp.rolling(90,min_periods=30).mean()-tp.rolling(365,min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]=anom
    nc["heat_drought_idx"]=(nc["prec_deficit_90d"]*anom.clip(lower=0)).astype(np.float32)
    dry=(panel["prec"].shift(1)<DRY_THRESHOLD).astype(np.float32)
    nc["dry_days_14d"]=dry.rolling(14,min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"]=dry.rolling(30,min_periods=7).sum().astype(np.float32)
    panel = pd.concat([panel, pd.DataFrame(nc, index=panel.index)], axis=1)
    n = len(tr)
    return panel.iloc[:n].copy(), panel.iloc[n:].copy()


# ─── Dataset assembly ─────────────────────────────────────────────────────────

def daily_to_weekly(df):
    wk = df["ordinal"] // WEEK_BUCKET
    return df.loc[df.groupby(wk, sort=False)["ordinal"].idxmax()].reset_index(drop=True)


def build_sliding_windows(weekly, skip_regions, num_features, stride=1):
    Xp, yp, rp = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions: continue
        g = g.sort_values("ordinal")
        sc = g["score"].to_numpy(np.float32)
        Xn = g[num_features].to_numpy(np.float32)
        n  = len(g)
        if n < 6: continue
        nw = n - 5
        yr = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        idx = list(range(0, nw, stride))
        if (nw-1) not in idx: idx.append(nw-1)
        Xp.append(Xn[idx]); yp.append(yr[idx]); rp.extend([region]*len(idx))
    return (np.vstack(Xp).astype(np.float32),
            np.vstack(yp).astype(np.float32),
            np.array(rp))


def build_val_samples(weekly, val_regions, num_features):
    Xp, yp, rp = [], [], []
    for region in val_regions:
        g = weekly.loc[weekly["region_id"]==region].sort_values("ordinal")
        if len(g) < 6: continue
        Xp.append(g.iloc[-6][num_features].to_numpy(np.float32))
        yp.append(g.iloc[-5:]["score"].to_numpy(np.float32))
        rp.append(region)
    return (np.vstack(Xp).astype(np.float32),
            np.vstack(yp).astype(np.float32),
            np.array(rp))


# ─── DL sequence builder ──────────────────────────────────────────────────────

def build_dl_sequences(weekly, skip_regions, num_features,
                       w_mean, w_std, f_mean, f_std,
                       stride=1, max_samples=None, last_only=False):
    all_seq, all_feat, all_y = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if skip_regions and region in skip_regions: continue
        g = g.sort_values("ordinal").reset_index(drop=True)
        n = len(g)
        if n < 6: continue
        w_norm = ((g[WEATHER_COLS].fillna(0).values.astype(np.float32) - w_mean) / w_std)
        padded = np.concatenate([np.zeros((SEQ_LEN-1, len(WEATHER_COLS)), np.float32), w_norm])
        f_norm = ((g[num_features].fillna(0).values.astype(np.float32) - f_mean) / f_std)
        scores = g["score"].values.astype(np.float32)
        n_win  = n - 5
        indices = [n_win-1] if last_only else list(range(0, n_win, stride))
        if not last_only and (n_win-1) not in indices: indices.append(n_win-1)
        for i in indices:
            tgt = scores[i+1:i+6]
            if np.any(np.isnan(tgt)): continue
            all_seq.append(padded[i:i+SEQ_LEN])
            all_feat.append(f_norm[i])
            all_y.append(tgt)
    seqs  = np.stack(all_seq).astype(np.float32)
    feats = np.stack(all_feat).astype(np.float32)
    ys    = np.stack(all_y).astype(np.float32)
    if max_samples and not last_only and len(seqs) > max_samples:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(seqs), max_samples, replace=False)
        seqs, feats, ys = seqs[idx], feats[idx], ys[idx]
    return seqs, feats, ys


# ─── Feature importance + selection ───────────────────────────────────────────

def get_feature_importances(X_tr, y_tr, X_va, y_va, r_tr, num_features):
    """Train quick LGB (week 1 proxy) and return feature importances."""
    print("   Training LGB for feature importances ...")
    X_df = pd.DataFrame(X_tr, columns=num_features)
    X_df["region_id"] = pd.Categorical(r_tr)
    X_va_df = pd.DataFrame(X_va, columns=num_features)
    X_va_df["region_id"] = pd.Categorical(r_tr[:len(X_va)])  # dummy, just for shape

    # Use val set properly
    X_va_full = pd.DataFrame(X_va, columns=num_features)
    X_va_full["region_id"] = pd.Categorical(
        np.random.choice(np.unique(r_tr), size=len(X_va))
    )

    m = lgb.LGBMRegressor(**dict(LGB_PARAMS, random_state=RANDOM_STATE))
    m.fit(
        X_df, y_tr[:, 0].ravel(),
        eval_set=[(X_va_full, y_va[:, 0].ravel())],
        eval_metric="mae",
        categorical_feature=["region_id"],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    imp = pd.Series(m.feature_importances_, index=num_features + ["region_id"])
    imp = imp.drop("region_id", errors="ignore").sort_values(ascending=False)
    return imp


def select_features(importances: pd.Series, top_k: int) -> list[str]:
    """Return top-k feature names by importance."""
    if top_k <= 0 or top_k >= len(importances):
        return list(importances.index)
    selected = list(importances.head(top_k).index)
    print(f"   Feature selection: {len(importances)} → {len(selected)} features")
    print(f"   Dropped: {importances.tail(len(importances)-top_k).index.tolist()[:10]}...")
    return selected


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    NUM_FEATURES = build_feature_list()
    print("=" * 64)
    print("  Precompute: Feature Engineering + Sliding Windows + Selection")
    print(f"  Output: {OUT_DIR}")
    print(f"  Features: {len(NUM_FEATURES)}  →  Top-{TOP_K_FEATURES} after selection")
    print("=" * 64)

    # ── Step 1: Feature engineering (with checkpoint) ─────────────────────────
    ck = load_checkpoint("weekly")
    if ck is not None:
        # Reload weekly data from checkpoint
        # We need to reconstruct train_weekly from saved arrays
        print("   Skipping feature engineering — loading from checkpoint")
        weekly_feats  = ck["weekly_feats"]
        weekly_scores = ck["weekly_scores"]
        weekly_region = ck["weekly_region"]
        weekly_ordinal= ck["weekly_ordinal"]
        X_test_arr    = ck["X_test"]
        test_region_ids = ck["test_region_ids"]
        feature_names   = list(ck["feature_names"])
        NUM_FEATURES    = feature_names
        # Rebuild train_weekly DataFrame
        train_weekly = pd.DataFrame(weekly_feats, columns=NUM_FEATURES)
        train_weekly["score"]     = weekly_scores
        train_weekly["region_id"] = weekly_region
        train_weekly["ordinal"]   = weekly_ordinal
    else:
        print("\n[1/6] Loading CSV ...")
        dtypes    = {c: np.float32 for c in WEATHER_COLS}
        train_raw = pd.read_csv(DATA_DIR / "train.csv", dtype=dtypes)
        test_raw  = pd.read_csv(DATA_DIR / "test.csv",  dtype=dtypes)
        _parse_dates(train_raw); _parse_dates(test_raw)
        train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
        regions = train_raw["region_id"].unique()
        print(f"   Train={len(train_raw):,}  Test={len(test_raw):,}  [{elapsed(t0)}]")
        region_means = train_raw.groupby("region_id")["score"].mean()

        print("\n[2/6] Feature engineering per region ...")
        tr_by_r = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
        te_by_r = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
        del train_raw, test_raw
        all_tr, all_te = [], []
        for i, region in enumerate(regions, 1):
            if i % 500 == 0 or i == len(regions):
                print(f"   Region {i}/{len(regions)}  [{elapsed(t0)}]")
            tf, ef = region_features(tr_by_r[region], te_by_r.get(region, pd.DataFrame()))
            all_tr.append(tf); all_te.append(ef)
        train_feat = pd.concat(all_tr, ignore_index=True)
        test_feat  = pd.concat(all_te, ignore_index=True)
        del all_tr, all_te
        train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
        test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)

        print("\n[3/6] Weekly aggregation ...")
        labeled = train_feat[train_feat["score"].notna()].copy()
        wk_parts = [daily_to_weekly(g) for _, g in labeled.groupby("region_id", sort=False)]
        train_weekly = pd.concat(wk_parts, ignore_index=True)
        del labeled
        print(f"   {len(train_weekly):,} weekly rows  [{elapsed(t0)}]")

        # Test feature vectors
        X_test_df = (
            test_feat.sort_values(["region_id","ordinal"])
            .groupby("region_id", sort=False).tail(1)
            [["region_id"] + NUM_FEATURES].reset_index(drop=True)
        )
        test_region_ids = X_test_df["region_id"].values
        X_test_arr = X_test_df[NUM_FEATURES].to_numpy(np.float32)

        # Save checkpoint
        save_checkpoint("weekly",
            weekly_feats  = train_weekly[NUM_FEATURES].to_numpy(np.float32),
            weekly_scores = train_weekly["score"].to_numpy(np.float32),
            weekly_region = train_weekly["region_id"].values.astype(str),
            weekly_ordinal= train_weekly["ordinal"].to_numpy(np.int32),
            X_test        = X_test_arr,
            test_region_ids = test_region_ids.astype(str),
            feature_names = np.array(NUM_FEATURES, dtype=object),
        )

    # ── Step 2: Sliding windows (with checkpoint) ─────────────────────────────
    ck2 = load_checkpoint("windows")
    if ck2 is not None:
        X_tr    = ck2["X_tr"];    y_tr = ck2["y_tr"];    r_tr = ck2["r_tr"]
        X_va    = ck2["X_va"];    y_va = ck2["y_va"];    r_va = ck2["r_va"]
        X_all   = ck2["X_all"];   y_all= ck2["y_all"];   r_all= ck2["r_all"]
        val_regions = set(ck2["val_regions"].tolist())
        all_reg = list(ck2["all_reg"])
    else:
        print("\n[4/6] Building sliding windows ...")
        rng = np.random.default_rng(RANDOM_STATE)
        all_reg = sorted(train_weekly["region_id"].unique())
        val_regions = set(rng.choice(all_reg, max(1, int(len(all_reg)*VAL_REGION_FRAC)), replace=False))

        X_tr, y_tr, r_tr = build_sliding_windows(train_weekly, val_regions, NUM_FEATURES, WINDOW_STRIDE)
        X_va, y_va, r_va = build_val_samples(train_weekly, sorted(val_regions), NUM_FEATURES)
        X_all, y_all, r_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, WINDOW_STRIDE)
        print(f"   X_tr={X_tr.shape}  X_va={X_va.shape}  X_all={X_all.shape}  [{elapsed(t0)}]")

        save_checkpoint("windows",
            X_tr=X_tr, y_tr=y_tr, r_tr=r_tr.astype(str),
            X_va=X_va, y_va=y_va, r_va=r_va.astype(str),
            X_all=X_all, y_all=y_all, r_all=r_all.astype(str),
            val_regions=np.array(sorted(val_regions), dtype=object),
            all_reg=np.array(all_reg, dtype=object),
        )

    # ── Step 3: Feature importances + selection ────────────────────────────────
    print("\n[5/6] Feature importance + selection ...")
    importances = get_feature_importances(X_tr, y_tr, X_va, y_va, r_tr, NUM_FEATURES)
    selected_features = select_features(importances, TOP_K_FEATURES)

    # Indices of selected features
    feat_idx = [NUM_FEATURES.index(f) for f in selected_features]

    # Reduce all arrays to selected features
    X_tr_sel  = X_tr[:, feat_idx]
    X_va_sel  = X_va[:, feat_idx]
    X_all_sel = X_all[:, feat_idx]
    X_test_sel = X_test_arr[:, feat_idx]

    # ── Step 4: DL sequences ──────────────────────────────────────────────────
    print("\n[6/6] Building DL sequences ...")
    w_mean = train_weekly[WEATHER_COLS].mean().values.astype(np.float32)
    w_std  = train_weekly[WEATHER_COLS].std().clip(lower=1e-8).values.astype(np.float32)
    f_mean = X_tr_sel.mean(axis=0).astype(np.float32)
    f_std  = np.where(X_tr_sel.std(axis=0) < 1e-8, 1.0, X_tr_sel.std(axis=0)).astype(np.float32)

    print("   Training sequences ...")
    tr_seqs, tr_feats, tr_ys = build_dl_sequences(
        train_weekly, val_regions, selected_features,
        w_mean, w_std, f_mean, f_std,
        stride=WINDOW_STRIDE, max_samples=350_000,
    )
    print("   Val sequences ...")
    va_seqs, va_feats, va_ys = build_dl_sequences(
        train_weekly, set(all_reg)-val_regions, selected_features,
        w_mean, w_std, f_mean, f_std,
        last_only=True,
    )
    print("   Test sequences ...")
    test_seqs_list, test_feats_list = [], []
    for region in test_region_ids:
        g = train_weekly.loc[train_weekly["region_id"]==region].sort_values("ordinal")
        n_g = len(g)
        i = n_g - 1
        wn = ((g[WEATHER_COLS].fillna(0).values.astype(np.float32) - w_mean) / w_std)
        fn = ((g[selected_features].fillna(0).values.astype(np.float32) - f_mean) / f_std)
        pad = np.concatenate([np.zeros((SEQ_LEN-1, len(WEATHER_COLS)), np.float32), wn])
        test_seqs_list.append(pad[i:i+SEQ_LEN])
        test_feats_list.append(fn[i])
    te_seqs  = np.stack(test_seqs_list).astype(np.float32)
    te_feats = np.stack(test_feats_list).astype(np.float32)
    # Tabular test features: from test_feat (has test-period rolling/lags)
    te_feats_tab = ((X_test_sel - f_mean) / f_std).astype(np.float32)
    print(f"   tr_seqs={tr_seqs.shape}  va_seqs={va_seqs.shape}  te_seqs={te_seqs.shape}  [{elapsed(t0)}]")

    # ── Save final output files ────────────────────────────────────────────────
    print("\n   Saving final files ...")

    # Full features (for LGB — handles all 135 natively)
    np.savez_compressed(OUT_DIR / "features_full.npz",
        X_tr=X_tr, r_tr=r_tr.astype(str),
        X_va=X_va, r_va=r_va.astype(str),
        X_all=X_all, r_all=r_all.astype(str),
        X_test=X_test_arr, test_region_ids=test_region_ids.astype(str),
    )
    print(f"   features_full.npz  {_mb(OUT_DIR/'features_full.npz')}")

    # Reduced features (for DL MLP branch — cleaner, faster)
    np.savez_compressed(OUT_DIR / "features_reduced.npz",
        X_tr=X_tr_sel, X_va=X_va_sel,
        X_all=X_all_sel, X_test=X_test_sel,
    )
    print(f"   features_reduced.npz  {_mb(OUT_DIR/'features_reduced.npz')}")

    # Targets
    np.savez_compressed(OUT_DIR / "targets.npz",
        y_tr=y_tr, y_va=y_va, y_all=y_all,
    )

    # DL sequences
    np.savez_compressed(OUT_DIR / "sequences.npz",
        tr_seqs=tr_seqs, tr_feats=tr_feats, tr_ys=tr_ys,
        va_seqs=va_seqs, va_feats=va_feats, va_ys=va_ys,
        te_seqs=te_seqs, te_feats=te_feats, te_feats_tab=te_feats_tab,
    )
    print(f"   sequences.npz  {_mb(OUT_DIR/'sequences.npz')}")

    # Metadata: feature names, val regions, normalization, importances
    np.savez(OUT_DIR / "meta.npz",
        feature_names_full    = np.array(NUM_FEATURES, dtype=object),
        feature_names_reduced = np.array(selected_features, dtype=object),
        feature_importances   = importances.values.astype(np.float32),
        importance_names      = np.array(importances.index.tolist(), dtype=object),
        val_regions  = np.array(sorted(val_regions), dtype=object),
        all_regions  = np.array(all_reg, dtype=object),
        test_region_ids = test_region_ids.astype(str),
        weather_mean = w_mean, weather_std = w_std,
        feat_mean    = f_mean, feat_std    = f_std,
        y_va         = y_va,
    )
    print(f"   meta.npz saved")

    # Summary
    files = list(OUT_DIR.glob("*.npz")) + list(OUT_DIR.glob("*.npy"))
    total_mb = sum(f.stat().st_size for f in files if not f.name.startswith("_")) / 1e6
    print(f"\n{'='*64}")
    print(f"  Done in {elapsed(t0)}")
    print(f"  Upload folder to Kaggle: {OUT_DIR}")
    print(f"  Total upload size: {total_mb:.0f} MB")
    print(f"  Selected features: {len(selected_features)} / {len(NUM_FEATURES)}")
    print(f"  Top 10: {selected_features[:10]}")
    print(f"\n  Next: upload data/precomputed/ to Kaggle, then run kaggle_train.py")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
