"""
run_v16_chain.py  –  Drought Severity Prediction v16: Chain Forecasting

Problem mit v15 (5 unabhängige Modelle):
  Modell für Woche 4 weiß nicht ob Woche 3 eine Dürre vorhersagt.
  Alle 5 Modelle sehen nur die originalen Features, nie die Vorhersagen der anderen.

Lösung: Autoregressive Chain
  Modell W1: X                                    → pred_w1
  Modell W2: X + pred_w1                          → pred_w2
  Modell W3: X + pred_w1 + pred_w2                → pred_w3
  Modell W4: X + pred_w1 + pred_w2 + pred_w3      → pred_w4
  Modell W5: X + pred_w1 + ... + pred_w4           → pred_w5

  Training:  echte Scores als Chain-Features (Teacher Forcing)
  Inference: eigene Vorhersagen weitergegeben

Warum das besser ist:
  - Dürre ist hochpersistent (Autokorr. 0.966). Wenn Woche 3 Score=4 vorhersagt,
    ist das das beste Signal für Woche 4 und 5.
  - Die 5 unabhängigen Modelle können das nicht nutzen.

Features: identisch zu v15 (Scout-bereinigt, score_lag1/2/3, seasonal dev)
Checkpoints: Weekly-Cache + Windows-Cache (geteilt mit v15 wenn gleiche Features)

Usage:
    python scripts/run_v16_chain.py
Output: outputs/submission_v16_chain.csv
"""
from __future__ import annotations
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

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
OUT_PATH      = OUT_DIR / "submission_v16_chain.csv"
WEEKLY_CACHE  = CACHE_DIR / "_checkpoint_weekly.npz"
WINDOWS_CACHE = CACHE_DIR / "_checkpoint_v15_windows.npz"  # gleiche Features wie v15

# ─── Knobs ────────────────────────────────────────────────────────────────────
QUICK_MODE      = False
RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0
RECENT_YEARS    = None   # None = alle 13 Jahre; 5 = letzte 5 Jahre pro Region

WINDOW_STRIDE = 4 if QUICK_MODE else 1
N_ESTIMATORS  = 500 if QUICK_MODE else 1000

# ─── Feature-Konfiguration (identisch zu v15) ─────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS   = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS       = [1, 3, 7, 14, 21]
ROLL_COLS  = ["prec", "humidity", "tmp"]
ROLL_WINS  = [7, 14, 30, 60, 90, 180]
ROLL_STATS = ["mean", "std"]

BASE_FEATURES: list[str] = []   # Features ohne Chain-Spalten
LGB_PARAMS = dict(
    objective="regression", metric="mae", n_estimators=N_ESTIMATORS,
    learning_rate=0.04, num_leaves=127, min_child_samples=60,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, verbose=-1,
)


def build_base_features() -> list[str]:
    """Feature-Liste OHNE Chain-Spalten (pred_w1 usw.)."""
    f = list(WEATHER_COLS)
    f += [f"{c}_lag{l}" for c in LAG_COLS for l in LAGS]
    f += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS for s in ROLL_STATS]
    f += ["month_sin", "month_cos", "day_sin", "day_cos"]
    f += ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
          "tmp_anomaly_90d", "heat_drought_idx"]
    f += ["regional_mean_score"]
    f += ["score_lag1", "score_lag2", "score_lag3"]
    f += ["prec_seasonal_dev", "humidity_seasonal_dev", "tmp_seasonal_dev"]
    return f


def chain_features(week: int) -> list[str]:
    """Features die Modell für Woche `week` (0-basiert) zusätzlich bekommt."""
    return [f"pred_w{k+1}" for k in range(week)]  # week=0 → [], week=4 → [pred_w1..w4]


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────
def elapsed(t0): s=time.time()-t0; return f"{s/60:.1f} Min." if s>=60 else f"{s:.0f}s"
def mae(y, p):   return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))
def show_mae(n, y, p): print(f"  {n:<52s}  MAE = {mae(y, p):.4f}")

def _parse_dates(df):
    p = df["date"].str.split("-", expand=True)
    df["year"]=p[0].astype(np.int32); df["month"]=p[1].astype(np.int32)
    df["day"]=p[2].astype(np.int32)
    df["ordinal"]=df["year"]*372+df["month"]*31+df["day"]

def _best_n(m, default):
    for a in ("best_iteration_","best_iteration"):
        v=getattr(m,a,None)
        if v is not None: return int(v)
    try: return int(m.get_best_iteration())
    except: return default


# ─── Feature Engineering (nur ohne Weekly-Cache) ──────────────────────────────
def _region_features(tr, te):
    te=te.copy(); te["score"]=np.nan
    panel=pd.concat([tr,te],ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc={}
    nc["month_sin"]=np.sin(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["month_cos"]=np.cos(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["day_sin"]  =np.sin(2*np.pi*panel["day"]/31).astype(np.float32)
    nc["day_cos"]  =np.cos(2*np.pi*panel["day"]/31).astype(np.float32)
    woy=(panel["ordinal"]//7)%52
    nc["week_sin"]=np.sin(2*np.pi*woy/52).astype(np.float32)
    nc["week_cos"]=np.cos(2*np.pi*woy/52).astype(np.float32)
    for col in LAG_COLS:
        s=panel[col]
        for lag in LAGS: nc[f"{col}_lag{lag}"]=s.shift(lag).astype(np.float32)
    for col in ["prec","humidity","tmp","wind"]:
        prior=panel[col].shift(1)
        for w in ROLL_WINS:
            r=prior.rolling(w,min_periods=max(3,w//10))
            nc[f"{col}_roll{w}_mean"]=r.mean().astype(np.float32)
            nc[f"{col}_roll{w}_std"] =r.std().astype(np.float32)
            nc[f"{col}_roll{w}_max"] =r.max().astype(np.float32)
    pp=panel["prec"].shift(1)
    nc["prec_deficit_90d"]=(pp.rolling(90,min_periods=30).mean()-pp.rolling(365,min_periods=60).mean()).astype(np.float32)
    p7=pp.rolling(7,min_periods=3).mean(); p30=pp.rolling(30,min_periods=10).mean()
    nc["prec_trend_30d"]=((p7-p30)/pp.rolling(30,min_periods=10).std().clip(lower=0.01)).astype(np.float32)
    hp=panel["humidity"].shift(1)
    nc["humidity_deficit_90d"]=(hp.rolling(90,min_periods=30).mean()-hp.rolling(365,min_periods=60).mean()).astype(np.float32)
    tp=panel["tmp"].shift(1); anom=(tp.rolling(90,min_periods=30).mean()-tp.rolling(365,min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]=anom; nc["heat_drought_idx"]=(nc["prec_deficit_90d"]*anom.clip(lower=0)).astype(np.float32)
    dry=(panel["prec"].shift(1)<DRY_THRESHOLD).astype(np.float32)
    nc["dry_days_14d"]=dry.rolling(14,min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"]=dry.rolling(30,min_periods=7).sum().astype(np.float32)
    panel=pd.concat([panel,pd.DataFrame(nc,index=panel.index)],axis=1)
    n=len(tr)
    return panel.iloc[:n].copy(), panel.iloc[n:].copy()

def _daily_to_weekly(df):
    wk=df["ordinal"]//WEEK_BUCKET
    return df.loc[df.groupby(wk,sort=False)["ordinal"].idxmax()].reset_index(drop=True)

def add_v15_features(weekly):
    weekly=weekly.sort_values(["region_id","ordinal"]).copy()
    g=weekly.groupby("region_id")["score"]
    weekly["score_lag1"]=g.transform(lambda x: x).astype(np.float32)
    weekly["score_lag2"]=g.shift(1).astype(np.float32)
    weekly["score_lag3"]=g.shift(2).astype(np.float32)
    weekly["score_lag2"].fillna(weekly["score_lag1"],inplace=True)
    weekly["score_lag3"].fillna(weekly["score_lag2"],inplace=True)
    weekly["_month"]=((weekly["ordinal"]%372)//31).clip(0,11)
    for col in ["prec","humidity","tmp"]:
        if col in weekly.columns:
            norm=weekly.groupby(["region_id","_month"])[col].transform("mean")
            weekly[f"{col}_seasonal_dev"]=(weekly[col]-norm).astype(np.float32)
    weekly.drop(columns=["_month"],inplace=True)
    return weekly


# ─── Weekly-Cache ─────────────────────────────────────────────────────────────
def load_weekly():
    if WEEKLY_CACHE.exists():
        print(f"   Weekly-Cache: {WEEKLY_CACHE.name}  ({WEEKLY_CACHE.stat().st_size/1e6:.0f} MB)")
        ck=dict(np.load(WEEKLY_CACHE,allow_pickle=True))
        base=list(ck["feature_names"])
        weekly=pd.DataFrame(ck["weekly_feats"],columns=base)
        weekly["score"]    =ck["weekly_scores"].astype(np.float32)
        weekly["region_id"]=ck["weekly_region"].astype(str)
        weekly["ordinal"]  =ck["weekly_ordinal"].astype(np.int32)
        return weekly, ck["X_test"].astype(np.float32), ck["test_region_ids"].astype(str)
    print("   Kein Cache — Feature Engineering (~20 Min) ...")
    dtypes={c:np.float32 for c in WEATHER_COLS}
    train_raw=pd.read_csv(TRAIN_CSV,dtype=dtypes); test_raw=pd.read_csv(TEST_CSV,dtype=dtypes)
    _parse_dates(train_raw); _parse_dates(test_raw)
    train_raw["score"]=pd.to_numeric(train_raw["score"],errors="coerce").astype(np.float32)
    regions=train_raw["region_id"].unique()
    region_means=train_raw.groupby("region_id")["score"].mean()
    tr_by={r:g.reset_index(drop=True) for r,g in train_raw.groupby("region_id",sort=False)}
    te_by={r:g.reset_index(drop=True) for r,g in test_raw.groupby("region_id",sort=False)}
    del train_raw,test_raw
    all_tr,all_te=[],[]
    for i,region in enumerate(regions,1):
        tf,ef=_region_features(tr_by[region],te_by.get(region,pd.DataFrame()))
        all_tr.append(tf); all_te.append(ef)
        if i%500==0 or i==len(regions): print(f"   Region {i}/{len(regions)}")
    train_feat=pd.concat(all_tr,ignore_index=True); test_feat=pd.concat(all_te,ignore_index=True)
    del all_tr,all_te
    train_feat["regional_mean_score"]=train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"] =test_feat["region_id"].map(region_means).astype(np.float32)
    labeled=train_feat[train_feat["score"].notna()].copy()
    weekly=pd.concat([_daily_to_weekly(g) for _,g in labeled.groupby("region_id",sort=False)],ignore_index=True)
    del labeled
    base_cols=[c for c in weekly.columns if c not in ("score","region_id","ordinal","date","year","month","day")]
    X_test_df=(test_feat.sort_values(["region_id","ordinal"]).groupby("region_id",sort=False).tail(1)[["region_id"]+base_cols].reset_index(drop=True))
    test_region_ids=X_test_df["region_id"].values.astype(str)
    X_test_arr=X_test_df[base_cols].to_numpy(np.float32)
    np.savez_compressed(WEEKLY_CACHE,weekly_feats=weekly[base_cols].to_numpy(np.float32),
        weekly_scores=weekly["score"].to_numpy(np.float32),weekly_region=weekly["region_id"].values.astype(str),
        weekly_ordinal=weekly["ordinal"].to_numpy(np.int32),X_test=X_test_arr,
        test_region_ids=test_region_ids,feature_names=np.array(base_cols,dtype=object))
    print(f"   Weekly-Cache gespeichert"); return weekly, X_test_arr, test_region_ids


# ─── Windows-Cache ────────────────────────────────────────────────────────────
def _per_region_cutoff(g):
    if RECENT_YEARS is None: return g
    return g[g["ordinal"] >= int(g["ordinal"].max()) - int(RECENT_YEARS*372)]

def _build_windows(weekly, skip, features, stride=1):
    Xp,yp,rp=[],[],[]
    for region,g in weekly.groupby("region_id",sort=False):
        if region in skip: continue
        g=_per_region_cutoff(g.sort_values("ordinal"))
        sc=g["score"].to_numpy(np.float32); Xn=g[features].to_numpy(np.float32)
        n=len(g)
        if n<6: continue
        nw=n-5; yr=np.lib.stride_tricks.sliding_window_view(sc[1:],5)[:nw]
        idx=list(range(0,nw,stride))
        if (nw-1) not in idx: idx.append(nw-1)
        Xp.append(Xn[idx]); yp.append(yr[idx]); rp.extend([region]*len(idx))
    X=pd.DataFrame(np.vstack(Xp).astype(np.float32),columns=features)
    X["region_id"]=pd.Categorical(rp)
    return X, np.vstack(yp).astype(np.float32)

def _build_val(weekly, val_regions, features):
    Xp,yp,rp=[],[],[]
    for region in val_regions:
        g=weekly.loc[weekly["region_id"]==region].sort_values("ordinal")
        if len(g)<6: continue
        Xp.append(g.iloc[-6][features].to_numpy(np.float32))
        yp.append(g.iloc[-5:]["score"].to_numpy(np.float32)); rp.append(region)
    X=pd.DataFrame(np.vstack(Xp),columns=features); X["region_id"]=pd.Categorical(rp)
    return X, np.vstack(yp)

def load_or_build_windows(weekly, val_regions, features, t0):
    if WINDOWS_CACHE.exists():
        ck=dict(np.load(WINDOWS_CACHE,allow_pickle=True))
        if list(ck["feature_names"])==features and set(ck["val_regions"].astype(str).tolist())==val_regions:
            print(f"   Windows-Cache: {WINDOWS_CACHE.name}  ({WINDOWS_CACHE.stat().st_size/1e6:.0f} MB)")
            def _r(p):
                X=pd.DataFrame(ck[f"X_{p}"],columns=features); X["region_id"]=pd.Categorical(ck[f"r_{p}"].astype(str).tolist())
                return X,ck[f"y_{p}"]
            return *_r("tr"), *_r("va"), *_r("all")
        print("   Cache veraltet — Windows neu berechnen ...")
    print(f"   Berechne Windows ...  [{elapsed(t0)}]")
    X_tr,y_tr=_build_windows(weekly,val_regions,features,WINDOW_STRIDE)
    X_va,y_va=_build_val(weekly,sorted(val_regions),features)
    X_all,y_all=_build_windows(weekly,set(),features,WINDOW_STRIDE)
    np.savez_compressed(WINDOWS_CACHE,
        X_tr=X_tr[features].to_numpy(np.float32),y_tr=y_tr,r_tr=np.array(X_tr["region_id"].astype(str),dtype=object),
        X_va=X_va[features].to_numpy(np.float32),y_va=y_va,r_va=np.array(X_va["region_id"].astype(str),dtype=object),
        X_all=X_all[features].to_numpy(np.float32),y_all=y_all,r_all=np.array(X_all["region_id"].astype(str),dtype=object),
        val_regions=np.array(sorted(val_regions),dtype=object),feature_names=np.array(features,dtype=object))
    print(f"   Windows-Cache gespeichert  [{elapsed(t0)}]")
    return X_tr,y_tr,X_va,y_va,X_all,y_all


# ─── Chain-Training ───────────────────────────────────────────────────────────

def _add_chain_cols(X: pd.DataFrame, chain_vals: np.ndarray, week: int) -> pd.DataFrame:
    """Fügt pred_w1..pred_wN als Spalten zu X hinzu (N = week)."""
    X = X.copy()
    for k in range(week):
        X[f"pred_w{k+1}"] = chain_vals[:, k].astype(np.float32)
    return X


def train_chain_models(X_tr, y_tr, X_va, y_va) -> list:
    """
    Trainiert 5 LightGBM-Modelle in der Chain.

    Training (Teacher Forcing):
      Modell W1 sieht X
      Modell W2 sieht X + true_score_w1
      Modell W3 sieht X + true_score_w1 + true_score_w2
      ...
    So lernt jedes Modell, mit echten Vorwochen-Scores zu arbeiten.

    Bei Inference werden stattdessen eigene Vorhersagen weitergegeben.
    """
    models = []
    for week in range(5):
        cf = chain_features(week)  # z.B. ["pred_w1", "pred_w2"] für week=2
        features_w = BASE_FEATURES + cf + ["region_id"]

        # Training: Teacher Forcing — echte Labels als Chain-Features
        X_tr_w = _add_chain_cols(X_tr, y_tr, week)   # y_tr[:, 0..week-1] = echte Scores
        X_va_w = _add_chain_cols(X_va, y_va, week)

        p = dict(LGB_PARAMS, random_state=RANDOM_STATE + week)
        m = lgb.LGBMRegressor(**p)
        m.fit(
            X_tr_w[features_w], y_tr[:, week].ravel(),
            eval_set=[(X_va_w[features_w], y_va[:, week].ravel())],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=["region_id"],
        )
        val_pred = np.clip(m.predict(X_va_w[features_w]), 0, 5)
        val_mae  = mae(y_va[:, week], val_pred)
        n_iter   = _best_n(m, N_ESTIMATORS)
        print(f"    W{week+1}: best_iter={n_iter:4d}  val_MAE={val_mae:.4f}"
              f"  chain_feats={cf if cf else 'keine'}")
        models.append(m)
    return models


def predict_chain(models: list, X: pd.DataFrame) -> np.ndarray:
    """
    Autoregressive Inference: Vorhersagen werden als Chain-Features weitergegeben.

    W1 → pred_w1
    W2(X + pred_w1) → pred_w2
    W3(X + pred_w1 + pred_w2) → pred_w3
    ...
    """
    preds = np.zeros((len(X), 5), dtype=np.float32)
    for week, m in enumerate(models):
        cf = chain_features(week)
        features_w = BASE_FEATURES + cf + ["region_id"]
        X_w = _add_chain_cols(X, preds, week)  # preds[:, 0..week-1] = eigene Vorhersagen
        preds[:, week] = np.clip(m.predict(X_w[features_w]), 0, 5)
    return preds


def retrain_final(X_all, y_all, n_iters: list) -> list:
    """Finales Training auf allen Daten mit optimaler Baum-Anzahl."""
    models = []
    for week in range(5):
        cf = chain_features(week)
        features_w = BASE_FEATURES + cf + ["region_id"]
        X_w = _add_chain_cols(X_all, y_all, week)  # Teacher Forcing mit echten Labels
        p = dict(LGB_PARAMS, random_state=RANDOM_STATE + week, n_estimators=n_iters[week])
        m = lgb.LGBMRegressor(**p)
        m.fit(X_w[features_w], y_all[:, week].ravel(), categorical_feature=["region_id"])
        models.append(m)
    return models


# ─── Feature Importance ───────────────────────────────────────────────────────

def print_feature_importance(models: list) -> None:
    """Gibt Feature Importance für alle 5 Woche-Modelle aus."""
    print(f"\n{'─'*66}")
    print(f"  FEATURE IMPORTANCE  (LightGBM Gain, pro Woche + Gesamt-Ø)")
    print(f"{'─'*66}")

    all_names = np.array(models[-1].booster_.feature_name())  # W5 hat die meisten Features
    total_imp  = np.zeros(len(all_names))

    for week, m in enumerate(models):
        feat_names = np.array(m.booster_.feature_name())
        imp        = m.booster_.feature_importance(importance_type="gain").astype(float)
        mask       = feat_names != "region_id"
        feat_names = feat_names[mask]; imp = imp[mask]
        total = imp.sum(); order = np.argsort(imp)[::-1]
        print(f"\n  Woche {week+1}  (chain_feats: {chain_features(week) or 'keine'}):")
        print(f"  {'Feature':<36}  {'Gain':>10}  {'%':>6}")
        for i in order[:15]:
            print(f"  {feat_names[i]:<36}  {imp[i]:>10.0f}  {100*imp[i]/total:>5.2f}%")
        # Chain-Features separat hervorheben
        for cf in chain_features(week):
            idx = np.where(feat_names == cf)[0]
            if len(idx): print(f"  ↳ {cf:<34}  {imp[idx[0]]:>10.0f}  {100*imp[idx[0]]/total:>5.2f}%  ← Chain")
        # Für Gesamt-Ø: auf gemeinsame Basis-Features reduzieren
        for i, n in enumerate(feat_names):
            j = np.where(all_names == n)[0]
            if len(j): total_imp[j[0]] += imp[i] / 5

    # Gesamt-Ø über alle 5 Modelle (nur Base-Features)
    mask  = all_names != "region_id"
    anames= all_names[mask]; aimp = total_imp[mask]
    total = aimp.sum(); order = np.argsort(aimp)[::-1]
    print(f"\n{'─'*66}")
    print(f"  Gesamt-Ø Top-30 (base features, über alle 5 Wochen):")
    print(f"  {'Feature':<36}  {'Gain Ø':>10}  {'%':>6}")
    for rank, i in enumerate(order[:30], 1):
        if aimp[i] == 0: break
        print(f"  {rank:<3d}  {anames[i]:<34}  {aimp[i]:>10.0f}  {100*aimp[i]/total:>5.2f}%")
    print(f"\n  Top-10 kumulativ: {100*aimp[order[:10]].sum()/total:.1f}%")
    print(f"{'─'*66}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    global BASE_FEATURES
    BASE_FEATURES = build_base_features()

    t0 = time.time()
    print("=" * 66)
    print("  Drought Severity Prediction  —  run_v16_chain.py")
    print(f"  Mode: {'QUICK' if QUICK_MODE else 'FULL'}  |  stride={WINDOW_STRIDE}  trees={N_ESTIMATORS}")
    print(f"  Base Features: {len(BASE_FEATURES)}  |  RECENT_YEARS={RECENT_YEARS}")
    print(f"  Ansatz: Chain Forecasting — jede Woche kennt Vorhersagen der Vorwochen")
    print("=" * 66)

    # 1. Wöchentliche Daten
    print(f"\n[1/5] Wöchentliche Daten laden ...")
    weekly, X_test_base, test_region_ids = load_weekly()
    print(f"   {len(weekly):,} Rows  |  [{elapsed(t0)}]")

    # 2. V15/V16-Features berechnen
    print(f"\n[2/5] Features berechnen (Score-Lags, Seasonal Dev) ...")
    weekly = add_v15_features(weekly)
    last_score = weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    for f in BASE_FEATURES:
        if f not in weekly.columns: weekly[f] = np.float32(0)
    print(f"   Done  |  [{elapsed(t0)}]")

    # 3. Sliding Windows (Cache-kompatibel mit v15)
    print(f"\n[3/5] Sliding Windows ...")
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg     = sorted(weekly["region_id"].unique())
    val_regions = set(rng.choice(all_reg, max(1, int(len(all_reg)*VAL_REGION_FRAC)), replace=False))
    X_tr, y_tr, X_va, y_va, X_all, y_all = load_or_build_windows(
        weekly, val_regions, BASE_FEATURES, t0
    )
    print(f"   Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")

    # Baselines
    persist_va = np.column_stack([last_score.reindex(sorted(val_regions)).fillna(0).to_numpy()]*5)
    show_mae("Persistence-Baseline", y_va, persist_va)

    # 4. Chain Training
    print(f"\n[4/5] Chain Training  |  [{elapsed(t0)}]")
    print("  Jedes Modell bekommt Vorhersagen der Vorwochen als Features (Teacher Forcing)")
    chain_models = train_chain_models(X_tr, y_tr, X_va, y_va)

    # Val-MAE im Inference-Modus (eigene Vorhersagen als Chain — wie echtes Test-Szenario)
    val_preds_chain = predict_chain(chain_models, X_va)
    show_mae("Chain LightGBM (Inference-Modus, val)", y_va, val_preds_chain)

    # Zum Vergleich: Teacher Forcing val MAE (obere Schranke)
    val_preds_tf = np.column_stack([
        np.clip(chain_models[wk].predict(
            _add_chain_cols(X_va, y_va, wk)[BASE_FEATURES + chain_features(wk) + ["region_id"]]
        ), 0, 5)
        for wk in range(5)
    ])
    show_mae("Chain LightGBM (Teacher Forcing, val) ", y_va, val_preds_tf)

    # Feature Importance
    print_feature_importance(chain_models)

    if QUICK_MODE:
        print(f"\n  QUICK_MODE — kein Final-Training. Laufzeit: {elapsed(t0)}\n")
        return

    # 5. Final Training + Submission
    print(f"\n[5/5] Final Training (alle Regionen)  |  [{elapsed(t0)}]")
    n_iters = [_best_n(m, N_ESTIMATORS) for m in chain_models]
    final_models = retrain_final(X_all, y_all, n_iters)
    print(f"   Done  |  [{elapsed(t0)}]")

    # Test-Features
    cache_cols = list(dict(np.load(WEEKLY_CACHE, allow_pickle=True))["feature_names"])
    X_test = pd.DataFrame(X_test_base, columns=cache_cols)
    X_test["region_id"] = pd.Categorical(test_region_ids)
    X_test["score_lag1"] = X_test["region_id"].map(last_score).astype(np.float32).fillna(0)
    X_test["score_lag2"] = X_test["score_lag1"]
    X_test["score_lag3"] = X_test["score_lag1"]
    last_w = weekly.sort_values("ordinal").groupby("region_id").last()
    for col in ["prec_seasonal_dev", "humidity_seasonal_dev", "tmp_seasonal_dev"]:
        X_test[col] = X_test["region_id"].map(
            last_w[col] if col in last_w.columns else pd.Series(dtype=float)
        ).astype(np.float32).fillna(0)
    for f in BASE_FEATURES:
        if f not in X_test.columns: X_test[f] = np.float32(0)

    # Autoregressive Inference: jede Woche bekommt Vorhersagen der Vorwochen
    test_preds = predict_chain(final_models, X_test)

    sub = pd.DataFrame({"region_id": test_region_ids})
    for k in range(5): sub[f"pred_week{k+1}"] = test_preds[:, k]
    template = pd.read_csv(SAMPLE_SUB)
    sub = template[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]: sub[col] = sub[col].fillna(0.0)
    sub.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*66}")
    print(f"  Submission: {OUT_PATH}  ({len(sub):,} Rows)")
    print(f"  Val MAE (Inference): {mae(y_va, val_preds_chain):.4f}")
    print(f"  Gesamtlaufzeit: {elapsed(t0)}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
