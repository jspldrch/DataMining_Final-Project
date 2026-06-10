"""
kaggle_v33_xgb2000.py  --  v30-Features + Multi-Seed XGB N=2000 + GPU
======================================================================
Analyse v32 (0.8090 < v30 0.8047):
  - SPI verdrängte Rolling Stats: prec_spi180 allein 34% Importance
  - Blend flippte von xgb=0.90 auf lgb=0.80 → schlechtere Generalisierung
  - XGB traf Limit bei N=1000 → brauchte mehr Bäume
  - Sample Weights dreimal bestätigt schädlich

v33 = zurück zu v30-Featureset + drei Fixes:
  1. Multi-Seed XGB: 3 Seeds × N=2000 (GPU) / 2 Seeds × N=1500 (CPU)
     → XGB hat in v30 xgb=0.90 Blend-Gewicht. Seed-Averaging direkt dort.
  2. Multi-Seed LGB: 3 Seeds × N=1500 (wie v31/v32)
  3. Sample Weights: entfernt (dreimal schädlich bestätigt)

KEIN SPI — KEINE TRAJECTORY — KEINE SAMPLE WEIGHTS.

Features (175, identisch zu v30):
  Weather raw:       14
  Lags:              35
  Rolling stats:    108  (6 cols × 6 wins × 3 stats, surf_pre + dp_tmp)
  Cyclic:             4
  Drought indices:    7
  Regional mean:      1
  Seasonal current:   1
  Seasonal forward:   5  rsm_fw_wk{1..5}

GPU Accelerator empfohlen:
  Mit GPU (T4/P100): N_EST_XGB=2000, 3 Seeds → ~2.5 Stunden
  Ohne GPU (CPU):    N_EST_XGB=1500, 2 Seeds → ~4 Stunden

Dataset: gleiche Pfade wie v32 (glob findet NPZ automatisch)
Output:  /kaggle/working/submission_v33_xgb2000.csv
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

# ── GPU Detection ──────────────────────────────────────────────────────────────
def _detect_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            ver  = tuple(int(x) for x in xgb.__version__.split(".")[:2])
            param = {"device": "cuda"} if ver >= (2, 0) else {"tree_method": "gpu_hist"}
            print(f"  GPU: {name}  |  XGB: {param}")
            return True, param
    except ImportError:
        pass
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, timeout=8)
        if r.returncode == 0:
            name = r.stdout.decode().strip().split("\n")[0]
            ver  = tuple(int(x) for x in xgb.__version__.split(".")[:2])
            param = {"device": "cuda"} if ver >= (2, 0) else {"tree_method": "gpu_hist"}
            print(f"  GPU: {name}  |  XGB: {param}")
            return True, param
    except Exception:
        pass
    print("  Kein GPU — XGB auf CPU")
    return False, {}

HAS_GPU, XGB_GPU_PARAM = _detect_gpu()

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR      = Path("/kaggle/working")
WEEKLY_CACHE  = WORK_DIR / "cache_weekly_v33.npz"
WINDOWS_CACHE = WORK_DIR / "cache_windows_v33.npz"
OUT_PATH      = WORK_DIR / "submission_v33_xgb2000.csv"

def _find_npz(name):
    for slug in ["datafinal","datafiles","datatrain","datatest",
                 "traindataset","testdataset","data"]:
        p = Path(f"/kaggle/input/{slug}/{name}")
        if p.exists(): return p
    found = sorted(_g.glob(f"/kaggle/input/**/{name}", recursive=True))
    if found: return Path(found[0])
    p = WORK_DIR / name
    if p.exists(): return p
    avail = sorted(str(x) for x in Path("/kaggle/input/").iterdir()) \
            if Path("/kaggle/input/").exists() else ["(none)"]
    raise FileNotFoundError(f"'{name}' not found.\n  " + "\n  ".join(avail))

def _find_sample_sub():
    for p in ["/kaggle/input/datafinal/sample_submission.csv",
              "/kaggle/input/samplesub/sample_submission.csv",
              "/kaggle/input/samplesubmission/sample_submission.csv"]:
        if Path(p).exists(): return Path(p)
    found = _g.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(sorted(found)[0]) if found else None

SAMPLE_SUB = _find_sample_sub()

# ── Knobs (GPU-abhängig) ───────────────────────────────────────────────────────
RANDOM_STATE     = 42
HOLDOUT_FRAC     = 0.20
HOLDOUT_SEED     = 42
WEEK_BUCKET      = 7
DRY_THRESHOLD    = 1.0
WINDOW_STRIDE    = 1
RECENT_YEARS     = 8
ORDINAL_PER_YEAR = 372
DAYS_PER_MONTH   = 31

N_EST_LGB  = 1500
SEEDS_LGB  = [42, 123, 777]

# GPU: 3 Seeds × 2000 Bäume | CPU: 2 Seeds × 1500 Bäume
if HAS_GPU:
    N_EST_XGB = 2000
    SEEDS_XGB = [42, 123, 777]
else:
    N_EST_XGB = 1500
    SEEDS_XGB = [42, 123]

N_EST_CAT  = 1000

WEATHER_COLS = [
    "prec","surf_pre","humidity","tmp","dp_tmp","wb_tmp",
    "tmp_max","tmp_min","tmp_range","surf_tmp",
    "wind","wind_max","wind_min","wind_range",
]
LAG_COLS  = ["tmp_range","tmp_max","tmp","prec","wind","surf_pre","humidity"]
LAGS      = [1,3,7,14,21]
ROLL_COLS = ["prec","humidity","tmp","wind","surf_pre","dp_tmp"]  # wie v30
ROLL_WINS = [7,14,30,60,90,180]

LGB_P = dict(
    objective="regression", metric="mae", n_estimators=N_EST_LGB,
    learning_rate=0.03, num_leaves=127, min_child_samples=60,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, verbose=-1,
)
XGB_P = dict(
    objective="reg:squarederror", n_estimators=N_EST_XGB, learning_rate=0.03,
    max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=(1 if HAS_GPU else -1),
    verbosity=0,
    **XGB_GPU_PARAM,
)
CAT_P = dict(
    iterations=N_EST_CAT, learning_rate=0.03, depth=6,
    loss_function="MAE", eval_metric="MAE",
    random_seed=RANDOM_STATE, verbose=False, thread_count=-1,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def elapsed(t0): s=time.time()-t0; return f"{s/60:.1f}m" if s>=60 else f"{s:.0f}s"
def mae(y,p): return float(np.mean(np.abs(np.clip(p,0,5)-y)))
def show(name,y,p): print(f"  {name:<58s}  MAE={mae(y,p):.4f}")
def _best_n(m,default):
    for a in ("best_iteration_","best_iteration"):
        v=getattr(m,a,None)
        if v is not None: return int(v)
    try: return int(m.get_best_iteration())
    except: return default

def _save_npz_atomic(path,**arrays):
    tmp=path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,**arrays)
    tmp.rename(path)

# ── Feature list (175, identisch zu v30) ──────────────────────────────────────
def build_features():
    f  = list(WEATHER_COLS)
    f += [f"{c}_lag{l}" for c in LAG_COLS for l in LAGS]
    f += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS
          for s in ("mean","std","max")]
    f += ["month_sin","month_cos","day_sin","day_cos"]
    f += ["prec_deficit_90d","prec_trend_30d","humidity_deficit_90d",
          "tmp_anomaly_90d","heat_drought_idx","dry_days_14d","dry_days_30d"]
    f.append("regional_mean_score")
    f.append("regional_seasonal_mean")
    f += [f"rsm_fw_wk{k}" for k in range(1,6)]
    return f  # 175 Features

def _future_month(month,day,k_weeks):
    total=(int(month)-1)*DAYS_PER_MONTH+int(day)+k_weeks*7
    return ((total-1)//DAYS_PER_MONTH)%12+1

# ── NPZ loading ────────────────────────────────────────────────────────────────
def load_npz(path):
    d=np.load(path,allow_pickle=True)
    names=d["region_names"]
    df=pd.DataFrame({
        "region_id":names[d["region_id"]],
        "year":d["year"].astype(np.int32),
        "month":d["month"].astype(np.int32),
        "day":d["day"].astype(np.int32),
    })
    df["ordinal"]=df["year"]*372+df["month"]*31+df["day"]
    for col in WEATHER_COLS:
        if col in d: df[col]=d[col].astype(np.float32)
    if "score" in d: df["score"]=d["score"].astype(np.float32)
    return df

# ── Feature engineering ────────────────────────────────────────────────────────
def _region_features(tr,te):
    te=te.copy(); te["score"]=np.nan
    panel=pd.concat([tr,te],ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc={}
    nc["month_sin"]=np.sin(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["month_cos"]=np.cos(2*np.pi*panel["month"]/12).astype(np.float32)
    nc["day_sin"]  =np.sin(2*np.pi*panel["day"]  /31).astype(np.float32)
    nc["day_cos"]  =np.cos(2*np.pi*panel["day"]  /31).astype(np.float32)
    for col in LAG_COLS:
        for lag in LAGS:
            nc[f"{col}_lag{lag}"]=panel[col].shift(lag).astype(np.float32)
    for col in ROLL_COLS:
        prior=panel[col].shift(1)
        for w in ROLL_WINS:
            r=prior.rolling(w,min_periods=max(3,w//10))
            nc[f"{col}_roll{w}_mean"]=r.mean().astype(np.float32)
            nc[f"{col}_roll{w}_std"] =r.std().astype(np.float32)
            nc[f"{col}_roll{w}_max"] =r.max().astype(np.float32)
    pp=panel["prec"].shift(1)
    nc["prec_deficit_90d"]    =(pp.rolling(90,min_periods=30).mean()-
                                 pp.rolling(365,min_periods=60).mean()).astype(np.float32)
    p7=pp.rolling(7,min_periods=3).mean(); p30=pp.rolling(30,min_periods=10).mean()
    nc["prec_trend_30d"]      =((p7-p30)/pp.rolling(30,min_periods=10).std().clip(lower=0.01)).astype(np.float32)
    hp=panel["humidity"].shift(1)
    nc["humidity_deficit_90d"]=(hp.rolling(90,min_periods=30).mean()-
                                 hp.rolling(365,min_periods=60).mean()).astype(np.float32)
    tp=panel["tmp"].shift(1)
    anom=(tp.rolling(90,min_periods=30).mean()-tp.rolling(365,min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]=anom
    nc["heat_drought_idx"]=(nc["prec_deficit_90d"]*anom.clip(lower=0)).astype(np.float32)
    dry=(panel["prec"].shift(1)<DRY_THRESHOLD).astype(np.float32)
    nc["dry_days_14d"]=dry.rolling(14,min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"]=dry.rolling(30,min_periods=7).sum().astype(np.float32)
    panel=pd.concat([panel,pd.DataFrame(nc,index=panel.index)],axis=1)
    n=len(tr)
    return panel.iloc[:n].copy(),panel.iloc[n:].copy()

def _daily_to_weekly(df):
    wk=df["ordinal"]//WEEK_BUCKET
    return df.loc[df.groupby(wk,sort=False)["ordinal"].idxmax()].reset_index(drop=True)

# ── Weekly cache ───────────────────────────────────────────────────────────────
def load_weekly(t0):
    if WEEKLY_CACHE.exists():
        print(f"  [Cache] Weekly v33: {WEEKLY_CACHE.stat().st_size/1e6:.0f} MB")
        ck=dict(np.load(WEEKLY_CACHE,allow_pickle=True))
        base=list(ck["feature_names"])
        weekly=pd.DataFrame(ck["weekly_feats"],columns=base)
        weekly["score"]    =ck["weekly_scores"].astype(np.float32)
        weekly["region_id"]=ck["weekly_region"].astype(str)
        weekly["ordinal"]  =ck["weekly_ordinal"].astype(np.int32)
        return weekly,ck["X_test_base"].astype(np.float32),ck["test_region_ids"].astype(str),base

    print(f"  No cache — feature engineering (~25 min) ... [{elapsed(t0)}]")
    train_raw=load_npz(_find_npz("train.npz"))
    test_raw =load_npz(_find_npz("test.npz"))
    train_raw["score"]=pd.to_numeric(train_raw["score"],errors="coerce").astype(np.float32)
    regions     =train_raw["region_id"].unique()
    region_means=train_raw.groupby("region_id")["score"].mean()
    tr_by={r:g.reset_index(drop=True) for r,g in train_raw.groupby("region_id",sort=False)}
    te_by={r:g.reset_index(drop=True) for r,g in test_raw.groupby("region_id", sort=False)}
    del train_raw

    all_tr,all_te=[],[]
    for i,region in enumerate(regions,1):
        tf,ef=_region_features(tr_by[region],te_by.get(region,pd.DataFrame()))
        all_tr.append(tf); all_te.append(ef)
        if i%500==0 or i==len(regions): print(f"    {i}/{len(regions)} [{elapsed(t0)}]")
    train_feat=pd.concat(all_tr,ignore_index=True)
    test_feat =pd.concat(all_te, ignore_index=True)
    del all_tr

    train_feat["regional_mean_score"]=train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"] =test_feat["region_id"].map(region_means).astype(np.float32)

    labeled =train_feat[train_feat["score"].notna()].copy()
    s_ser   =labeled.groupby(["region_id","month"])["score"].mean()
    s_map   =s_ser.to_dict()
    fallback=region_means.to_dict()
    labeled["regional_seasonal_mean"]=np.array(
        [s_map.get((r,int(m)),fallback.get(r,0.0))
         for r,m in zip(labeled["region_id"],labeled["month"])],dtype=np.float32)

    weekly=pd.concat(
        [_daily_to_weekly(g) for _,g in labeled.groupby("region_id",sort=False)],
        ignore_index=True)
    del labeled
    weekly=weekly.sort_values(["region_id","ordinal"]).reset_index(drop=True)

    # Forward seasonal features rsm_fw_wk{1..5}
    fw_bufs={f"rsm_fw_wk{k}":np.zeros(len(weekly),dtype=np.float32) for k in range(1,6)}
    for region,g in weekly.groupby("region_id",sort=True):
        idx=g.index.tolist(); months=g["month"].tolist(); n=len(months)
        for k in range(1,6):
            col=f"rsm_fw_wk{k}"
            for i in range(n):
                m_fwd=months[min(i+k,n-1)]
                fw_bufs[col][idx[i]]=s_map.get((region,int(m_fwd)),fallback.get(region,0.0))
    for k in range(1,6): weekly[f"rsm_fw_wk{k}"]=fw_bufs[f"rsm_fw_wk{k}"]
    del fw_bufs

    base_cols=[c for c in weekly.columns
               if c not in ("score","region_id","ordinal","date","year","month","day")]

    # Test: idxmax(ordinal) im letzten Ordinal-Bucket + seasonal features
    test_parts=[]
    for region,g in test_feat.groupby("region_id",sort=False):
        g=g.sort_values("ordinal")
        last_ord=int(g["ordinal"].max()); bucket=last_ord//WEEK_BUCKET
        mask=g["ordinal"]//WEEK_BUCKET==bucket
        row=g.loc[[g.loc[mask,"ordinal"].idxmax()]].copy()
        test_parts.append(row)
    X_test_df=pd.concat(test_parts,ignore_index=True)
    del all_te,test_feat

    X_test_df["regional_seasonal_mean"]=np.array(
        [s_map.get((r,int(m)),fallback.get(r,0.0))
         for r,m in zip(X_test_df["region_id"],X_test_df["month"])],dtype=np.float32)
    for k in range(1,6):
        X_test_df[f"rsm_fw_wk{k}"]=np.array([
            s_map.get((r,_future_month(int(m),int(d),k)),fallback.get(r,0.0))
            for r,m,d in zip(X_test_df["region_id"],X_test_df["month"],X_test_df["day"])
        ],dtype=np.float32)

    test_ids=X_test_df["region_id"].values.astype(str)
    X_test  =X_test_df[base_cols].to_numpy(np.float32)

    _save_npz_atomic(WEEKLY_CACHE,
        weekly_feats   =weekly[base_cols].to_numpy(np.float32),
        weekly_scores  =weekly["score"].to_numpy(np.float32),
        weekly_region  =weekly["region_id"].values.astype(str),
        weekly_ordinal =weekly["ordinal"].to_numpy(np.int32),
        X_test_base    =X_test, test_region_ids=test_ids,
        feature_names  =np.array(base_cols,dtype=object),
    )
    print(f"  Cache v33 saved (atomic) [{elapsed(t0)}]")
    return weekly,X_test,test_ids,base_cols

# ── Recent filter ──────────────────────────────────────────────────────────────
def filter_recent_per_region(weekly):
    parts=[]
    for _,g in weekly.groupby("region_id",sort=False):
        cutoff=int(g["ordinal"].max())-RECENT_YEARS*ORDINAL_PER_YEAR
        parts.append(g[g["ordinal"]>=cutoff])
    return pd.concat(parts,ignore_index=True)

# ── Stratifizierter Holdout-Val ────────────────────────────────────────────────
def build_stratified_holdout_windows(weekly_recent,features):
    rng        =np.random.default_rng(HOLDOUT_SEED)
    all_regions=np.array(weekly_recent["region_id"].unique())
    mean_scores=weekly_recent.groupby("region_id")["score"].mean()
    scores_arr =mean_scores.reindex(all_regions).fillna(0).values
    quartiles  =pd.qcut(scores_arr,q=4,labels=False,duplicates="drop")
    n_q        =int(quartiles.max())+1
    val_list=[]
    for q in range(n_q):
        q_regions=all_regions[quartiles==q]
        val_list.extend(rng.choice(q_regions,max(1,int(len(q_regions)*HOLDOUT_FRAC)),replace=False).tolist())
    val_set=set(val_list)
    train_set=set(all_regions)-val_set
    vm=weekly_recent[weekly_recent["region_id"].isin(val_set)]["score"].mean()
    tm=weekly_recent[weekly_recent["region_id"].isin(train_set)]["score"].mean()
    diff=abs(vm-tm)
    print(f"  Split: {len(train_set)} train / {len(val_set)} val  |  "
          f"Val={vm:.4f} Train={tm:.4f} Diff={diff:.4f} {'✓' if diff<0.02 else '⚠'}")
    Xtr,ytr,rtr=[],[],[]; Xva,yva,rva=[],[],[]; Xal,yal,ral=[],[],[]
    for region,g in weekly_recent.groupby("region_id",sort=False):
        g=g.sort_values("ordinal"); sc=g["score"].to_numpy(np.float32)
        Xn=g[features].to_numpy(np.float32); n=len(g)
        if n<6: continue
        nw=n-5; yr=np.lib.stride_tricks.sliding_window_view(sc[1:],5)[:nw]
        idx_all=list(range(0,nw,WINDOW_STRIDE))
        if (nw-1) not in idx_all: idx_all.append(nw-1)
        Xal.append(Xn[idx_all]); yal.append(yr[idx_all]); ral.extend([region]*len(idx_all))
        if region in val_set:
            Xva.append(Xn[nw-1]); yva.append(yr[nw-1]); rva.append(region)
        else:
            idx=list(range(0,nw,WINDOW_STRIDE))
            if (nw-1) not in idx: idx.append(nw-1)
            Xtr.append(Xn[idx]); ytr.append(yr[idx]); rtr.extend([region]*len(idx))
    def _mk(Xs,ys,rs):
        X=pd.DataFrame(np.vstack(Xs).astype(np.float32),columns=features)
        X["region_id"]=pd.Categorical(rs)
        return X,np.vstack(ys).astype(np.float32)
    return *_mk(Xtr,ytr,rtr),*_mk(Xva,yva,rva),*_mk(Xal,yal,ral)

def load_or_build_windows(weekly_recent,features,t0):
    if WINDOWS_CACHE.exists():
        ck=dict(np.load(WINDOWS_CACHE,allow_pickle=True))
        if list(ck["feature_names"])==features:
            print(f"  [Cache] Windows v33: {WINDOWS_CACHE.stat().st_size/1e6:.0f} MB")
            def _r(p):
                X=pd.DataFrame(ck[f"X_{p}"],columns=features)
                X["region_id"]=pd.Categorical(ck[f"r_{p}"].astype(str).tolist())
                return X,ck[f"y_{p}"]
            return *_r("tr"),*_r("va"),*_r("all")
        print("  Cache outdated — rebuilding ...")
    print(f"  Building windows [{elapsed(t0)}]")
    X_tr,y_tr,X_va,y_va,X_all,y_all=build_stratified_holdout_windows(weekly_recent,features)
    _save_npz_atomic(WINDOWS_CACHE,
        X_tr=X_tr[features].to_numpy(np.float32), y_tr=y_tr,
        r_tr=np.array(X_tr["region_id"].astype(str),dtype=object),
        X_va=X_va[features].to_numpy(np.float32), y_va=y_va,
        r_va=np.array(X_va["region_id"].astype(str),dtype=object),
        X_all=X_all[features].to_numpy(np.float32),y_all=y_all,
        r_all=np.array(X_all["region_id"].astype(str),dtype=object),
        feature_names=np.array(features,dtype=object),
    )
    print(f"  Windows cache v33 saved (atomic) [{elapsed(t0)}]")
    return X_tr,y_tr,X_va,y_va,X_all,y_all

# ── Multi-Seed LGB (CPU, keine Sample Weights) ────────────────────────────────
def train_lgb_ms(X_tr,y_tr,X_va,y_va,n_trees_pw=None):
    all_ms=[]
    for seed in SEEDS_LGB:
        wks=[]
        for wk in range(5):
            n=(n_trees_pw[wk] if n_trees_pw else None) or LGB_P["n_estimators"]
            m=lgb.LGBMRegressor(**dict(LGB_P,random_state=seed,n_estimators=n))
            kw=dict(categorical_feature=["region_id"])
            if X_va is not None:
                kw.update(eval_set=[(X_va,y_va[:,wk].ravel())],eval_metric="mae",
                          callbacks=[lgb.early_stopping(50,verbose=False)])
            m.fit(X_tr,y_tr[:,wk].ravel(),**kw)
            wks.append(m)
        all_ms.append(wks)
    return all_ms

def pred_lgb_ms(all_ms,X):
    feat=all_ms[0][0].booster_.feature_name()
    preds=[np.column_stack([m.predict(X[feat]) for m in wms]) for wms in all_ms]
    return np.clip(np.mean(preds,axis=0),0,5).astype(np.float32)

def get_avg_iters_lgb(all_ms):
    return [int(round(np.mean([_best_n(sm[wk],N_EST_LGB) for sm in all_ms])))
            for wk in range(5)]

# ── Multi-Seed XGB (GPU, keine Sample Weights) ────────────────────────────────
def train_xgb_ms(X_tr,y_tr,X_va,y_va,features,n_trees_pw=None):
    Xn=X_tr[features].to_numpy(np.float32)
    Vn=X_va[features].to_numpy(np.float32) if X_va is not None else None
    all_ms=[]
    for seed in SEEDS_XGB:
        wks=[]
        for wk in range(5):
            n=(n_trees_pw[wk] if n_trees_pw else None) or XGB_P["n_estimators"]
            p=dict(XGB_P,random_state=seed,n_estimators=n)
            kw={}
            if Vn is not None:
                p["early_stopping_rounds"]=50
                kw.update(eval_set=[(Vn,y_va[:,wk].ravel())],verbose=False)
            m=xgb.XGBRegressor(**p)
            m.fit(Xn,y_tr[:,wk].ravel(),**kw)
            wks.append(m)
        all_ms.append(wks)
    return all_ms

def pred_xgb_ms(all_ms,X,features):
    Xn=X[features].to_numpy(np.float32)
    preds=[np.clip(np.column_stack([m.predict(Xn) for m in wms]),0,5) for wms in all_ms]
    return np.mean(preds,axis=0).astype(np.float32)

def get_avg_iters_xgb(all_ms):
    return [int(round(np.mean([_best_n(sm[wk],N_EST_XGB) for sm in all_ms])))
            for wk in range(5)]

# ── Single-Seed CatBoost ───────────────────────────────────────────────────────
def train_cat(X_tr,y_tr,X_va,y_va,features,n_trees=None):
    if not CAT: return None
    Xn=X_tr[features].to_numpy(np.float32)
    Vn=X_va[features].to_numpy(np.float32) if X_va is not None else None
    models=[]
    for wk in range(5):
        n=(n_trees[wk] if n_trees else None) or CAT_P["iterations"]
        p=dict(CAT_P,iterations=n,random_seed=RANDOM_STATE+wk)
        kw={}
        if Vn is not None: kw.update(eval_set=(Vn,y_va[:,wk].ravel()),early_stopping_rounds=50)
        m=CatBoostRegressor(**p)
        m.fit(Xn,y_tr[:,wk].ravel(),**kw)
        models.append(m)
    return models

def pred_num(models,X,features):
    Xn=X[features].to_numpy(np.float32)
    return np.clip(np.column_stack([m.predict(Xn) for m in models]),0,5).astype(np.float32)

def blend(y_va,preds):
    names=list(preds); arrays=[preds[n] for n in names]
    alphas=[round(x*0.05,2) for x in range(1,20)]
    best_mae,best_w=999.,{n:1/len(names) for n in names}
    if len(names)==2:
        for a in alphas:
            m=mae(y_va,a*arrays[0]+(1-a)*arrays[1])
            if m<best_mae: best_mae,best_w=m,{names[0]:a,names[1]:round(1-a,8)}
    elif len(names)==3:
        for a in alphas:
            for b in alphas:
                c=round(1-a-b,8)
                if c<0.05: continue
                m=mae(y_va,a*arrays[0]+b*arrays[1]+c*arrays[2])
                if m<best_mae: best_mae,best_w=m,{names[0]:a,names[1]:b,names[2]:c}
    return best_w,best_mae

def print_importance(lgb_ms,features):
    feat=np.array(lgb_ms[0][0].booster_.feature_name())
    imp =sum(m.booster_.feature_importance("gain")
             for wms in lgb_ms for m in wms)/(len(lgb_ms)*5)
    mask=feat!="region_id"; feat=feat[mask]; imp=imp[mask]
    total=imp.sum(); order=np.argsort(imp)[::-1]
    print(f"\n{'='*66}")
    print(f"  FEATURE IMPORTANCE (LGB Gain, avg seeds/weeks, top 20)")
    for rank,i in enumerate(order[:20],1):
        tag=" ◄NEW" if any(x in feat[i] for x in ["surf_pre_roll","dp_tmp_roll"]) else \
            " ◄FW"  if "rsm_fw" in feat[i] else ""
        print(f"  {rank:<4d}  {feat[i]:<42}  {100*imp[i]/total:>5.2f}%{tag}")
    groups={
        "Rolling (prec/hum/tmp/wind)":["roll" in f and not any(x in f for x in ["surf_pre","dp_tmp"]) for f in feat],
        "Rolling surf_pre ◄":        ["surf_pre_roll" in f for f in feat],
        "Rolling dp_tmp ◄":          ["dp_tmp_roll"   in f for f in feat],
        "Drought indices":           [any(k in f for k in ["deficit","trend","anomaly","drought","dry_days"]) for f in feat],
        "Seasonal (current+forward)": [("seasonal" in f or "rsm_fw" in f) for f in feat],
        "Regional mean":             [f=="regional_mean_score" for f in feat],
        "Lags":                      ["_lag" in f for f in feat],
        "Weather raw":               [f in WEATHER_COLS for f in feat],
    }
    print(f"\n  Groups:")
    for gname,ml in groups.items():
        g_imp=imp[[i for i,v in enumerate(ml) if v]].sum()
        print(f"    {gname:<34}  {100*g_imp/total:>5.1f}%")
    print(f"  Top-10 cumul.: {100*imp[order[:10]].sum()/total:.1f}%")
    print(f"  → Referenz v30: Rolling 70.1% | Seasonal 6.8% | XGB blend 0.90")
    print(f"{'='*66}\n")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0      =time.time()
    FEATURES=build_features()
    dev_str =f"GPU ({list(XGB_GPU_PARAM.values())[0]})" if HAS_GPU else "CPU"
    est_h   ="~2.5h" if HAS_GPU else "~4h"
    print("="*66)
    print(f"  kaggle_v33_xgb2000  |  {len(FEATURES)} features (v30-Basis)")
    print(f"  LGB: {len(SEEDS_LGB)} seeds × N={N_EST_LGB} (CPU, keine Weights)")
    print(f"  XGB: {len(SEEDS_XGB)} seeds × N={N_EST_XGB} ({dev_str})")
    print(f"  KEIN SPI — KEINE TRAJECTORY — KEINE SAMPLE WEIGHTS")
    print(f"  Laufzeit: {est_h}")
    print(f"  Dataset: {_find_npz('train.npz').parent}")
    print("="*66)

    print(f"\n[1/5] Weekly features ... [{elapsed(t0)}]")
    weekly,X_test_base,test_ids,base_cols=load_weekly(t0)
    n_regions=weekly["region_id"].nunique(); n_all=len(weekly)
    print(f"  {n_all:,} weekly rows  |  {n_regions} regions  |  {len(FEATURES)} features")
    for f in FEATURES:
        if f not in weekly.columns: weekly[f]=np.float32(0)

    print(f"\n[2/5] Recent filter: last {RECENT_YEARS} years ... [{elapsed(t0)}]")
    weekly_recent=filter_recent_per_region(weekly)
    print(f"  {n_all:,} → {len(weekly_recent):,}  ({100*len(weekly_recent)/n_all:.0f}%)")

    print(f"\n[3/5] Build windows (stratified holdout) ... [{elapsed(t0)}]")
    X_tr,y_tr,X_va,y_va,X_all,y_all=load_or_build_windows(weekly_recent,FEATURES,t0)
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}  All: {len(X_all):,}")
    last_score=weekly_recent.sort_values("ordinal").groupby("region_id")["score"].last()
    persist=np.column_stack([last_score.reindex(X_va["region_id"].astype(str).tolist()).fillna(0).to_numpy()]*5)
    show("Persistence",y_va,persist)

    print(f"\n[4/5] Training ... [{elapsed(t0)}]")

    # LGB
    lgb_ms=train_lgb_ms(X_tr,y_tr,X_va,y_va)
    lgb_val=pred_lgb_ms(lgb_ms,X_va)
    show(f"LGB ({len(SEEDS_LGB)}-seed, CPU)",y_va,lgb_val)
    avg_iters_lgb=get_avg_iters_lgb(lgb_ms)
    for wk in range(5):
        si=[_best_n(lgb_ms[s][wk],N_EST_LGB) for s in range(len(SEEDS_LGB))]
        hit="  ← HIT LIMIT" if max(si)>=N_EST_LGB-5 else ""
        print(f"    LGB wk{wk+1}: {si}  avg={avg_iters_lgb[wk]}{hit}")

    # XGB
    print(f"  XGB training ({dev_str}) ... [{elapsed(t0)}]")
    xgb_ms=train_xgb_ms(X_tr,y_tr,X_va,y_va,FEATURES)
    xgb_val=pred_xgb_ms(xgb_ms,X_va,FEATURES)
    show(f"XGB ({len(SEEDS_XGB)}-seed, {dev_str})",y_va,xgb_val)
    avg_iters_xgb=get_avg_iters_xgb(xgb_ms)
    for wk in range(5):
        si=[_best_n(xgb_ms[s][wk],N_EST_XGB) for s in range(len(SEEDS_XGB))]
        hit="  ← HIT LIMIT" if max(si)>=N_EST_XGB-5 else ""
        print(f"    XGB wk{wk+1}: {si}  avg={avg_iters_xgb[wk]}{hit}")

    preds_val={"lgb":lgb_val,"xgb":xgb_val}
    cat_m=train_cat(X_tr,y_tr,X_va,y_va,FEATURES)
    if cat_m:
        cat_val=pred_num(cat_m,X_va,FEATURES)
        show("CatBoost (1-seed)",y_va,cat_val)
        preds_val["cat"]=cat_val

    best_w,best_val_mae=blend(y_va,preds_val)
    print(f"  Blend: {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}  MAE={best_val_mae:.4f}")
    print(f"  → Referenz v30: lgb=0.05 xgb=0.90 cat=0.05  MAE≈0.27")
    print_importance(lgb_ms,FEATURES)

    print(f"\n[5/5] Final training ({len(X_all):,} windows) ... [{elapsed(t0)}]")
    f_lgb=train_lgb_ms(X_all,y_all,None,avg_iters_lgb)
    f_xgb=train_xgb_ms(X_all,y_all,None,FEATURES,avg_iters_xgb)
    f_cat=None
    if cat_m:
        n_cat=[_best_n(m,N_EST_CAT) for m in cat_m]
        f_cat=train_cat(X_all,y_all,None,None,FEATURES,n_cat)

    X_test=pd.DataFrame(X_test_base,columns=base_cols)
    X_test["region_id"]=pd.Categorical(test_ids)
    for f in FEATURES:
        if f not in X_test.columns: X_test[f]=np.float32(0)

    test_preds=(best_w["lgb"]*pred_lgb_ms(f_lgb,X_test)+
                best_w["xgb"]*pred_xgb_ms(f_xgb,X_test,FEATURES))
    if f_cat and "cat" in best_w:
        test_preds+=best_w["cat"]*pred_num(f_cat,X_test,FEATURES)

    sub=pd.DataFrame({"region_id":test_ids})
    for k in range(5): sub[f"pred_week{k+1}"]=test_preds[:,k]
    if SAMPLE_SUB:
        sub=pd.read_csv(SAMPLE_SUB)[["region_id"]].merge(sub,on="region_id",how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col]=sub[col].fillna(0.0)
    sub.to_csv(OUT_PATH,index=False)
    print(f"  Saved: {OUT_PATH.name}  ({len(sub):,} rows)")

    print()
    print("="*66)
    print(f"  RESULTS — kaggle_v33_xgb2000")
    print(f"  {'-'*62}")
    print(f"  {'Features (v30-Basis, kein SPI)':.<40} {len(FEATURES)}")
    print(f"  {'XGB':.<40} {len(SEEDS_XGB)} seeds × N={N_EST_XGB} ({dev_str})")
    print(f"  {'LGB':.<40} {len(SEEDS_LGB)} seeds × N={N_EST_LGB} (CPU)")
    print(f"  {'Avg LGB iters (wk1-5)':.<40} {avg_iters_lgb}")
    print(f"  {'Avg XGB iters (wk1-5)':.<40} {avg_iters_xgb}")
    print(f"  {'Blend val MAE':.<40} {best_val_mae:.4f}")
    print(f"  {'Blend':.<40} {' '.join(f'{k}={v:.2f}' for k,v in best_w.items())}")
    print(f"  {'-'*62}")
    print(f"  Referenz: v30=0.8047 | v32=0.8090 | Baseline3=0.8056")
    print(f"  {'Runtime':.<40} {elapsed(t0)}")
    print("="*66)

main()
