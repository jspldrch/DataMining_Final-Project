"""
run_v5_dl.py  –  Deep Learning Ensemble for drought severity forecasting

Models
------
  1. AutoregressiveGRU  – 2-layer GRU encoder + step-by-step GRU decoder
                           (teacher forcing during training, free-run at test time)
  2. CNNLSTMModel       – Multi-scale Conv1d feature extractor + 1-layer LSTM
  3. TransformerModel   – Learned positional encoding + 2-layer Transformer encoder
  4. LightGBM           – GBDT point-feature baseline (blended with DL)

Key idea vs GBDT:
  The DL models consume the last SEQ_LEN=26 weekly weather observations as a
  sequence, capturing temporal dynamics (trends, cycles) that a single feature
  vector cannot represent.  No score features in the sequence input — avoids the
  train/test distribution shift that hurt v3.

Requirements: pip install torch lightgbm scikit-learn pandas numpy
Run         : python scripts/run_v5_dl.py
Output      : outputs/submission_v5.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found  →  pip install torch")
    print("          Continuing in LightGBM-only mode.")

# ─── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV  = DATA_DIR / "train.csv"
TEST_CSV   = DATA_DIR / "test.csv"
SAMPLE_SUB = ROOT / "resources" / "sample_submission.csv"
OUT_PATH   = OUT_DIR / "submission_v5.csv"

# ─── Config ─────────────────────────────────────────────────────────────────────
# QUICK_MODE = True  →  ~30 min  (verify pipeline, rough score)
# QUICK_MODE = False →  ~3-4 h   (best Kaggle result; GPU recommended)
QUICK_MODE = True

RANDOM_STATE     = 42
VAL_REGION_FRAC  = 0.20
WEEK_BUCKET      = 7
SEQ_LEN          = 26      # weeks of weather history fed to DL models (~6 months)
N_WEEKS          = 5       # prediction horizon
WINDOW_STRIDE    = 4 if QUICK_MODE else 1

# DL training
DL_EPOCHS   = 60  if QUICK_MODE else 200
DL_BATCH    = 512
DL_LR       = 1e-3
DL_HIDDEN   = 128 if QUICK_MODE else 256
DL_LAYERS   = 2
DL_DROPOUT  = 0.2
DEVICE      = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

# GBDT
N_ESTIMATORS = 400 if QUICK_MODE else 1000

# ─── Feature definitions ────────────────────────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp"]
ROLL_WINS = [7, 14, 30, 60, 90]
DRY_THR   = 1.0   # mm/day

GBDT_FEATURES: list[str] = []
DL_FEATURES:   list[str] = []   # smaller weather-only set for sequence models


def _build_feature_lists() -> None:
    global GBDT_FEATURES, DL_FEATURES
    lag_names  = [f"{c}_lag{lag}" for c in LAG_COLS for lag in LAGS]
    roll_all   = [f"{col}_roll{w}_{s}" for col in ROLL_COLS for w in ROLL_WINS
                  for s in ("mean", "std", "max")]
    roll_means = [f"{col}_roll{w}_mean" for col in ROLL_COLS for w in ROLL_WINS]
    calendar   = ["month_sin", "month_cos", "day_sin", "day_cos"]
    drought    = ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
                  "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    GBDT_FEATURES = WEATHER_COLS + lag_names + roll_all + calendar + ["score_persist"] + drought
    # DL uses only weather + rolling means + drought indices (no lags, no score features)
    DL_FEATURES   = WEATHER_COLS + roll_means + calendar + drought


# ─── Date parsing ────────────────────────────────────────────────────────────────
def _parse_dates_inplace(df: pd.DataFrame) -> None:
    parts = df["date"].str.split("-", expand=True)
    df["year"]    = parts[0].astype(np.int32)
    df["month"]   = parts[1].astype(np.int32)
    df["day"]     = parts[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]


# ─── Feature engineering ─────────────────────────────────────────────────────────
def compute_region_features(
    tr: pd.DataFrame, te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc: dict = {}

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
            r = prior.rolling(w, min_periods=3)
            nc[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            nc[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            nc[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)

    pp = panel["prec"].shift(1)
    nc["prec_deficit_90d"] = (
        pp.rolling(90, min_periods=30).mean() - pp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    p7   = pp.rolling(7, min_periods=3).mean()
    p30  = pp.rolling(30, min_periods=10).mean()
    p30s = pp.rolling(30, min_periods=10).std().clip(lower=0.01)
    nc["prec_trend_30d"] = ((p7 - p30) / p30s).astype(np.float32)

    hp = panel["humidity"].shift(1)
    nc["humidity_deficit_90d"] = (
        hp.rolling(90, min_periods=30).mean() - hp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    tp = panel["tmp"].shift(1)
    t_anom = (tp.rolling(90, min_periods=30).mean() - tp.rolling(365, min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]  = t_anom
    nc["heat_drought_idx"] = (nc["prec_deficit_90d"] * t_anom.clip(lower=0)).astype(np.float32)

    dry = (panel["prec"].shift(1) < DRY_THR).astype(np.float32)
    nc["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)

    # score_persist only for GBDT (single last-known level, no trend array)
    nc["score_persist"] = panel["score"].ffill().shift(7).astype(np.float32)

    panel = pd.concat([panel, pd.DataFrame(nc, index=panel.index)], axis=1)
    n_tr = len(tr)
    return panel.iloc[:n_tr].copy(), panel.iloc[n_tr:].copy()


# ─── Weekly helpers ───────────────────────────────────────────────────────────────
def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    week = df["ordinal"] // WEEK_BUCKET
    idx  = df.groupby(week, sort=False)["ordinal"].idxmax()
    return df.loc[idx].reset_index(drop=True)


# ─── GBDT dataset builders ───────────────────────────────────────────────────────
def _gbdt_sliding(weekly, feature_cols, skip, stride):
    Xp, yp, rp = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip:
            continue
        g = g.sort_values("ordinal")
        s = g["score"].to_numpy(dtype=np.float32)
        X = g[feature_cols].to_numpy(dtype=np.float32)
        n = len(g)
        if n < 6:
            continue
        nw = n - 5
        yr = np.lib.stride_tricks.sliding_window_view(s[1:], 5)[:nw]
        idx = list(range(0, nw, stride))
        if nw - 1 not in idx:
            idx.append(nw - 1)
        Xp.append(X[idx]); yp.append(yr[idx]); rp.extend([region]*len(idx))
    Xdf = pd.DataFrame(np.vstack(Xp), columns=feature_cols)
    Xdf["region_id"] = pd.Categorical(rp)
    return Xdf, np.vstack(yp).astype(np.float32)


def _gbdt_val(weekly, feature_cols, val_regions):
    Xp, yp, rp = [], [], []
    for r in val_regions:
        g = weekly.loc[weekly["region_id"] == r].sort_values("ordinal")
        if len(g) < 6:
            continue
        Xp.append(g.iloc[-6][feature_cols].to_numpy(dtype=np.float32))
        yp.append(g.iloc[-5:]["score"].to_numpy(dtype=np.float32))
        rp.append(r)
    Xdf = pd.DataFrame(np.vstack(Xp), columns=feature_cols)
    Xdf["region_id"] = pd.Categorical(rp)
    return Xdf, np.vstack(yp)


# ─── DL sequence builders ────────────────────────────────────────────────────────
def _build_seqs(weekly, feature_cols, skip, stride, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip:
            continue
        g = g.sort_values("ordinal")
        F = g[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        S = g["score"].to_numpy(dtype=np.float32)
        n = len(g)
        if n < seq_len + N_WEEKS:
            continue
        nw = n - seq_len - N_WEEKS + 1
        idx = list(range(0, nw, stride))
        if nw - 1 not in idx:
            idx.append(nw - 1)
        for i in idx:
            Xs.append(F[i:i+seq_len])
            ys.append(S[i+seq_len:i+seq_len+N_WEEKS])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _val_seqs(weekly, feature_cols, val_regions, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for r in val_regions:
        g = weekly.loc[weekly["region_id"] == r].sort_values("ordinal")
        n = len(g)
        if n < seq_len + N_WEEKS:
            continue
        F = g[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        S = g["score"].to_numpy(dtype=np.float32)
        Xs.append(F[-(seq_len+N_WEEKS):-N_WEEKS])
        ys.append(S[-N_WEEKS:])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _test_seqs(train_weekly, test_weekly, feature_cols, seq_len=SEQ_LEN):
    """Last seq_len weekly rows per region from train+test combined."""
    Xs, regions = [], []
    for region in sorted(train_weekly["region_id"].unique()):
        tr_g = train_weekly.loc[train_weekly["region_id"] == region].sort_values("ordinal")
        te_g = test_weekly.loc[test_weekly["region_id"] == region].sort_values("ordinal") \
               if region in test_weekly["region_id"].values else pd.DataFrame()
        comb = pd.concat([tr_g, te_g], ignore_index=True).sort_values("ordinal")
        F = comb[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        if len(F) >= seq_len:
            Xs.append(F[-seq_len:])
        else:
            pad = np.zeros((seq_len-len(F), F.shape[1]), dtype=np.float32)
            Xs.append(np.vstack([pad, F]))
        regions.append(region)
    return np.array(Xs, dtype=np.float32), regions


# ─── PyTorch model definitions ───────────────────────────────────────────────────
if TORCH_AVAILABLE:

    class AutoregressiveGRU(nn.Module):
        """
        Encoder : 2-layer bidirectional GRU processes weather sequence.
        Decoder : single GRU cell generates week1..5 step by step.
                  Teacher forcing ratio decays from 1.0 → 0.0 during training.
        """
        def __init__(self, n_feat, hidden=DL_HIDDEN, n_layers=DL_LAYERS,
                     n_weeks=N_WEEKS, dropout=DL_DROPOUT):
            super().__init__()
            self.n_weeks  = n_weeks
            self.encoder  = nn.GRU(n_feat, hidden, n_layers, batch_first=True,
                                   dropout=dropout if n_layers > 1 else 0.0,
                                   bidirectional=True)
            # Merge bidirectional → hidden
            self.bridge       = nn.Linear(hidden * 2, hidden)
            self.decoder_cell = nn.GRUCell(1, hidden)
            self.out_proj     = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.init_tok = nn.Parameter(torch.zeros(1))

        def forward(self, x, y_teacher=None, tf_ratio=0.5):
            # x: (B, T, F)
            out, _ = self.encoder(x)              # (B, T, 2*hidden)
            ctx    = self.bridge(out[:, -1, :])   # (B, hidden)  — last timestep
            h_dec  = torch.tanh(ctx)
            B      = x.size(0)
            inp    = self.init_tok.expand(B, 1)
            preds  = []
            for step in range(self.n_weeks):
                h_dec = self.decoder_cell(inp, h_dec)
                pred  = self.out_proj(h_dec)           # (B, 1)
                preds.append(pred)
                use_tf = (y_teacher is not None and self.training
                          and torch.rand(1).item() < tf_ratio)
                inp = y_teacher[:, step:step+1] if use_tf else pred.detach()
            return torch.cat(preds, dim=1)  # (B, n_weeks)


    class CNNLSTMModel(nn.Module):
        """
        Multi-scale Conv1d (kernels 3, 5, 7) extracts local weather patterns.
        1-layer LSTM captures long-range temporal dependencies.
        """
        def __init__(self, n_feat, hidden=DL_HIDDEN, n_weeks=N_WEEKS, dropout=DL_DROPOUT):
            super().__init__()
            ch = hidden // 2
            self.conv3 = nn.Sequential(nn.Conv1d(n_feat, ch, 3, padding=1), nn.ReLU())
            self.conv5 = nn.Sequential(nn.Conv1d(n_feat, ch, 5, padding=2), nn.ReLU())
            self.conv7 = nn.Sequential(nn.Conv1d(n_feat, ch, 7, padding=3), nn.ReLU())
            self.drop  = nn.Dropout(dropout)
            self.lstm  = nn.LSTM(ch * 3, hidden, 1, batch_first=True)
            self.head  = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_weeks))

        def forward(self, x, **_):
            # x: (B, T, F)
            xT = x.permute(0, 2, 1)          # (B, F, T) for Conv1d
            c  = torch.cat([self.conv3(xT), self.conv5(xT), self.conv7(xT)], dim=1)
            c  = self.drop(c).permute(0, 2, 1)   # (B, T, ch*3)
            _, (h, _) = self.lstm(c)
            return self.head(h[-1])           # (B, n_weeks)


    class TransformerModel(nn.Module):
        """
        Learned positional encoding + 2-layer Transformer encoder.
        Last timestep representation is projected to n_weeks outputs.
        """
        def __init__(self, n_feat, d_model=None, nhead=4, n_layers=DL_LAYERS,
                     n_weeks=N_WEEKS, dropout=DL_DROPOUT):
            super().__init__()
            d_model = ((DL_HIDDEN if d_model is None else d_model) // nhead) * nhead
            self.proj    = nn.Linear(n_feat, d_model)
            enc_layer    = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model*2,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
            self.pos_emb = nn.Embedding(512, d_model)
            self.head    = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_weeks))

        def forward(self, x, **_):
            B, T, _ = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            x   = self.proj(x) + self.pos_emb(pos)     # (B, T, d_model)
            x   = self.encoder(x)
            return self.head(x[:, -1, :])               # (B, n_weeks)


    def _train_model(model, X_tr, y_tr, X_va, y_va):
        model = model.to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=DL_LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DL_EPOCHS, eta_min=DL_LR/20)

        Xt = torch.tensor(X_tr, device=DEVICE)
        yt = torch.tensor(y_tr, device=DEVICE)
        Xv = torch.tensor(X_va, device=DEVICE)
        yv = torch.tensor(y_va, device=DEVICE)

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=DL_BATCH, shuffle=True,
                            drop_last=False)
        best_mae, best_state, no_imp = np.inf, None, 0
        patience = 20

        for epoch in range(DL_EPOCHS):
            # Teacher-forcing ratio: 1.0 → 0.0 over first half of training
            tf = max(0.0, 1.0 - 2.0 * epoch / DL_EPOCHS)
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                pred = model(xb, y_teacher=yb, tf_ratio=tf)
                loss = torch.mean(torch.abs(pred.clamp(0, 5) - yb))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                v_mae = torch.mean(torch.abs(model(Xv).clamp(0, 5) - yv)).item()
            if v_mae < best_mae:
                best_mae  = v_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model, best_mae


    @torch.no_grad()
    def _predict(model, X, batch=1024):
        model.eval()
        out = []
        Xt  = torch.tensor(X, device=DEVICE)
        for i in range(0, len(Xt), batch):
            out.append(model(Xt[i:i+batch]).clamp(0, 5).cpu().numpy())
        return np.vstack(out)


# ─── Utility ────────────────────────────────────────────────────────────────────
def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.clip(y_pred, 0, 5) - y_true)))

def _show(name, y_true, y_pred):
    print(f"  {name:<52s}  MAE = {_mae(y_true, y_pred):.4f}")

def _normalise(X_tr, X_va, X_te):
    sc = StandardScaler()
    shape_tr = X_tr.shape; shape_va = X_va.shape; shape_te = X_te.shape
    sc.fit(X_tr.reshape(-1, X_tr.shape[-1]))
    return (
        sc.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(shape_tr).astype(np.float32),
        sc.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(shape_va).astype(np.float32),
        sc.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(shape_te).astype(np.float32),
    )


# ─── Main pipeline ───────────────────────────────────────────────────────────────
def main():
    _build_feature_lists()
    t0 = time.time()

    print("=" * 66)
    print("  Natural Disaster Severity Prediction  -  run_v5_dl.py")
    mode = "QUICK (~30 min)" if QUICK_MODE else "FULL (~3-4 h)"
    torch_s = f"PyTorch ON  device={DEVICE}" if TORCH_AVAILABLE else "PyTorch OFF (LGB only)"
    print(f"  Mode: {mode}  |  {torch_s}")
    print(f"  GBDT features: {len(GBDT_FEATURES)}   DL features: {len(DL_FEATURES)}   seq_len: {SEQ_LEN}")
    print("=" * 66)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Lade CSV-Dateien ...")
    dtypes    = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw  = pd.read_csv(TEST_CSV,  dtype=dtypes)
    _parse_dates_inplace(train_raw)
    _parse_dates_inplace(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    print(f"   Train: {len(train_raw):,}  Test: {len(test_raw):,}  Regionen: {len(regions)}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/6] Feature Engineering ...")
    train_by = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    test_by  = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw

    tr_parts, te_parts = [], []
    for i, region in enumerate(regions, 1):
        if i % 500 == 0 or i == len(regions):
            print(f"   Region {i}/{len(regions)}  |  {time.time()-t0:.1f}s")
        tr_f, te_f = compute_region_features(train_by[region],
                                              test_by.get(region, pd.DataFrame()))
        tr_parts.append(tr_f); te_parts.append(te_f)

    train_feat = pd.concat(tr_parts, ignore_index=True)
    test_feat  = pd.concat(te_parts, ignore_index=True)
    del tr_parts, te_parts, train_by, test_by
    print(f"   Fertig  |  {time.time()-t0:.1f}s")

    # ── 3. Weekly aggregation ─────────────────────────────────────────────────
    print("\n[3/6] Woechentliche Aggregation ...")
    labeled = train_feat[train_feat["score"].notna()].copy()
    train_weekly = pd.concat(
        [daily_to_weekly(g) for _, g in labeled.groupby("region_id", sort=False)],
        ignore_index=True,
    )
    test_weekly = pd.concat(
        [daily_to_weekly(g) for _, g in test_feat.groupby("region_id", sort=False)],
        ignore_index=True,
    )
    del labeled
    print(f"   Train-Wochen: {len(train_weekly):,}  |  Test-Wochen: {len(test_weekly):,}")

    # ── 4. Train/val split ────────────────────────────────────────────────────
    print("\n[4/6] Train/Validierung aufbauen ...")
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg   = sorted(train_weekly["region_id"].unique())
    n_val     = max(1, int(len(all_reg) * VAL_REGION_FRAC))
    val_set   = set(rng.choice(all_reg, size=n_val, replace=False))
    val_list  = sorted(val_set)

    # GBDT data
    X_gbdt_tr, y_gbdt_tr = _gbdt_sliding(train_weekly, GBDT_FEATURES, val_set, WINDOW_STRIDE)
    X_gbdt_va, y_gbdt_va = _gbdt_val(train_weekly, GBDT_FEATURES, val_list)

    # DL sequence data
    X_dl_tr, y_dl_tr = _build_seqs(train_weekly, DL_FEATURES, val_set, WINDOW_STRIDE)
    X_dl_va, y_dl_va = _val_seqs(train_weekly, DL_FEATURES, val_list)
    X_dl_te, te_regions = _test_seqs(train_weekly, test_weekly, DL_FEATURES)

    print(f"   GBDT windows: {len(X_gbdt_tr):,}  |  DL sequences: {len(X_dl_tr):,}")
    print(f"   Val regions: {len(val_list)}  |  DL features per step: {len(DL_FEATURES)}")

    # Normalise for DL
    if TORCH_AVAILABLE:
        X_dl_tr, X_dl_va, X_dl_te = _normalise(X_dl_tr, X_dl_va, X_dl_te)

    # Persistence baseline
    last_score = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    persist_va = np.tile(last_score.reindex(val_list).fillna(0).to_numpy()[:, None], (1, 5))
    _show("Persistence-Baseline", y_gbdt_va, persist_va)

    # ── 5. Training ───────────────────────────────────────────────────────────
    print(f"\n[5/6] Training  (device={DEVICE}) ...")
    n_dl_feat = len(DL_FEATURES)

    # LightGBM
    print("   LightGBM ...")
    lgb_params = dict(objective="regression", metric="mae", n_estimators=N_ESTIMATORS,
                      learning_rate=0.04, num_leaves=63, min_child_samples=60,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1,
                      random_state=RANDOM_STATE)
    lgb_models = []
    for w in range(N_WEEKS):
        p = dict(lgb_params)
        m = lgb.LGBMRegressor(**p)
        m.fit(X_gbdt_tr, y_gbdt_tr[:, w],
              eval_set=[(X_gbdt_va, y_gbdt_va[:, w])],
              eval_metric="mae",
              callbacks=[lgb.early_stopping(50, verbose=False)],
              categorical_feature=["region_id"])
        lgb_models.append(m)
    lgb_val = np.clip(np.column_stack([m.predict(X_gbdt_va) for m in lgb_models]), 0, 5)
    _show("LightGBM", y_gbdt_va, lgb_val)

    dl_preds_val: dict[str, np.ndarray] = {}
    dl_models:    dict[str, list]        = {}

    if TORCH_AVAILABLE:
        for name, ModelClass in [
            ("AutoregressiveGRU", AutoregressiveGRU),
            ("CNNLSTMModel",      CNNLSTMModel),
            ("TransformerModel",  TransformerModel),
        ]:
            print(f"   {name} ...")
            model = ModelClass(n_dl_feat)
            model, val_mae = _train_model(model, X_dl_tr, y_dl_tr, X_dl_va, y_dl_va)
            pv = _predict(model, X_dl_va)
            _show(name, y_dl_va, pv)
            dl_preds_val[name] = pv
            dl_models[name]    = model

    # ── Blend optimisation ────────────────────────────────────────────────────
    print("\n  Blend-Optimierung (Validierung) ...")
    candidates: list[np.ndarray] = [lgb_val] + list(dl_preds_val.values())
    cand_names = ["LGB"] + list(dl_preds_val.keys())
    n_c = len(candidates)

    best_mae_v, best_w = np.inf, np.ones(n_c) / n_c
    step = 0.1
    # Simple equal-step grid (sufficient for ≤4 models)
    from itertools import product
    alphas = np.arange(0.0, 1.0 + step, step)
    for combo in product(alphas, repeat=n_c):
        combo = np.array(combo)
        if abs(combo.sum() - 1.0) > 1e-6:
            continue
        blend = sum(w * p for w, p in zip(combo, candidates))
        m = _mae(y_gbdt_va, blend)
        if m < best_mae_v:
            best_mae_v, best_w = m, combo

    weight_str = "  ".join(f"{n}={w:.2f}" for n, w in zip(cand_names, best_w))
    print(f"   {weight_str}   MAE={best_mae_v:.4f}")

    # ── 6. Final prediction on test set ───────────────────────────────────────
    print("\n[6/6] Test-Vorhersagen ...")

    # Retrain LGB on all data
    X_gbdt_all, y_gbdt_all = _gbdt_sliding(train_weekly, GBDT_FEATURES, set(), WINDOW_STRIDE)
    n_lgb_trees = [int(getattr(m, "best_iteration_", None) or N_ESTIMATORS) for m in lgb_models]
    X_gbdt_test = (
        test_feat.sort_values(["region_id", "ordinal"])
        .groupby("region_id", sort=False)
        .tail(1)[["region_id"] + GBDT_FEATURES]
        .reset_index(drop=True)
    )
    X_gbdt_test["region_id"] = X_gbdt_test["region_id"].astype("category")

    final_lgb = []
    for w in range(N_WEEKS):
        p = dict(lgb_params, n_estimators=n_lgb_trees[w])
        m = lgb.LGBMRegressor(**p)
        m.fit(X_gbdt_all, y_gbdt_all[:, w], categorical_feature=["region_id"])
        final_lgb.append(m)
    lgb_test = np.clip(np.column_stack([m.predict(X_gbdt_test) for m in final_lgb]), 0, 5)

    # Test DL predictions (already built X_dl_te from train+test sequences)
    X_dl_tr_all, y_dl_tr_all = _build_seqs(train_weekly, DL_FEATURES, set(), WINDOW_STRIDE)
    X_dl_tr_all, _, X_dl_te = _normalise(X_dl_tr_all, X_dl_va, X_dl_te)

    dl_test_preds: dict[str, np.ndarray] = {}
    if TORCH_AVAILABLE:
        for name, model in dl_models.items():
            # Retrain on all data
            model_all = type(model)(n_dl_feat).to(DEVICE)
            dummy_va  = X_dl_tr_all[:min(200, len(X_dl_tr_all))]
            dummy_y   = y_dl_tr_all[:min(200, len(y_dl_tr_all))]
            model_all, _ = _train_model(model_all, X_dl_tr_all, y_dl_tr_all,
                                        dummy_va, dummy_y)
            dl_test_preds[name] = _predict(model_all, X_dl_te)

    # Align LGB and DL predictions to same region order
    lgb_region_order = X_gbdt_test["region_id"].values
    dl_region_set    = dict(zip(te_regions, range(len(te_regions))))
    # Reorder DL preds to match LGB region order
    dl_reordered: list[np.ndarray] = []
    for name in dl_preds_val:
        p = dl_test_preds[name]
        idx = [dl_region_set.get(r, 0) for r in lgb_region_order]
        dl_reordered.append(p[idx])

    all_test = [lgb_test] + dl_reordered
    test_blend = sum(w * p for w, p in zip(best_w, all_test))
    test_blend = np.clip(test_blend, 0, 5)

    # Build submission
    sub = pd.DataFrame({"region_id": lgb_region_order})
    for k in range(N_WEEKS):
        sub[f"pred_week{k+1}"] = test_blend[:, k]

    template = pd.read_csv(SAMPLE_SUB)
    sub = template[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(N_WEEKS)]:
        sub[col] = sub[col].fillna(0.0)

    sub.to_csv(OUT_PATH, index=False)

    mins = (time.time() - t0) / 60
    print(f"\n{'='*66}")
    print(f"  Gespeichert: {OUT_PATH}")
    print(f"  Zeilen: {len(sub):,}  |  Gesamtzeit: {mins:.1f} Min.")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
