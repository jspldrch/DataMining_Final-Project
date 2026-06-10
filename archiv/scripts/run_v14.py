"""
run_v14.py  –  Parallel Transformer + MLP with FC Fusion

Architecture:
  Two parallel branches, fused by a fully connected layer:

  Branch A – WeatherTransformer:
    Input:  raw weekly weather sequence (last SEQ_LEN weeks × 14 variables)
    Model:  embed(14→d) + sinusoidal pos encoding
            → 2× TransformerEncoderLayer (d_model=64, 4 heads)
            → global mean pool
    Output: temporal embedding (B, 64)

  Branch B – FeatureMLP:
    Input:  135 engineered features (lags, rolling windows, drought indices)
    Model:  Linear(135→256) → BN → GELU → Dropout
            → Linear(256→128) → BN → GELU
            → Linear(128→64)
    Output: tabular embedding (B, 64)

  Fusion:
    Concat([temporal, tabular]) → (B, 128)
    → Linear(128→64) → GELU → Dropout(0.2)
    → Linear(64→5) → clamp[0,5]

Key improvements over v13:
  - Simpler, more direct FC fusion (no attention gate)
  - Faster sequence building (pre-pad per region, numpy slicing)
  - OneCycleLR scheduler (faster convergence)
  - Fixes wasteful val-sequence building bug from v13
  - Blended with LGB (proven strong baseline)

Setup:  pip install torch
Usage:  python scripts/run_v14.py
Output: outputs/submission_v14.csv
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("WARNING: PyTorch not found. DL disabled. Install: pip install torch")

# ─── Paths (auto-detects Kaggle vs local) ────────────────────────────────────
_ON_KAGGLE = Path("/kaggle/input").exists()

def _find_kaggle(filename: str) -> Path:
    """Search recursively under /kaggle/input for a file by name."""
    matches = list(Path("/kaggle/input").rglob(filename))
    if not matches:
        # Print all available files to help debug
        all_files = list(Path("/kaggle/input").rglob("*"))
        print(f"  Available files under /kaggle/input:")
        for f in all_files:
            print(f"    {f}")
        raise FileNotFoundError(
            f"'{filename}' not found under /kaggle/input. "
            f"Check dataset names above."
        )
    return matches[0]

if _ON_KAGGLE:
    TRAIN_NPZ  = _find_kaggle("train.npz")
    TEST_NPZ   = _find_kaggle("test.npz")
    _sub_candidates = list(Path("/kaggle/input").rglob("sample_submission.csv"))
    SAMPLE_SUB = _sub_candidates[0] if _sub_candidates else Path("/kaggle/input/sample_submission.csv")
    OUT_DIR    = Path("/kaggle/working")
    OUT_PATH   = OUT_DIR / "submission_v14.csv"
    print(f"Kaggle paths found:  train={TRAIN_NPZ}  test={TEST_NPZ}")
else:
    ROOT       = Path(__file__).parent.parent
    TRAIN_NPZ  = ROOT / "data" / "train.npz"
    TEST_NPZ   = ROOT / "data" / "test.npz"
    SAMPLE_SUB = ROOT / "resources" / "sample_submission.csv"
    OUT_DIR    = ROOT / "outputs"
    OUT_PATH   = OUT_DIR / "submission_v14.csv"

OUT_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
QUICK_MODE = False

RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0
WINDOW_STRIDE   = 1 if not QUICK_MODE else 4
N_LGB_EST       = 1000 if not QUICK_MODE else 400

SEQ_LEN      = 26      # weeks of history for transformer
D_MODEL      = 64      # embedding dimension for both branches
N_HEADS      = 4       # transformer attention heads
N_TF_LAYERS  = 2       # transformer encoder layers
DL_EPOCHS    = 60 if not QUICK_MODE else 20
DL_BATCH     = 512
DL_LR        = 3e-4
DL_WD        = 1e-3
DL_PATIENCE  = 15      # early stopping patience
MAX_DL_TRAIN = 350_000 # max training samples for DL

# ─── Feature config (v7/v12 base) ─────────────────────────────────────────────
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
N_WEATHER = len(WEATHER_COLS)

LAG_COLS  = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre", "humidity"]
LAGS      = [1, 3, 7, 14, 21]
ROLL_COLS = ["prec", "wind", "tmp", "humidity"]
ROLL_WINS = [7, 14, 30, 60, 90, 180]

LGB_PARAMS = dict(
    objective="regression", metric="mae",
    n_estimators=N_LGB_EST, learning_rate=0.04, num_leaves=127,
    min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1,
)

NUM_FEATURES: list[str] = []


# ─── NPZ loader ───────────────────────────────────────────────────────────────

def load_npz(npz_path: Path) -> pd.DataFrame:
    """Load a .npz file created by convert_to_npz.py and return a DataFrame."""
    d            = np.load(npz_path, allow_pickle=True)
    region_names = d["region_names"]
    df = pd.DataFrame({
        "region_id": region_names[d["region_id"]],
        "year":      d["year"].astype(np.int32),
        "month":     d["month"].astype(np.int32),
        "day":       d["day"].astype(np.int32),
    })
    # Reconstruct date string (used by _parse_dates)
    df["date"] = (df["year"].astype(str) + "-"
                  + df["month"].astype(str).str.zfill(2) + "-"
                  + df["day"].astype(str).str.zfill(2))
    for col in WEATHER_COLS:
        if col in d:
            df[col] = d[col].astype(np.float32)
    if "score" in d:
        df["score"] = d["score"].astype(np.float32)
    return df


def build_feature_list() -> list[str]:
    lags  = [f"{c}_lag{l}"         for c in LAG_COLS  for l in LAGS]
    rolls = [f"{c}_roll{w}_{s}"    for c in ROLL_COLS for w in ROLL_WINS for s in ("mean","std","max")]
    cal   = ["month_sin","month_cos","day_sin","day_cos","week_sin","week_cos"]
    drt   = ["prec_deficit_90d","prec_trend_30d","humidity_deficit_90d",
             "tmp_anomaly_90d","heat_drought_idx","dry_days_14d","dry_days_30d"]
    return WEATHER_COLS + lags + rolls + cal + drt + ["regional_mean_score"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_dates(df: pd.DataFrame) -> None:
    p = df["date"].str.split("-", expand=True)
    df["year"]    = p[0].astype(np.int32)
    df["month"]   = p[1].astype(np.int32)
    df["day"]     = p[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]

def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} Min." if s >= 60 else f"{s:.0f}s"

def mae_np(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(yp, 0, 5) - yt)))

def show_mae(name: str, yt: np.ndarray, yp: np.ndarray) -> None:
    print(f"  {name:<50s}  MAE = {mae_np(yt, yp):.4f}")


# ─── Feature engineering (identical to v12) ────────────────────────────────────

def region_features(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = te.copy(); te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc: dict = {}
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
            r = prior.rolling(w, min_periods=max(3,w//10))
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

def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
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
        nw = n-5
        yr = np.lib.stride_tricks.sliding_window_view(sc[1:], 5)[:nw]
        idx = list(range(0, nw, stride))
        if (nw-1) not in idx: idx.append(nw-1)
        Xp.append(Xn[idx]); yp.append(yr[idx]); rp.extend([region]*len(idx))
    Xdf = pd.DataFrame(np.vstack(Xp).astype(np.float32), columns=num_features)
    Xdf["region_id"] = pd.Categorical(rp)
    return Xdf, np.vstack(yp).astype(np.float32)

def build_val_samples(weekly, val_regions, num_features):
    Xp, yp, rp = [], [], []
    for region in val_regions:
        g = weekly.loc[weekly["region_id"]==region].sort_values("ordinal")
        if len(g) < 6: continue
        Xp.append(g.iloc[-6][num_features].to_numpy(np.float32))
        yp.append(g.iloc[-5:]["score"].to_numpy(np.float32))
        rp.append(region)
    Xdf = pd.DataFrame(np.vstack(Xp), columns=num_features)
    Xdf["region_id"] = pd.Categorical(rp)
    return Xdf, np.vstack(yp)


# ─── Sequence builder (efficient pre-pad approach) ────────────────────────────

def build_sequences(
    weekly: pd.DataFrame,
    num_features: list[str],
    weather_mean: np.ndarray,
    weather_std: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    skip_regions: set | None,
    stride: int = 1,
    max_samples: int | None = None,
    last_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (sequences, features, targets) for DL training/inference.

    Pre-pads each region's weather with SEQ_LEN-1 zeros so every window
    maps to a simple slice — no conditional logic needed per window.

    last_only=True: only extract the last window per region (for val/test).
    """
    all_seq, all_feat, all_y = [], [], []

    for region, g in weekly.groupby("region_id", sort=False):
        if skip_regions and region in skip_regions:
            continue
        g   = g.sort_values("ordinal").reset_index(drop=True)
        n   = len(g)
        if n < 6:
            continue

        # Normalised weather matrix (n, N_WEATHER)
        w_norm = ((g[WEATHER_COLS].fillna(0).values.astype(np.float32) - weather_mean) / weather_std)
        # Pre-pad: prepend SEQ_LEN-1 zero rows → every window i is just padded[i:i+SEQ_LEN]
        padded = np.concatenate([np.zeros((SEQ_LEN-1, N_WEATHER), dtype=np.float32), w_norm], axis=0)

        # Normalised tabular features (n, n_feat)
        f_norm = ((g[num_features].fillna(0).values.astype(np.float32) - feat_mean) / feat_std)

        scores = g["score"].values.astype(np.float32)
        n_win  = n - 5

        if last_only:
            indices = [n_win - 1]   # last valid window only
        else:
            indices = list(range(0, n_win, stride))
            if (n_win-1) not in indices:
                indices.append(n_win-1)

        for i in indices:
            tgt = scores[i+1:i+6]
            if np.any(np.isnan(tgt)):
                continue
            all_seq.append(padded[i: i + SEQ_LEN])   # (SEQ_LEN, N_WEATHER)
            all_feat.append(f_norm[i])                # (n_feat,)
            all_y.append(tgt)                         # (5,)

    seqs  = np.stack(all_seq,  axis=0).astype(np.float32)   # (N, SEQ_LEN, N_WEATHER)
    feats = np.stack(all_feat, axis=0).astype(np.float32)   # (N, n_feat)
    ys    = np.stack(all_y,    axis=0).astype(np.float32)   # (N, 5)

    if max_samples and not last_only and len(seqs) > max_samples:
        rng  = np.random.default_rng(RANDOM_STATE)
        idx  = rng.choice(len(seqs), max_samples, replace=False)
        seqs, feats, ys = seqs[idx], feats[idx], ys[idx]

    return seqs, feats, ys


# ─── DL: Dataset ─────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class DroughtDataset(Dataset):
        def __init__(self, seqs: np.ndarray, feats: np.ndarray, ys: np.ndarray):
            self.seqs  = torch.from_numpy(seqs)
            self.feats = torch.from_numpy(feats)
            self.ys    = torch.from_numpy(ys)
        def __len__(self): return len(self.ys)
        def __getitem__(self, i): return self.seqs[i], self.feats[i], self.ys[i]


# ─── DL: Model ────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    def _sinusoidal_pos_enc(seq_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe  = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, seq_len, d_model)


    class ParallelDroughtModel(nn.Module):
        """
        Branch A: Transformer on weather sequences
        Branch B: MLP on engineered tabular features
        Fusion:   Concat(A, B) → FC → 5 predictions
        """

        def __init__(self, n_features: int, d_model: int = D_MODEL,
                     n_heads: int = N_HEADS, n_layers: int = N_TF_LAYERS):
            super().__init__()

            # ── Branch A: Temporal Transformer ───────────────────────────────
            self.seq_embed = nn.Linear(N_WEATHER, d_model)
            self.register_buffer("pos_enc", _sinusoidal_pos_enc(SEQ_LEN, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            self.transformer   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.temporal_norm = nn.LayerNorm(d_model)

            # ── Branch B: Feature MLP ─────────────────────────────────────────
            self.feat_mlp = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, d_model),
            )

            # ── Fusion: Concat → FC ───────────────────────────────────────────
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(d_model, 32),
                nn.GELU(),
                nn.Linear(32, 5),
            )

        def forward(self, seq: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
            # Branch A
            x_t = self.seq_embed(seq) + self.pos_enc          # (B, L, d)
            x_t = self.transformer(x_t)
            x_t = self.temporal_norm(x_t).mean(dim=1)         # (B, d)

            # Branch B
            x_f = self.feat_mlp(feat)                          # (B, d)

            # Concat + FC
            x   = torch.cat([x_t, x_f], dim=1)                # (B, 2d)
            out = self.fusion(x)                               # (B, 5)
            return torch.clamp(out, 0.0, 5.0)


# ─── DL: Training loop ────────────────────────────────────────────────────────

def train_dl(
    tr_seqs: np.ndarray, tr_feats: np.ndarray, tr_ys: np.ndarray,
    va_seqs: np.ndarray, va_feats: np.ndarray, va_ys: np.ndarray,
    n_features: int,
) -> "ParallelDroughtModel | None":
    if not TORCH_AVAILABLE:
        return None

    print(f"  Device={DEVICE}  train={len(tr_ys):,}  val={len(va_ys)}")
    model     = ParallelDroughtModel(n_features).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DL_LR, weight_decay=DL_WD)
    steps     = math.ceil(len(tr_ys) / DL_BATCH) * DL_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=DL_LR, total_steps=steps, pct_start=0.1, final_div_factor=100,
    )
    criterion = nn.HuberLoss(delta=1.0)

    loader = DataLoader(
        DroughtDataset(tr_seqs, tr_feats, tr_ys),
        batch_size=DL_BATCH, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    va_s = torch.from_numpy(va_seqs).to(DEVICE)
    va_f = torch.from_numpy(va_feats).to(DEVICE)
    va_y = torch.from_numpy(va_ys)

    best_mae, best_state, patience = 999.0, None, 0

    for epoch in range(1, DL_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for sb, fb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(sb.to(DEVICE), fb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(va_s, va_f).cpu().numpy()
        val_mae = mae_np(va_y.numpy(), val_pred)

        if epoch % 5 == 0 or epoch <= 3:
            print(f"    Epoch {epoch:3d}/{DL_EPOCHS}  "
                  f"loss={total_loss/len(loader):.4f}  val_mae={val_mae:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_mae < best_mae:
            best_mae  = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience  += 1
            if patience >= DL_PATIENCE:
                print(f"    Early stop @ epoch {epoch}  best val MAE={best_mae:.4f}")
                break

    model.load_state_dict(best_state)
    return model


def predict_dl(model, seqs: np.ndarray, feats: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 2048):
            s = torch.from_numpy(seqs[i:i+2048]).to(DEVICE)
            f = torch.from_numpy(feats[i:i+2048]).to(DEVICE)
            out.append(model(s, f).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ─── LGB training (v12 identical) ─────────────────────────────────────────────

def train_lgb(X_tr, y_tr, X_va, y_va, n_trees=None):
    models = []
    for week in range(5):
        n = (n_trees[week] if n_trees else None) or LGB_PARAMS["n_estimators"]
        p = dict(LGB_PARAMS, random_state=RANDOM_STATE+week, n_estimators=n)
        m = lgb.LGBMRegressor(**p)
        kw: dict = dict(categorical_feature=["region_id"])
        if X_va is not None:
            kw["eval_set"]    = [(X_va, y_va[:, week].ravel())]
            kw["eval_metric"] = "mae"
            kw["callbacks"]   = [lgb.early_stopping(50, verbose=False)]
        m.fit(X_tr, y_tr[:, week].ravel(), **kw)
        models.append(m)
    return models

def predict_lgb(models, X) -> np.ndarray:
    return np.clip(np.column_stack([m.predict(X) for m in models]), 0, 5).astype(np.float32)

def blend_search(yt, pa, pb):
    best_mae, best_a = 999.0, 0.5
    for a in [round(x*0.05, 2) for x in range(1, 20)]:
        m = mae_np(yt, a*pa + (1-a)*pb)
        if m < best_mae:
            best_mae, best_a = m, a
    return best_a, best_mae


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 68)
    print("  Natural Disaster Severity Prediction  -  run_v14.py")
    print(f"  Mode: {'QUICK' if QUICK_MODE else 'FULL'}  stride={WINDOW_STRIDE}  LGB={N_LGB_EST}")
    if TORCH_AVAILABLE:
        print(f"  DL: {D_MODEL}d  {N_HEADS}h  {N_TF_LAYERS}L  seq={SEQ_LEN}w  "
              f"epochs={DL_EPOCHS}  device={DEVICE}")
    print(f"  Features: {len(NUM_FEATURES)}")
    print("=" * 68)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    print(f"   Mode: {'Kaggle NPZ' if _ON_KAGGLE else 'Local NPZ'}")
    train_raw = load_npz(TRAIN_NPZ)
    test_raw  = load_npz(TEST_NPZ)
    # date already parsed by load_npz; call _parse_dates only if ordinal missing
    if "ordinal" not in train_raw.columns:
        _parse_dates(train_raw)
    if "ordinal" not in test_raw.columns:
        _parse_dates(test_raw)
    if "score" not in train_raw.columns:
        train_raw["score"] = np.nan
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    print(f"   Train={len(train_raw):,}  Test={len(test_raw):,}  Regions={len(regions)}  [{elapsed(t0)}]")
    region_means = train_raw.groupby("region_id")["score"].mean()

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/7] Feature engineering ...")
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
    print(f"   Done  [{elapsed(t0)}]")

    # ── 3. Weekly aggregation ─────────────────────────────────────────────────
    print("\n[3/7] Weekly aggregation ...")
    labeled = train_feat[train_feat["score"].notna()].copy()
    wk_parts = [daily_to_weekly(g) for _, g in labeled.groupby("region_id", sort=False)]
    train_weekly = pd.concat(wk_parts, ignore_index=True)
    del labeled
    print(f"   {len(train_weekly):,} weekly rows  [{elapsed(t0)}]")

    # ── 4. Train/val split ────────────────────────────────────────────────────
    print("\n[4/7] Train/val split ...")
    rng         = np.random.default_rng(RANDOM_STATE)
    all_reg     = sorted(train_weekly["region_id"].unique())
    val_regions = set(rng.choice(all_reg, max(1, int(len(all_reg)*VAL_REGION_FRAC)), replace=False))
    X_tr, y_tr  = build_sliding_windows(train_weekly, val_regions, NUM_FEATURES, WINDOW_STRIDE)
    X_va, y_va  = build_val_samples(train_weekly, sorted(val_regions), NUM_FEATURES)
    print(f"   Train={len(X_tr):,}  Val={len(X_va)}  [{elapsed(t0)}]")

    last_sc = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    persist  = np.column_stack([last_sc.reindex(sorted(val_regions)).fillna(0).values]*5)
    show_mae("Persistence baseline", y_va, persist)

    # ── 5. LGB ────────────────────────────────────────────────────────────────
    print("\n[5/7] Training LightGBM ...")
    lgb_models = train_lgb(X_tr, y_tr, X_va, y_va)
    lgb_val    = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (val)", y_va, lgb_val)

    # ── 6. DL ─────────────────────────────────────────────────────────────────
    dl_model = None
    if TORCH_AVAILABLE:
        print("\n[6/7] Building sequences + training DL model ...")

        # Normalization stats from training data only
        w_mean = train_weekly[WEATHER_COLS].mean().values.astype(np.float32)
        w_std  = train_weekly[WEATHER_COLS].std().clip(lower=1e-8).values.astype(np.float32)
        f_mean = X_tr[NUM_FEATURES].mean().values.astype(np.float32)
        f_std  = X_tr[NUM_FEATURES].std().clip(lower=1e-8).values.astype(np.float32)

        print(f"  Building training sequences (stride={WINDOW_STRIDE}, max={MAX_DL_TRAIN:,}) ...")
        tr_seqs, tr_feats, tr_ys = build_sequences(
            train_weekly, NUM_FEATURES, w_mean, w_std, f_mean, f_std,
            skip_regions=val_regions, stride=WINDOW_STRIDE, max_samples=MAX_DL_TRAIN,
        )

        print(f"  Building val sequences (last window per region) ...")
        va_seqs, va_feats, va_ys = build_sequences(
            train_weekly, NUM_FEATURES, w_mean, w_std, f_mean, f_std,
            skip_regions=set(all_reg)-val_regions, last_only=True,
        )
        # va_ys from build_sequences = same last 5 weeks as y_va from build_val_samples
        print(f"  Train seqs={tr_seqs.shape}  Val seqs={va_seqs.shape}  [{elapsed(t0)}]")

        dl_model = train_dl(tr_seqs, tr_feats, tr_ys, va_seqs, va_feats, y_va, len(NUM_FEATURES))

        if dl_model is not None:
            dl_val = predict_dl(dl_model, va_seqs, va_feats)
            show_mae("DL Parallel Transformer (val)", y_va, dl_val)
            lgb_w, blend_mae = blend_search(y_va, lgb_val, dl_val)
            dl_w = round(1-lgb_w, 2)
            print(f"  Blend: LGB={lgb_w:.2f}  DL={dl_w:.2f}  val MAE={blend_mae:.4f}")
        else:
            lgb_w, dl_w = 1.0, 0.0
    else:
        lgb_w, dl_w = 1.0, 0.0
        print("\n[6/7] DL skipped (no torch).")

    # ── 7. Final training + test predictions ──────────────────────────────────
    print("\n[7/7] Final training + predictions ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, WINDOW_STRIDE)
    n_lgb  = [int(getattr(m,"best_iteration_",None) or N_LGB_EST) for m in lgb_models]
    final_lgb = train_lgb(X_all, y_all, None, None, n_lgb)

    X_test = (
        test_feat.sort_values(["region_id","ordinal"])
        .groupby("region_id", sort=False).tail(1)
        [["region_id"] + NUM_FEATURES].reset_index(drop=True)
    )
    X_test["region_id"] = X_test["region_id"].astype("category")
    lgb_test = predict_lgb(final_lgb, X_test)

    if dl_model is not None and dl_w > 0:
        # Normalization for final DL (use all training data)
        w_mean2 = train_weekly[WEATHER_COLS].mean().values.astype(np.float32)
        w_std2  = train_weekly[WEATHER_COLS].std().clip(lower=1e-8).values.astype(np.float32)
        f_mean2 = X_all[NUM_FEATURES].mean().values.astype(np.float32)
        f_std2  = X_all[NUM_FEATURES].std().clip(lower=1e-8).values.astype(np.float32)

        print("  Retraining DL on all regions ...")
        all_seqs, all_feats, all_ys = build_sequences(
            train_weekly, NUM_FEATURES, w_mean2, w_std2, f_mean2, f_std2,
            skip_regions=None, stride=WINDOW_STRIDE, max_samples=MAX_DL_TRAIN,
        )
        final_dl = train_dl(all_seqs, all_feats, all_ys, va_seqs, va_feats, y_va, len(NUM_FEATURES))

        # Test sequences: last window per region
        print("  Building test sequences ...")
        test_seqs, test_feats, _ = build_sequences(
            train_weekly, NUM_FEATURES, w_mean2, w_std2, f_mean2, f_std2,
            skip_regions=None, last_only=True,
        )
        # Align with X_test region order
        test_region_order = X_test["region_id"].tolist()
        weekly_last = {
            r: i for i, r in enumerate(
                g.sort_values("ordinal").iloc[-1]["region_id"]
                for _, g in train_weekly.groupby("region_id", sort=False)
            )
        }
        # Rebuild test sequences in correct order
        region_to_idx = {}
        idx_counter = 0
        for region, g in train_weekly.groupby("region_id", sort=False):
            region_to_idx[region] = idx_counter
            idx_counter += 1
        # Since build_sequences(last_only=True) iterates in groupby order,
        # regions in test_seqs match the groupby order, not X_test order.
        # Rebuild for exact X_test order:
        ts_list, tf_list = [], []
        for region in test_region_order:
            g = train_weekly.loc[train_weekly["region_id"]==region].sort_values("ordinal")
            n_g = len(g)
            i   = n_g - 1
            wn  = ((g[WEATHER_COLS].fillna(0).values.astype(np.float32) - w_mean2) / w_std2)
            fn  = ((g[NUM_FEATURES].fillna(0).values.astype(np.float32) - f_mean2) / f_std2)
            pad = np.concatenate([np.zeros((SEQ_LEN-1, N_WEATHER), np.float32), wn])
            ts_list.append(pad[i: i+SEQ_LEN])
            tf_list.append(fn[i])
        ts_arr = np.stack(ts_list).astype(np.float32)
        tf_arr = np.stack(tf_list).astype(np.float32)

        # Use tabular features from test_feat (includes test-period weather in rolling/lag)
        tf_tab = ((X_test[NUM_FEATURES].fillna(0).values.astype(np.float32) - f_mean2) / f_std2)

        dl_test    = predict_dl(final_dl, ts_arr, tf_tab)
        test_preds = lgb_w * lgb_test + dl_w * dl_test
    else:
        test_preds = lgb_test

    sub = pd.DataFrame({"region_id": X_test["region_id"].values})
    for k in range(5):
        sub[f"pred_week{k+1}"] = test_preds[:, k]
    template = pd.read_csv(SAMPLE_SUB)
    sub = template[["region_id"]].merge(sub, on="region_id", how="left")
    for col in [f"pred_week{k+1}" for k in range(5)]:
        sub[col] = sub[col].fillna(0.0)
    sub.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*68}")
    print(f"  Saved: {OUT_PATH}")
    print(f"  Rows={len(sub):,}  Total={elapsed(t0)}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
