"""
run_v13.py  –  Hybrid Transformer + MLP Drought Predictor

Architecture (inspired by friend's approach):
  1. WeatherTransformer  – transformer encoder on the last 26 weeks of raw weather
                           sequences. Learns temporal patterns (droughts build over time).
  2. TabularMLP          – deep MLP on 135 engineered features (same as v12: lags,
                           rolling windows, drought indices, regional mean).
  3. AttentionGate       – learned soft weighting: how much to trust the temporal
                           sequence vs the aggregated tabular features (per sample).
  4. OutputHead          – predicts drought scores for all 5 future weeks at once.
  5. LightGBM            – best tree model (v12 basis) blended with DL output.

Why this can beat pure GBM:
  - Transformer sees raw temporal sequences, not just hand-crafted lags → captures
    complex multi-scale patterns (drought onset, recovery, oscillations)
  - Attention gate adapts per sample: some regions/times rely more on the recent
    sequence, others on the long-run aggregated statistics
  - DL + GBM blend: very different inductive biases → low prediction correlation →
    meaningful variance reduction

Setup:
    pip install torch

Usage:
    python scripts/run_v13.py
Output: outputs/submission_v13.csv
Estimated runtime: ~60-90 min (GPU: ~30 min, CPU: ~90 min)
"""

from __future__ import annotations

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
    print("PyTorch not found. Install: pip install torch")
    print("Falling back to LGB-only mode.")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV  = DATA_DIR / "train.csv"
TEST_CSV   = DATA_DIR / "test.csv"
SAMPLE_SUB = ROOT / "resources" / "sample_submission.csv"
OUT_PATH   = OUT_DIR / "submission_v13.csv"

# ─── Mode ─────────────────────────────────────────────────────────────────────
QUICK_MODE = False

RANDOM_STATE    = 42
VAL_REGION_FRAC = 0.20
WEEK_BUCKET     = 7
DRY_THRESHOLD   = 1.0

WINDOW_STRIDE = 1 if not QUICK_MODE else 4
N_ESTIMATORS  = 1000 if not QUICK_MODE else 400

# DL config
SEQ_LEN          = 26     # weeks of history for transformer (6 months)
D_MODEL          = 64     # transformer hidden dim
N_HEADS          = 4      # attention heads
N_TRANSFORMER    = 2      # transformer encoder layers
DL_EPOCHS        = 100 if not QUICK_MODE else 30
DL_BATCH         = 512
DL_LR            = 3e-4
DL_WD            = 1e-3
DL_PATIENCE      = 20     # early stopping patience (epochs)
MAX_DL_TRAIN     = 400_000  # subsample for DL training (memory/speed)

# ─── Feature config (identical to v12/v7) ─────────────────────────────────────
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
    objective="regression",
    metric="mae",
    n_estimators=N_ESTIMATORS,
    learning_rate=0.04,
    num_leaves=127,
    min_child_samples=60,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)

NUM_FEATURES: list[str] = []
N_WEATHER = len(WEATHER_COLS)


# ─── Feature list ─────────────────────────────────────────────────────────────

def build_feature_list() -> list[str]:
    lag_names  = [f"{c}_lag{lag}"      for c in LAG_COLS  for lag in LAGS]
    roll_names = [
        f"{col}_roll{w}_{stat}"
        for col in ROLL_COLS for w in ROLL_WINS for stat in ("mean", "std", "max")
    ]
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos", "week_sin", "week_cos"]
    drought  = [
        "prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
        "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d",
    ]
    return WEATHER_COLS + lag_names + roll_names + calendar + drought + ["regional_mean_score"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_dates_inplace(df: pd.DataFrame) -> None:
    parts = df["date"].str.split("-", expand=True)
    df["year"]  = parts[0].astype(np.int32)
    df["month"] = parts[1].astype(np.int32)
    df["day"]   = parts[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]


def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} Min." if s >= 60 else f"{s:.0f}s"


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(y_pred, 0, 5) - y_true)))


def show_mae(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"  {name:<52s}  MAE = {mae_np(y_true, y_pred):.4f}")


# ─── Feature engineering (identical to v12) ────────────────────────────────────

def compute_region_features(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    new_cols: dict[str, np.ndarray] = {}

    new_cols["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["day_sin"]   = np.sin(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    new_cols["day_cos"]   = np.cos(2 * np.pi * panel["day"]   / 31).astype(np.float32)
    week_of_year = (panel["ordinal"] // 7) % 52
    new_cols["week_sin"]  = np.sin(2 * np.pi * week_of_year / 52).astype(np.float32)
    new_cols["week_cos"]  = np.cos(2 * np.pi * week_of_year / 52).astype(np.float32)

    for col in LAG_COLS:
        s = panel[col]
        for lag in LAGS:
            new_cols[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)

    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=max(3, w // 10))
            new_cols[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            new_cols[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)
            new_cols[f"{col}_roll{w}_max"]  = r.max().astype(np.float32)

    prec_p = panel["prec"].shift(1)
    new_cols["prec_deficit_90d"] = (
        prec_p.rolling(90, min_periods=30).mean() - prec_p.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    p7   = prec_p.rolling(7, min_periods=3).mean()
    p30  = prec_p.rolling(30, min_periods=10).mean()
    p30s = prec_p.rolling(30, min_periods=10).std().clip(lower=0.01)
    new_cols["prec_trend_30d"] = ((p7 - p30) / p30s).astype(np.float32)

    hum_p = panel["humidity"].shift(1)
    new_cols["humidity_deficit_90d"] = (
        hum_p.rolling(90, min_periods=30).mean() - hum_p.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    tmp_p   = panel["tmp"].shift(1)
    tmp_anom = (tmp_p.rolling(90, min_periods=30).mean() - tmp_p.rolling(365, min_periods=60).mean()).astype(np.float32)
    new_cols["tmp_anomaly_90d"]  = tmp_anom
    new_cols["heat_drought_idx"] = (new_cols["prec_deficit_90d"] * tmp_anom.clip(lower=0)).astype(np.float32)

    dry = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    new_cols["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    new_cols["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)

    panel = pd.concat([panel, pd.DataFrame(new_cols, index=panel.index)], axis=1)
    n_tr = len(tr)
    return panel.iloc[:n_tr].copy(), panel.iloc[n_tr:].copy()


# ─── Dataset assembly ─────────────────────────────────────────────────────────

def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    week = df["ordinal"] // WEEK_BUCKET
    return df.loc[df.groupby(week, sort=False)["ordinal"].idxmax()].reset_index(drop=True)


def build_sliding_windows(weekly, skip_regions, num_features, stride=1):
    X_parts, y_parts, r_parts = [], [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions:
            continue
        g = g.sort_values("ordinal")
        scores = g["score"].to_numpy(dtype=np.float32)
        X_num  = g[num_features].to_numpy(dtype=np.float32)
        n = len(g)
        if n < 6:
            continue
        n_win = n - 5
        y_reg = np.lib.stride_tricks.sliding_window_view(scores[1:], 5)[:n_win]
        idx   = list(range(0, n_win, stride))
        if (n_win - 1) not in idx:
            idx.append(n_win - 1)
        X_parts.append(X_num[idx])
        y_parts.append(y_reg[idx])
        r_parts.extend([region] * len(idx))
    X_df = pd.DataFrame(np.vstack(X_parts).astype(np.float32), columns=num_features)
    X_df["region_id"] = pd.Categorical(r_parts)
    return X_df, np.vstack(y_parts).astype(np.float32)


def build_val_samples(weekly, val_regions, num_features):
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


# ─── DL: sequence dataset builder ─────────────────────────────────────────────

def build_dl_sequences(
    weekly: pd.DataFrame,
    num_features: list[str],
    weather_mean: np.ndarray,
    weather_std: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    skip_regions: set | None = None,
    stride: int = 1,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      seqs:  (N, SEQ_LEN, N_WEATHER) normalised weather sequences
      feats: (N, n_features) normalised tabular features
      ys:    (N, 5) target scores
    """
    all_seqs, all_feats, all_ys = [], [], []

    for region, g in weekly.groupby("region_id", sort=False):
        if skip_regions and region in skip_regions:
            continue
        g = g.sort_values("ordinal").reset_index(drop=True)
        n = len(g)
        if n < 6:
            continue

        # Raw weather for this region (normalised)
        weather = ((g[WEATHER_COLS].fillna(0).values.astype(np.float32) - weather_mean) / weather_std)
        feat_mat = ((g[num_features].fillna(0).values.astype(np.float32) - feat_mean) / feat_std)
        scores   = g["score"].values.astype(np.float32)

        n_win = n - 5
        indices = list(range(0, n_win, stride))
        if (n_win - 1) not in indices:
            indices.append(n_win - 1)

        for i in indices:
            targets = scores[i + 1: i + 6]
            if np.any(np.isnan(targets)):
                continue

            # Sequence: last SEQ_LEN weeks ending at i
            start = max(0, i - SEQ_LEN + 1)
            raw_seq = weather[start: i + 1]            # variable length
            pad_len = SEQ_LEN - len(raw_seq)
            if pad_len > 0:
                raw_seq = np.concatenate([np.zeros((pad_len, N_WEATHER), dtype=np.float32), raw_seq])

            all_seqs.append(raw_seq)
            all_feats.append(feat_mat[i])
            all_ys.append(targets)

    seqs  = np.stack(all_seqs,  axis=0).astype(np.float32)
    feats = np.stack(all_feats, axis=0).astype(np.float32)
    ys    = np.stack(all_ys,    axis=0).astype(np.float32)

    # Subsample if needed
    if max_samples and len(seqs) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(seqs), max_samples, replace=False)
        seqs, feats, ys = seqs[idx], feats[idx], ys[idx]

    return seqs, feats, ys


class DroughtDataset(Dataset):
    def __init__(self, seqs: np.ndarray, feats: np.ndarray, ys: np.ndarray):
        self.seqs  = torch.from_numpy(seqs)
        self.feats = torch.from_numpy(feats)
        self.ys    = torch.from_numpy(ys)

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.feats[idx], self.ys[idx]


# ─── DL: model architecture ───────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class HybridDroughtModel(nn.Module):
        """
        WeatherTransformer + TabularMLP fused by a soft AttentionGate.

        seq  (B, L, n_weather) → transformer encoder → mean pool → (B, d_model)
        feat (B, n_feat)       → MLP                             → (B, d_model)
        gate = softmax(Linear(2*d_model, 2))                     → (B, 2)
        fused = gate[:,0] * temporal + gate[:,1] * tabular
        out  = head(fused)                                        → (B, 5)
        """

        def __init__(self, n_weather: int, n_features: int,
                     d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
            super().__init__()

            # Temporal branch
            self.seq_embed = nn.Linear(n_weather, d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.temporal_norm = nn.LayerNorm(d_model)

            # Tabular branch
            self.tabular_mlp = nn.Sequential(
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

            # Attention gate: learns how much to trust each branch per sample
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=-1),
            )

            # Output head
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 32),
                nn.GELU(),
                nn.Linear(32, 5),
            )

        def forward(self, seq: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
            # Temporal branch
            x_t = self.seq_embed(seq) + self.pos_embed   # (B, L, d)
            x_t = self.transformer(x_t)
            x_t = self.temporal_norm(x_t).mean(dim=1)    # global avg pool → (B, d)

            # Tabular branch
            x_f = self.tabular_mlp(feat)                  # (B, d)

            # Gate: soft weighting
            gate = self.gate(torch.cat([x_t, x_f], dim=1))  # (B, 2)
            fused = gate[:, 0:1] * x_t + gate[:, 1:2] * x_f

            return torch.clamp(self.head(fused), 0.0, 5.0)


# ─── DL: training ─────────────────────────────────────────────────────────────

def train_dl_model(
    train_seqs: np.ndarray, train_feats: np.ndarray, train_ys: np.ndarray,
    val_seqs:   np.ndarray, val_feats:   np.ndarray, val_ys:   np.ndarray,
    n_features: int,
) -> "HybridDroughtModel | None":
    if not TORCH_AVAILABLE:
        return None

    print(f"  Device: {DEVICE}  |  Train: {len(train_ys):,}  |  Val: {len(val_ys)}")

    model = HybridDroughtModel(N_WEATHER, n_features, D_MODEL, N_HEADS, N_TRANSFORMER).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DL_LR, weight_decay=DL_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DL_EPOCHS, eta_min=DL_LR * 0.1)
    criterion = nn.HuberLoss(delta=1.0)

    train_ds = DroughtDataset(train_seqs, train_feats, train_ys)
    loader   = DataLoader(train_ds, batch_size=DL_BATCH, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    val_seq_t  = torch.from_numpy(val_seqs).to(DEVICE)
    val_feat_t = torch.from_numpy(val_feats).to(DEVICE)
    val_y_t    = torch.from_numpy(val_ys)

    best_val_mae = 999.0
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, DL_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq_b, feat_b, y_b in loader:
            seq_b  = seq_b.to(DEVICE)
            feat_b = feat_b.to(DEVICE)
            y_b    = y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(seq_b, feat_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation MAE
        model.eval()
        with torch.no_grad():
            val_pred = model(val_seq_t, val_feat_t).cpu().numpy()
        val_mae = mae_np(val_y_t.numpy(), val_pred)

        if epoch % 10 == 0 or epoch <= 5:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch:3d}/{DL_EPOCHS}  loss={total_loss/len(loader):.4f}"
                  f"  val_mae={val_mae:.4f}  lr={lr_now:.5f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= DL_PATIENCE:
                print(f"    Early stopping at epoch {epoch} (best val MAE={best_val_mae:.4f})")
                break

    model.load_state_dict(best_state)
    print(f"  DL best val MAE: {best_val_mae:.4f}")
    return model


def predict_dl(model: "HybridDroughtModel", seqs: np.ndarray, feats: np.ndarray) -> np.ndarray:
    if not TORCH_AVAILABLE or model is None:
        return None
    model.eval()
    all_preds = []
    chunk = 1024
    with torch.no_grad():
        for i in range(0, len(seqs), chunk):
            s = torch.from_numpy(seqs[i:i+chunk]).to(DEVICE)
            f = torch.from_numpy(feats[i:i+chunk]).to(DEVICE)
            all_preds.append(model(s, f).cpu().numpy())
    return np.concatenate(all_preds, axis=0).astype(np.float32)


# ─── LGB training (identical to v12) ──────────────────────────────────────────

def train_lgb_models(X_tr, y_tr, X_va, y_va, n_trees_per_week=None):
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


def predict_lgb(models, X) -> np.ndarray:
    return np.clip(np.column_stack([m.predict(X) for m in models]), 0.0, 5.0).astype(np.float32)


def optimize_blend_2way(y_va, pred_a, pred_b):
    alphas = [round(x * 0.05, 2) for x in range(1, 20)]
    best_mae, best_a = 999.0, 0.5
    for a in alphas:
        m = mae_np(y_va, a * pred_a + (1 - a) * pred_b)
        if m < best_mae:
            best_mae, best_a = m, a
    return best_a, best_mae


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    global NUM_FEATURES
    NUM_FEATURES = build_feature_list()

    t0 = time.time()
    print("=" * 68)
    print("  Natural Disaster Severity Prediction  -  run_v13.py")
    mode_label = "QUICK" if QUICK_MODE else "FULL"
    print(f"  Mode: {mode_label}  |  stride={WINDOW_STRIDE}  LGB_est={N_ESTIMATORS}")
    if TORCH_AVAILABLE:
        print(f"  DL: {D_MODEL}d/{N_HEADS}h/{N_TRANSFORMER}L transformer  seq={SEQ_LEN}w  epochs={DL_EPOCHS}")
        print(f"  Device: {DEVICE}  |  Max DL train: {MAX_DL_TRAIN:,}")
    else:
        print("  DL: DISABLED (pip install torch)")
    print(f"  Features: {len(NUM_FEATURES)}")
    print("=" * 68)

    # 1. Load
    print("\n[1/7] Loading data ...")
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw  = pd.read_csv(TEST_CSV,  dtype=dtypes)
    _parse_dates_inplace(train_raw)
    _parse_dates_inplace(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    regions = train_raw["region_id"].unique()
    print(f"   Train: {len(train_raw):>10,}  Test: {len(test_raw):>8,}  Regions: {len(regions)}  [{elapsed(t0)}]")

    region_means = train_raw.groupby("region_id")["score"].mean()

    # 2. Feature engineering
    print("\n[2/7] Feature engineering ...")
    train_by_region = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    test_by_region  = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id",  sort=False)}
    del train_raw, test_raw

    all_tr, all_te = [], []
    n = len(regions)
    for i, region in enumerate(regions, 1):
        if i % 500 == 0 or i == n:
            print(f"   Region {i}/{n}  [{elapsed(t0)}]")
        tr_f, te_f = compute_region_features(
            train_by_region[region], test_by_region.get(region, pd.DataFrame())
        )
        all_tr.append(tr_f)
        all_te.append(te_f)

    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te

    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)
    print(f"   Done  [{elapsed(t0)}]")

    # 3. Weekly aggregation
    print("\n[3/7] Weekly aggregation ...")
    labeled = train_feat[train_feat["score"].notna()].copy()
    weekly_parts = []
    for region, g in labeled.groupby("region_id", sort=False):
        weekly_parts.append(daily_to_weekly(g))
    train_weekly = pd.concat(weekly_parts, ignore_index=True)
    del labeled
    print(f"   {len(train_weekly):,} weekly rows  [{elapsed(t0)}]")

    # 4. Train/val split (region holdout)
    print("\n[4/7] Train/val split ...")
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg = sorted(train_weekly["region_id"].unique())
    n_val   = max(1, int(len(all_reg) * VAL_REGION_FRAC))
    val_regions = set(rng.choice(all_reg, size=n_val, replace=False))

    X_tr, y_tr = build_sliding_windows(train_weekly, val_regions, NUM_FEATURES, stride=WINDOW_STRIDE)
    X_va, y_va = build_val_samples(train_weekly, sorted(val_regions), NUM_FEATURES)
    print(f"   Train: {len(X_tr):,}  Val: {len(X_va)}  [{elapsed(t0)}]")

    last_score = train_weekly.sort_values("ordinal").groupby("region_id")["score"].last()
    persist_va = np.column_stack([last_score.reindex(sorted(val_regions)).fillna(0).to_numpy()] * 5)
    show_mae("Persistence-Baseline", y_va, persist_va)

    # 5. LGB
    print("\n[5/7] Training LightGBM ...")
    lgb_models = train_lgb_models(X_tr, y_tr, X_va, y_va)
    lgb_val = predict_lgb(lgb_models, X_va)
    show_mae("LightGBM (val)", y_va, lgb_val)

    # 6. DL hybrid model
    dl_model = None
    dl_val   = None
    if TORCH_AVAILABLE:
        print(f"\n[6/7] Building DL sequences + training hybrid model ...")

        # Normalisation stats from training data
        weather_mean = train_weekly[WEATHER_COLS].mean().values.astype(np.float32)
        weather_std  = train_weekly[WEATHER_COLS].std().clip(lower=1e-8).values.astype(np.float32)
        feat_mean    = X_tr[NUM_FEATURES].mean().values.astype(np.float32)
        feat_std     = X_tr[NUM_FEATURES].std().clip(lower=1e-8).values.astype(np.float32)

        print("  Building training sequences ...")
        tr_seqs, tr_feats, tr_ys = build_dl_sequences(
            train_weekly, NUM_FEATURES, weather_mean, weather_std, feat_mean, feat_std,
            skip_regions=val_regions, stride=WINDOW_STRIDE, max_samples=MAX_DL_TRAIN,
        )
        print("  Building validation sequences ...")
        va_seqs, va_feats, _ = build_dl_sequences(
            train_weekly, NUM_FEATURES, weather_mean, weather_std, feat_mean, feat_std,
            skip_regions=set(all_reg) - val_regions,  # only val regions
            stride=1, max_samples=None,
        )
        # Use same val y as GBM val
        va_ys = y_va  # (n_val, 5) from build_val_samples — same regions/order

        print(f"  Seq train: {tr_seqs.shape}  val: {va_seqs.shape}  [{elapsed(t0)}]")

        # If val sequences have more samples than y_va (due to stride=1 over all windows)
        # use only the LAST window per val region (matches build_val_samples logic)
        # build_dl_sequences with val regions and stride=1 gives all windows — we only need last
        # Solution: rebuild with only the last window index per region
        print("  Extracting last-window val sequences ...")
        va_seqs_last, va_feats_last = [], []
        for region in sorted(val_regions):
            g = train_weekly.loc[train_weekly["region_id"] == region].sort_values("ordinal")
            if len(g) < 6:
                continue
            i = len(g) - 6  # same as build_val_samples: feature at -6
            weather_g = ((g[WEATHER_COLS].fillna(0).values.astype(np.float32) - weather_mean) / weather_std)
            feat_g    = ((g[NUM_FEATURES].fillna(0).values.astype(np.float32) - feat_mean) / feat_std)
            start = max(0, i - SEQ_LEN + 1)
            raw_seq = weather_g[start: i + 1]
            pad_len = SEQ_LEN - len(raw_seq)
            if pad_len > 0:
                raw_seq = np.concatenate([np.zeros((pad_len, N_WEATHER), dtype=np.float32), raw_seq])
            va_seqs_last.append(raw_seq)
            va_feats_last.append(feat_g[i])

        va_seqs_arr  = np.stack(va_seqs_last,  axis=0).astype(np.float32)
        va_feats_arr = np.stack(va_feats_last, axis=0).astype(np.float32)

        dl_model = train_dl_model(tr_seqs, tr_feats, tr_ys, va_seqs_arr, va_feats_arr, va_ys, len(NUM_FEATURES))

        if dl_model is not None:
            dl_val = predict_dl(dl_model, va_seqs_arr, va_feats_arr)
            show_mae("DL Hybrid (val)", y_va, dl_val)

            # Blend
            lgb_w, blend_mae = optimize_blend_2way(y_va, lgb_val, dl_val)
            dl_w = round(1 - lgb_w, 2)
            print(f"  Best blend: LGB={lgb_w:.2f}  DL={dl_w:.2f}  MAE={blend_mae:.4f}")
        else:
            lgb_w, dl_w = 1.0, 0.0
    else:
        lgb_w, dl_w = 1.0, 0.0
        print("\n[6/7] Skipping DL (torch not available).")

    # Final training on all data
    print("\n  Final LGB training (all regions) ...")
    X_all, y_all = build_sliding_windows(train_weekly, set(), NUM_FEATURES, stride=WINDOW_STRIDE)
    n_lgb = [int(getattr(m, "best_iteration_", None) or LGB_PARAMS["n_estimators"]) for m in lgb_models]
    final_lgb = train_lgb_models(X_all, y_all, None, None, n_lgb)

    # Test predictions
    print("\n[7/7] Test predictions ...")
    X_test = (
        test_feat.sort_values(["region_id", "ordinal"])
        .groupby("region_id", sort=False)
        .tail(1)[["region_id"] + NUM_FEATURES]
        .reset_index(drop=True)
    )
    X_test["region_id"] = X_test["region_id"].astype("category")

    lgb_test = predict_lgb(final_lgb, X_test)

    if dl_model is not None and dl_w > 0:
        # Build test sequences
        weather_mean = train_weekly[WEATHER_COLS].mean().values.astype(np.float32)
        weather_std  = train_weekly[WEATHER_COLS].std().clip(lower=1e-8).values.astype(np.float32)
        feat_mean    = X_all[NUM_FEATURES].mean().values.astype(np.float32)
        feat_std     = X_all[NUM_FEATURES].std().clip(lower=1e-8).values.astype(np.float32)

        # Retrain DL on all data with same epochs
        print("  Retraining DL on all regions ...")
        all_seqs, all_feats, all_ys = build_dl_sequences(
            train_weekly, NUM_FEATURES, weather_mean, weather_std, feat_mean, feat_std,
            skip_regions=None, stride=WINDOW_STRIDE, max_samples=MAX_DL_TRAIN,
        )
        # Use val arrays from last-window for final training val (just for early stopping reference)
        final_dl = train_dl_model(
            all_seqs, all_feats, all_ys,
            va_seqs_arr, va_feats_arr, va_ys, len(NUM_FEATURES)
        )

        # Test sequences: last weekly row per test region
        print("  Building test sequences ...")
        test_seqs_list, test_feats_list = [], []
        test_region_ids = X_test["region_id"].tolist()
        for region in test_region_ids:
            g = train_weekly.loc[train_weekly["region_id"] == region].sort_values("ordinal")
            n_g = len(g)
            i   = n_g - 1  # last training week = feature row for test prediction
            weather_g = ((g[WEATHER_COLS].fillna(0).values.astype(np.float32) - weather_mean) / weather_std)
            feat_g    = ((g[NUM_FEATURES].fillna(0).values.astype(np.float32) - feat_mean) / feat_std)
            start = max(0, i - SEQ_LEN + 1)
            raw_seq = weather_g[start: i + 1]
            pad_len = SEQ_LEN - len(raw_seq)
            if pad_len > 0:
                raw_seq = np.concatenate([np.zeros((pad_len, N_WEATHER), dtype=np.float32), raw_seq])
            test_seqs_list.append(raw_seq)
            test_feats_list.append(feat_g[i])

        test_seqs_arr  = np.stack(test_seqs_list,  axis=0).astype(np.float32)
        test_feats_arr = np.stack(test_feats_list, axis=0).astype(np.float32)

        # Use tabular features from test_feat for the MLP part
        test_feat_tab = X_test[NUM_FEATURES].fillna(0).values.astype(np.float32)
        test_feat_tab = (test_feat_tab - feat_mean) / feat_std

        dl_test = predict_dl(final_dl, test_seqs_arr, test_feat_tab)
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
    total_min = (time.time() - t0) / 60
    print(f"\n{'='*68}")
    print(f"  Saved: {OUT_PATH}")
    print(f"  Rows: {len(sub):,}  |  Total: {total_min:.1f} Min.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
