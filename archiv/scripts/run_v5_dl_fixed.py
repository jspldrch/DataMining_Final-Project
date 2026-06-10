"""
run_v5_dl_fixed.py  –  Deep Learning Ensemble for drought severity forecasting
COMPLETE FIXED VERSION - Läuft auf 16GB RAM in ~25-35 Minuten (Quick Mode)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found. Install: pip install torch")
    print("          Running in LightGBM-only mode.")

# ─── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
OUT_PATH = OUT_DIR / "submission_v5_fixed.csv"

# ─── Config ─────────────────────────────────────────────────────────────────────
QUICK_MODE = True  # False für bessere Ergebnisse (länger)

RANDOM_STATE = 42
VAL_WEEKS = 10 if not QUICK_MODE else 5
WEEK_BUCKET = 7
SEQ_LEN = 26
N_WEEKS = 5

# Feature reduction für DL
KEEP_DL_FEATURES = [
    'prec', 'tmp', 'humidity', 'tmp_max', 'tmp_min', 'wind',
    'prec_roll7_mean', 'prec_roll30_mean', 'prec_roll90_mean',
    'tmp_roll7_mean', 'tmp_roll30_mean',
    'month_sin', 'month_cos', 'day_sin', 'day_cos',
    'prec_deficit_90d', 'prec_trend_30d', 'tmp_anomaly_90d'
]

# DL training
DL_EPOCHS = 40 if QUICK_MODE else 150
DL_BATCH = 256
DL_LR = 1e-3
DL_HIDDEN = 96 if QUICK_MODE else 128
DL_LAYERS = 2
DL_DROPOUT = 0.2
DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

# GBDT
N_ESTIMATORS = 300 if QUICK_MODE else 600

# Feature definitions
WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

LAG_COLS = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre"]
LAGS = [1, 3, 7, 14, 21]
ROLL_WINS = [7, 14, 30, 60, 90]
DRY_THR = 1.0

GBDT_FEATURES: list[str] = []
DL_FEATURES: list[str] = []


def _build_feature_lists() -> None:
    """Build feature lists für GBDT und DL."""
    global GBDT_FEATURES, DL_FEATURES
    
    # GBDT Features (alle)
    lag_names = [f"{c}_lag{lag}" for c in LAG_COLS for lag in LAGS]
    roll_all = [f"{col}_roll{w}_{s}" for col in ["prec", "wind", "tmp"] 
                for w in ROLL_WINS for s in ("mean", "std", "max")]
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos"]
    drought = ["prec_deficit_90d", "prec_trend_30d", "humidity_deficit_90d",
               "tmp_anomaly_90d", "heat_drought_idx", "dry_days_14d", "dry_days_30d"]
    
    GBDT_FEATURES = WEATHER_COLS + lag_names + roll_all + calendar + drought
    
    # DL Features (reduziert)
    roll_means = [f"prec_roll{w}_mean" for w in [7, 14, 30, 60, 90]]
    roll_means += [f"tmp_roll{w}_mean" for w in [7, 14, 30, 60, 90]]
    roll_means += [f"wind_roll{w}_mean" for w in [7, 14, 30]]
    
    dl_base = WEATHER_COLS + roll_means + calendar + drought
    DL_FEATURES = [f for f in dl_base if f in KEEP_DL_FEATURES]
    DL_FEATURES = list(dict.fromkeys(DL_FEATURES))


def _parse_dates_inplace(df: pd.DataFrame) -> None:
    """Schnelle Date-Parsing ohne datetime objects."""
    parts = df["date"].str.split("-", expand=True)
    df["year"] = parts[0].astype(np.int32)
    df["month"] = parts[1].astype(np.int32)
    df["day"] = parts[2].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]


def compute_region_features(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature Engineering für eine Region (daily level)."""
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    
    new_cols = {}
    
    # Calendar features
    new_cols["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    new_cols["day_sin"] = np.sin(2 * np.pi * panel["day"] / 31).astype(np.float32)
    new_cols["day_cos"] = np.cos(2 * np.pi * panel["day"] / 31).astype(np.float32)
    
    # Lag features
    for col in LAG_COLS:
        if col not in panel.columns:
            continue
        for lag in LAGS:
            new_cols[f"{col}_lag{lag}"] = panel[col].shift(lag).astype(np.float32)
    
    # Rolling features
    for col in ["prec", "wind", "tmp"]:
        if col not in panel.columns:
            continue
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            roll = prior.rolling(w, min_periods=max(3, w//4))
            new_cols[f"{col}_roll{w}_mean"] = roll.mean().astype(np.float32)
            new_cols[f"{col}_roll{w}_std"] = roll.std().astype(np.float32)
            new_cols[f"{col}_roll{w}_max"] = roll.max().astype(np.float32)
    
    # Drought indicators
    pp = panel["prec"].shift(1)
    new_cols["prec_deficit_90d"] = (
        pp.rolling(90, min_periods=30).mean() - pp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    
    p7 = pp.rolling(7, min_periods=3).mean()
    p30 = pp.rolling(30, min_periods=10).mean()
    p30_std = pp.rolling(30, min_periods=10).std().clip(lower=0.01)
    new_cols["prec_trend_30d"] = ((p7 - p30) / p30_std).astype(np.float32)
    
    hp = panel["humidity"].shift(1)
    new_cols["humidity_deficit_90d"] = (
        hp.rolling(90, min_periods=30).mean() - hp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)
    
    tp = panel["tmp"].shift(1)
    t_anom = (tp.rolling(90, min_periods=30).mean() - tp.rolling(365, min_periods=60).mean()).astype(np.float32)
    new_cols["tmp_anomaly_90d"] = t_anom
    new_cols["heat_drought_idx"] = (new_cols["prec_deficit_90d"] * t_anom.clip(lower=0)).astype(np.float32)
    
    dry = (panel["prec"].shift(1) < DRY_THR).astype(np.float32)
    new_cols["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    new_cols["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)
    
    for col_name, col_values in new_cols.items():
        panel[col_name] = col_values
    
    # Fill NAs
    for col in panel.columns:
        if panel[col].dtype in ['float32', 'float64']:
            panel[col] = panel[col].fillna(0)
    
    n_tr = len(tr)
    return panel.iloc[:n_tr].copy(), panel.iloc[n_tr:].copy()


def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert Daily-Daten zu Weekly."""
    df = df.copy()
    df["week_id"] = df["ordinal"] // WEEK_BUCKET
    idx = df.groupby("week_id", sort=False)["ordinal"].idxmax()
    weekly = df.loc[idx].reset_index(drop=True)
    return weekly


def add_score_persist(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Score persistence auf Wochenbasis."""
    weekly_df = weekly_df.copy()
    weekly_df["score_persist"] = weekly_df.groupby("region_id")["score"].shift(1)
    weekly_df["score_persist"] = weekly_df["score_persist"].fillna(0).astype(np.float32)
    return weekly_df


def temporal_validation_split(
    weekly_df: pd.DataFrame,
    features: list[str],
    val_weeks: int = 10
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, set]:
    """Zeitbasierte Validierung."""
    X_train, y_train = [], []
    X_val, y_val = [], []
    val_regions = set()
    
    for region, group in weekly_df.groupby("region_id", sort=False):
        group = group.sort_values("ordinal")
        n = len(group)
        
        if n < val_weeks + 6:
            continue
        
        val_regions.add(region)
        train_end = n - val_weeks
        
        for i in range(train_end - 4):
            X_train.append(group.iloc[i][features].values)
            y_train.append(group.iloc[i+1:i+6]["score"].values)
        
        for i in range(train_end, n - 5):
            X_val.append(group.iloc[i][features].values)
            y_val.append(group.iloc[i+1:i+6]["score"].values)
    
    X_train_df = pd.DataFrame(np.array(X_train, dtype=np.float32), columns=features)
    X_val_df = pd.DataFrame(np.array(X_val, dtype=np.float32), columns=features)
    y_train_arr = np.array(y_train, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.float32)
    
    print(f"   Train windows: {len(X_train_df):,} | Val windows: {len(X_val_df):,}")
    print(f"   Val regions: {len(val_regions)}")
    
    return X_train_df, y_train_arr, X_val_df, y_val_arr, val_regions


def build_sequence_data(
    weekly_df: pd.DataFrame,
    features: list[str],
    seq_len: int,
    val_regions: set = None,
    stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Baut Sequenzen für DL-Modelle."""
    X_seq, y_seq = [], []
    
    for region, group in weekly_df.groupby("region_id", sort=False):
        if val_regions is not None and region in val_regions:
            continue
        
        group = group.sort_values("ordinal")
        data = group[features].fillna(0).to_numpy(dtype=np.float32)
        scores = group["score"].to_numpy(dtype=np.float32)
        n = len(data)
        
        if n < seq_len + N_WEEKS:
            continue
        
        n_windows = n - seq_len - N_WEEKS + 1
        indices = list(range(0, n_windows, stride))
        if n_windows - 1 not in indices:
            indices.append(n_windows - 1)
        
        for i in indices:
            X_seq.append(data[i:i+seq_len])
            y_seq.append(scores[i+seq_len:i+seq_len+N_WEEKS])
    
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def build_val_sequences(
    weekly_df: pd.DataFrame,
    features: list[str],
    seq_len: int,
    val_regions: set
) -> tuple[np.ndarray, np.ndarray]:
    """Baut Validierungssequenzen."""
    X_seq, y_seq = [], []
    
    for region in val_regions:
        group = weekly_df[weekly_df["region_id"] == region].sort_values("ordinal")
        n = len(group)
        
        if n < seq_len + N_WEEKS:
            continue
        
        data = group[features].fillna(0).to_numpy(dtype=np.float32)
        scores = group["score"].to_numpy(dtype=np.float32)
        
        X_seq.append(data[-(seq_len+N_WEEKS):-N_WEEKS])
        y_seq.append(scores[-N_WEEKS:])
    
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def build_test_sequences(
    train_weekly: pd.DataFrame,
    test_weekly: pd.DataFrame,
    features: list[str],
    seq_len: int
) -> tuple[np.ndarray, list]:
    """Baut Testsequenzen."""
    X_seq, regions = [], []
    
    for region in sorted(train_weekly["region_id"].unique()):
        train_data = train_weekly[train_weekly["region_id"] == region].sort_values("ordinal")
        test_data = test_weekly[test_weekly["region_id"] == region].sort_values("ordinal") if region in test_weekly["region_id"].values else pd.DataFrame()
        
        combined = pd.concat([train_data, test_data], ignore_index=True).sort_values("ordinal")
        data = combined[features].fillna(0).to_numpy(dtype=np.float32)
        
        if len(data) >= seq_len:
            X_seq.append(data[-seq_len:])
        else:
            pad = np.zeros((seq_len - len(data), data.shape[1]), dtype=np.float32)
            X_seq.append(np.vstack([pad, data]))
        
        regions.append(region)
    
    return np.array(X_seq, dtype=np.float32), regions


# ─── PyTorch Modelle ───────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    
    class AutoregressiveGRU(nn.Module):
        def __init__(self, n_feat: int, hidden: int = DL_HIDDEN, n_layers: int = DL_LAYERS,
                     n_weeks: int = N_WEEKS, dropout: float = DL_DROPOUT):
            super().__init__()
            self.n_weeks = n_weeks
            self.encoder = nn.GRU(n_feat, hidden, n_layers, batch_first=True,
                                  dropout=dropout if n_layers > 1 else 0.0,
                                  bidirectional=True)
            self.bridge = nn.Linear(hidden * 2, hidden)
            self.decoder_cell = nn.GRUCell(1, hidden)
            self.out_proj = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.init_tok = nn.Parameter(torch.zeros(1))
        
        def forward(self, x, y_teacher=None, tf_ratio=0.5):
            out, _ = self.encoder(x)
            ctx = self.bridge(out[:, -1, :])
            h_dec = torch.tanh(ctx)
            B = x.size(0)
            inp = self.init_tok.expand(B, 1)
            preds = []
            
            for step in range(self.n_weeks):
                h_dec = self.decoder_cell(inp, h_dec)
                pred = self.out_proj(h_dec)
                preds.append(pred)
                
                use_tf = (y_teacher is not None and self.training and torch.rand(1).item() < tf_ratio)
                inp = y_teacher[:, step:step+1] if use_tf else pred.detach()
            
            return torch.cat(preds, dim=1)
    
    
    class CNNLSTMModel(nn.Module):
        def __init__(self, n_feat: int, hidden: int = DL_HIDDEN, n_weeks: int = N_WEEKS, dropout: float = DL_DROPOUT):
            super().__init__()
            ch = max(32, hidden // 3)
            self.conv3 = nn.Sequential(nn.Conv1d(n_feat, ch, 3, padding=1), nn.ReLU())
            self.conv5 = nn.Sequential(nn.Conv1d(n_feat, ch, 5, padding=2), nn.ReLU())
            self.conv7 = nn.Sequential(nn.Conv1d(n_feat, ch, 7, padding=3), nn.ReLU())
            self.drop = nn.Dropout(dropout)
            self.lstm = nn.LSTM(ch * 3, hidden, 1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_weeks))
        
        def forward(self, x, **kwargs):
            xT = x.permute(0, 2, 1)
            c = torch.cat([self.conv3(xT), self.conv5(xT), self.conv7(xT)], dim=1)
            c = self.drop(c).permute(0, 2, 1)
            _, (h, _) = self.lstm(c)
            return self.head(h[-1])
    
    
    class TransformerModel(nn.Module):
        def __init__(self, n_feat: int, d_model: int = None, nhead: int = 4,
                     n_layers: int = DL_LAYERS, n_weeks: int = N_WEEKS, dropout: float = DL_DROPOUT):
            super().__init__()
            d_model = d_model or max(64, (DL_HIDDEN // nhead) * nhead)
            self.proj = nn.Linear(n_feat, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
            self.pos_emb = nn.Embedding(512, d_model)
            self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_weeks))
        
        def forward(self, x, **kwargs):
            B, T, _ = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            x = self.proj(x) + self.pos_emb(pos)
            x = self.encoder(x)
            return self.head(x[:, -1, :])
    
    
    def train_dl_model(
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = DL_EPOCHS,
        batch_size: int = DL_BATCH
    ) -> tuple[nn.Module, float]:
        """Trainiert ein DL-Modell."""
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=DL_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=DL_LR/20)
        
        X_train_t = torch.tensor(X_train, device=DEVICE)
        y_train_t = torch.tensor(y_train, device=DEVICE)
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size, shuffle=True, drop_last=False
        )
        
        best_mae = np.inf
        best_state = None
        patience = 15
        no_improve = 0
        
        for epoch in range(epochs):
            tf_ratio = max(0.0, 1.0 - 1.5 * epoch / epochs)
            model.train()
            
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb, y_teacher=yb, tf_ratio=tf_ratio)
                loss = torch.mean(torch.abs(pred.clamp(0, 5) - yb))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            model.eval()
            val_preds = []
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch = torch.tensor(X_val[i:i+batch_size], device=DEVICE)
                    val_preds.append(model(batch).clamp(0, 5).cpu().numpy())
            
            val_preds = np.vstack(val_preds)
            val_mae = np.mean(np.abs(val_preds - y_val))
            
            if val_mae < best_mae:
                best_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        
        return model, best_mae
    
    
    @torch.no_grad()
    def predict_dl_model(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Batch-Prediction."""
        model.eval()
        predictions = []
        model = model.to(DEVICE)
        
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], device=DEVICE)
            pred = model(batch).clamp(0, 5).cpu().numpy()
            predictions.append(pred)
        
        return np.vstack(predictions)


def train_gbdt_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_estimators: int = N_ESTIMATORS
) -> list[lgb.LGBMRegressor]:
    """Trainiert 5 LightGBM Modelle."""
    models = []
    
    for week in range(N_WEEKS):
        model = lgb.LGBMRegressor(
            objective="regression", metric="mae",
            n_estimators=n_estimators, learning_rate=0.04,
            num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE + week, n_jobs=-1, verbose=-1
        )
        
        model.fit(
            X_train, y_train[:, week],
            eval_set=[(X_val, y_val[:, week])],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        models.append(model)
    
    return models


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(y_pred, 0, 5) - y_true)))


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    _build_feature_lists()
    t0 = time.time()
    
    print("=" * 70)
    print("  Natural Disaster Severity Prediction  -  run_v5_dl_fixed.py")
    mode = "QUICK (~25-35 min)" if QUICK_MODE else "FULL (~2-3 hours)"
    torch_status = f"PyTorch ON  |  device={DEVICE}" if TORCH_AVAILABLE else "PyTorch OFF (LGB only)"
    print(f"  Mode: {mode}  |  {torch_status}")
    print(f"  GBDT features: {len(GBDT_FEATURES)}  |  DL features: {len(DL_FEATURES)}")
    print("=" * 70)
    
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    dtypes = {c: np.float32 for c in WEATHER_COLS}
    train_raw = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test_raw = pd.read_csv(TEST_CSV, dtype=dtypes)
    _parse_dates_inplace(train_raw)
    _parse_dates_inplace(test_raw)
    train_raw["score"] = pd.to_numeric(train_raw["score"], errors="coerce").astype(np.float32)
    
    print(f"   Train: {len(train_raw):,} rows | Test: {len(test_raw):,} rows")
    print(f"   Regions: {train_raw['region_id'].nunique():,}")
    
    # ── 2. Feature Engineering ────────────────────────────────────────────────
    print("\n[2/6] Feature engineering (this takes a few minutes)...")
    
    train_by_region = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    test_by_region = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id", sort=False)}
    
    train_parts, test_parts = [], []
    regions = list(train_by_region.keys())
    
    for i, region in enumerate(regions, 1):
        if i % 500 == 0 or i == len(regions):
            print(f"   Region {i}/{len(regions)}  |  {time.time()-t0:.1f}s")
        
        tr_feat, te_feat = compute_region_features(
            train_by_region[region],
            test_by_region.get(region, pd.DataFrame())
        )
        train_parts.append(tr_feat)
        test_parts.append(te_feat)
    
    train_feat = pd.concat(train_parts, ignore_index=True)
    test_feat = pd.concat(test_parts, ignore_index=True)
    print(f"   Done | {time.time()-t0:.1f}s")
    
    # ── 3. Weekly aggregation ─────────────────────────────────────────────────
    print("\n[3/6] Weekly aggregation...")
    train_weekly = daily_to_weekly(train_feat[train_feat["score"].notna()])
    test_weekly = daily_to_weekly(test_feat)
    
    train_weekly = add_score_persist(train_weekly)
    test_weekly = add_score_persist(test_weekly)
    
    if "score_persist" not in GBDT_FEATURES:
        GBDT_FEATURES.append("score_persist")
    
    print(f"   Train weeks: {len(train_weekly):,} | Test weeks: {len(test_weekly):,}")
    
    # ── 4. Train/Val split ────────────────────────────────────────────────────
    print("\n[4/6] Creating temporal validation split...")
    
    X_gbdt_train, y_gbdt_train, X_gbdt_val, y_gbdt_val, val_regions = temporal_validation_split(
        train_weekly, GBDT_FEATURES, VAL_WEEKS
    )
    
    X_dl_train, y_dl_train = build_sequence_data(train_weekly, DL_FEATURES, SEQ_LEN, stride=1)
    X_dl_val, y_dl_val = build_val_sequences(train_weekly, DL_FEATURES, SEQ_LEN, val_regions)
    X_dl_test, test_regions = build_test_sequences(train_weekly, test_weekly, DL_FEATURES, SEQ_LEN)
    
    print(f"   DL sequences: train={len(X_dl_train):,} | val={len(X_dl_val):,} | test={len(X_dl_test):,}")
    
    # Persistence baseline
    persist_preds = np.tile(
        train_weekly.groupby("region_id")["score"].last().reindex(list(val_regions)).fillna(0).values[:, None],
        (1, 5)
    )
    print(f"\n   Persistence baseline MAE: {mae(y_gbdt_val, persist_preds):.4f}")
    
    # Normalize DL data
    if TORCH_AVAILABLE and len(X_dl_train) > 0:
        scaler = StandardScaler()
        orig_shape = X_dl_train.shape
        X_dl_train = scaler.fit_transform(X_dl_train.reshape(-1, X_dl_train.shape[-1])).reshape(orig_shape)
        
        if len(X_dl_val) > 0:
            X_dl_val = scaler.transform(X_dl_val.reshape(-1, X_dl_val.shape[-1])).reshape(X_dl_val.shape)
        
        if len(X_dl_test) > 0:
            X_dl_test = scaler.transform(X_dl_test.reshape(-1, X_dl_test.shape[-1])).reshape(X_dl_test.shape)
    
    # ── 5. Training ───────────────────────────────────────────────────────────
    print(f"\n[5/6] Training (device={DEVICE})...")
    
    # LightGBM
    print("   LightGBM...")
    lgb_models = train_gbdt_models(X_gbdt_train, y_gbdt_train, X_gbdt_val, y_gbdt_val)
    lgb_val_preds = np.column_stack([m.predict(X_gbdt_val) for m in lgb_models])
    lgb_val_mae = mae(y_gbdt_val, lgb_val_preds)
    print(f"      Validation MAE: {lgb_val_mae:.4f}")
    
    # Deep Learning models
    dl_val_preds = {}
    dl_models = {}
    
    if TORCH_AVAILABLE and len(X_dl_train) > 0:
        for name, ModelClass in [
            ("GRU", AutoregressiveGRU),
            ("CNN-LSTM", CNNLSTMModel),
            ("Transformer", TransformerModel),
        ]:
            print(f"   {name}...")
            model = ModelClass(len(DL_FEATURES))
            model, val_mae = train_dl_model(model, X_dl_train, y_dl_train, X_dl_val, y_dl_val)
            dl_val_preds[name] = predict_dl_model(model, X_dl_val)
            dl_models[name] = model
            print(f"      Validation MAE: {val_mae:.4f}")
    
    # ── 6. Ensemble blending ──────────────────────────────────────────────────
    print("\n[6/6] Ensemble optimization & submission...")
    
    candidates = [lgb_val_preds] + list(dl_val_preds.values())
    candidate_names = ["LGB"] + list(dl_val_preds.keys())
    
    print(f"   Blending {len(candidates)} models: {candidate_names}")
    
    best_mae_val = np.inf
    best_weights = np.ones(len(candidates)) / len(candidates)
    
    # Grid search for optimal weights
    alphas = np.arange(0.0, 1.01, 0.1)
    n_combos = 0
    
    for combo in product(alphas, repeat=len(candidates)):
        combo = np.array(combo)
        if abs(combo.sum() - 1.0) > 0.01:
            continue
        
        n_combos += 1
        blend = np.zeros_like(candidates[0])
        for w, pred in zip(combo, candidates):
            blend += w * pred
        
        curr_mae = mae(y_gbdt_val, blend)
        if curr_mae < best_mae_val:
            best_mae_val = curr_mae
            best_weights = combo
    
    # Print best weights
    print(f"\n   Best weights:")
    for name, w in zip(candidate_names, best_weights):
        print(f"      {name}: {w:.2f}")
    print(f"   Ensemble validation MAE: {best_mae_val:.4f}")
    
    # ── 7. Final predictions ──────────────────────────────────────────────────
    print("\n   Generating final predictions...")
    
    # Retrain LGB on all data
    X_gbdt_all, y_gbdt_all, _, _, _ = temporal_validation_split(train_weekly, GBDT_FEATURES, 0)
    final_lgb_models = []
    
    for week in range(N_WEEKS):
        model = lgb.LGBMRegressor(
            objective="regression", metric="mae",
            n_estimators=N_ESTIMATORS, learning_rate=0.04,
            num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE + week, n_jobs=-1, verbose=-1
        )
        model.fit(X_gbdt_all, y_gbdt_all[:, week])
        final_lgb_models.append(model)
    
    # GBDT test predictions
    X_gbdt_test = test_weekly.sort_values(["region_id", "ordinal"]).groupby("region_id").tail(1)
    X_gbdt_test = X_gbdt_test[GBDT_FEATURES].fillna(0)
    lgb_test_preds = np.column_stack([m.predict(X_gbdt_test) for m in final_lgb_models])
    
    # DL test predictions
    dl_test_preds = []
    if TORCH_AVAILABLE and len(dl_models) > 0:
        for name, model in dl_models.items():
            preds = predict_dl_model(model, X_dl_test)
            dl_test_preds.append(preds)
    
    # Ensemble test predictions
    all_test_preds = [lgb_test_preds] + dl_test_preds
    final_preds = np.zeros_like(lgb_test_preds