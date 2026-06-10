"""
kaggle_transformer.py — Transformer für Drought Severity Prediction
====================================================================
Architektur:
  1. Feature Engineering: pro Zeitschritt (täglich → wöchentlich aggregiert)
  2. Transformer Encoder: lernt zeitliche Muster über die 13-Wochen-Sequenz
  3. Fully Connected Head: → 5 Wochen Output

Pfade (Kaggle Notebook):
  /kaggle/input/datasets/jaspspsp/traindataset/train.npz
  /kaggle/input/datasets/jaspspsp/testdataset/test.npz
  /kaggle/input/datasets/jaspspsp/samplesubmission/sample_submission.csv

Output:
  /kaggle/working/submission_transformer.csv
"""
from __future__ import annotations
import time, warnings, math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")  # kein Display noetig auf Kaggle
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore")

# ── Pfade ─────────────────────────────────────────────────────────────────────
TRAIN_NPZ  = "/kaggle/input/datasets/jaspspsp/traindataset/train.npz"
TEST_NPZ   = "/kaggle/input/datasets/jaspspsp/testdataset/test.npz"
SAMPLE_SUB = "/kaggle/input/datasets/jaspspsp/samplesubmission/sample_submission.csv"
OUT_PATH   = "/kaggle/working/submission_transformer.csv"

# ── Hyperparameter ────────────────────────────────────────────────────────────
SEQ_LEN         = 13       # 13 Wochen = 91 Tage Input-Sequenz
PRED_LEN        = 5        # 5 Wochen vorhersagen
RECENT_YEARS    = 8        # nur letzte 8 Jahre für Training
ORDINAL_PER_YEAR= 372
DRY_THRESHOLD   = 1.0
VAL_REGION_FRAC = 0.20     # 20% Regionen als Holdout-Val
RANDOM_STATE    = 42

# Transformer Architektur
D_MODEL     = 128          # Embedding-Dimension (größer = mehr Kapazität)
N_HEADS     = 4            # Attention Heads (D_MODEL muss durch N_HEADS teilbar sein)
N_LAYERS    = 3            # Transformer Encoder Schichten
D_FF        = 512          # Feed-Forward Dimension im Transformer
DROPOUT     = 0.1

# Training
BATCH_SIZE  = 512
EPOCHS      = 50
LR          = 1e-3
WEIGHT_DECAY= 1e-4
PATIENCE    = 8            # Early Stopping Patience
SEEDS       = [42, 123, 777]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
ROLL_COLS = ["prec", "humidity", "tmp", "wind"]
ROLL_WINS = [7, 14, 30, 60, 90]   # max 90d — Test hat nur 91 Tage

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def elapsed(t0):
    s = time.time() - t0
    return f"{s/60:.1f}min" if s >= 60 else f"{s:.0f}s"

def mae_np(y, p):
    return float(np.mean(np.abs(np.clip(p, 0, 5) - y)))


def weighted_l1_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """L1-Loss mit Upweighting für höhere Drought-Scores — verbessert Kalibrierung für Dürre."""
    weight = (1.0 + target / 2.5).clamp(1.0, 3.0)
    return (torch.abs(pred - target) * weight).mean()


# ── NPZ laden ─────────────────────────────────────────────────────────────────
def load_npz(path: str) -> pd.DataFrame:
    d = np.load(path, allow_pickle=True)
    df = pd.DataFrame()
    df["region_id"] = d["region_names"][d["region_id"]].astype(str)
    df["year"]  = d["year"].astype(np.int32)
    df["month"] = d["month"].astype(np.int32)
    df["day"]   = d["day"].astype(np.int32)
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]
    for col in WEATHER_COLS:
        df[col] = d[col].astype(np.float32) if col in d else np.float32(0)
    if "score" in d:
        df["score"] = d["score"].astype(np.float32)
    return df


# ── Feature Engineering pro Region ───────────────────────────────────────────
def _region_features(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Berechnet Features auf dem kombinierten Train+Test Panel.
    Gibt (train_features, test_features) zurück.
    """
    te = te.copy()
    te["score"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True).sort_values("ordinal").reset_index(drop=True)
    nc: dict = {}

    # Zyklische Zeit-Features
    nc["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    nc["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)
    nc["week_sin"]  = np.sin(2 * np.pi * (panel["ordinal"] % 52) / 52).astype(np.float32)
    nc["week_cos"]  = np.cos(2 * np.pi * (panel["ordinal"] % 52) / 52).astype(np.float32)

    # Rolling Stats — max 90d wegen 91-Tage Test-Grenze
    for col in ROLL_COLS:
        prior = panel[col].shift(1)
        for w in ROLL_WINS:
            r = prior.rolling(w, min_periods=max(3, w // 10))
            nc[f"{col}_roll{w}_mean"] = r.mean().astype(np.float32)
            nc[f"{col}_roll{w}_std"]  = r.std().astype(np.float32)

    # Dürre-Indices
    pp = panel["prec"].shift(1)
    nc["prec_deficit_90d"] = (
        pp.rolling(90, min_periods=30).mean() -
        pp.rolling(365, min_periods=60).mean()
    ).astype(np.float32)

    tp   = panel["tmp"].shift(1)
    anom = (tp.rolling(90, min_periods=30).mean() -
            tp.rolling(365, min_periods=60).mean()).astype(np.float32)
    nc["tmp_anomaly_90d"]  = anom

    dry = (panel["prec"].shift(1) < DRY_THRESHOLD).astype(np.float32)
    nc["dry_days_14d"] = dry.rolling(14, min_periods=3).sum().astype(np.float32)
    nc["dry_days_30d"] = dry.rolling(30, min_periods=7).sum().astype(np.float32)

    # VPD — Vapor Pressure Deficit
    tmp_p = panel["tmp"].shift(1)
    hum_p = panel["humidity"].shift(1)
    e_sat = 6.112 * np.exp(17.67 * tmp_p / (tmp_p + 243.5))
    vpd   = (e_sat * (1.0 - hum_p / 100.0)).clip(lower=0).astype(np.float32)
    nc["vpd"]             = vpd
    nc["vpd_roll30_mean"] = vpd.rolling(30, min_periods=7).mean().astype(np.float32)
    nc["vpd_roll90_mean"] = vpd.rolling(90, min_periods=20).mean().astype(np.float32)

    panel = pd.concat([panel, pd.DataFrame(nc, index=panel.index)], axis=1)
    n = len(tr)
    return panel.iloc[:n].copy(), panel.iloc[n:].copy()


def _daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    wk = df["ordinal"] // 7
    return df.loc[df.groupby(wk, sort=False)["ordinal"].idxmax()].reset_index(drop=True)


def build_step_features() -> list[str]:
    """Features pro einzelnem Zeitschritt (werden nicht geflattened — Transformer bekommt Sequenz)."""
    f = list(WEATHER_COLS)
    f += [f"{c}_roll{w}_{s}" for c in ROLL_COLS for w in ROLL_WINS for s in ("mean", "std")]
    f += ["month_sin", "month_cos", "week_sin", "week_cos"]
    f += ["prec_deficit_90d", "tmp_anomaly_90d", "dry_days_14d", "dry_days_30d"]
    f += ["vpd", "vpd_roll30_mean", "vpd_roll90_mean"]
    return f


# ── Compute Features über alle Regionen ──────────────────────────────────────
def compute_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame, t0: float):
    region_means = train_raw.groupby("region_id")["score"].mean()

    # regional_seasonal_mean: Ø Score pro Region × Monat
    labeled_raw = train_raw[train_raw["score"].notna()].copy()
    seas = labeled_raw.groupby(["region_id", "month"])["score"].mean()

    regions = train_raw["region_id"].unique()
    tr_by = {r: g.reset_index(drop=True) for r, g in train_raw.groupby("region_id", sort=False)}
    te_by = {r: g.reset_index(drop=True) for r, g in test_raw.groupby("region_id", sort=False)}
    del train_raw, test_raw

    all_tr, all_te = [], []
    for i, region in enumerate(regions, 1):
        tf, ef = _region_features(tr_by[region], te_by.get(region, pd.DataFrame(columns=tr_by[region].columns)))
        all_tr.append(tf)
        all_te.append(ef)
        if i % 500 == 0 or i == len(regions):
            print(f"   Region {i}/{len(regions)}  [{elapsed(t0)}]")

    train_feat = pd.concat(all_tr, ignore_index=True)
    test_feat  = pd.concat(all_te, ignore_index=True)
    del all_tr, all_te

    # Region-Level Features anhängen
    train_feat["regional_mean_score"] = train_feat["region_id"].map(region_means).astype(np.float32)
    test_feat["regional_mean_score"]  = test_feat["region_id"].map(region_means).astype(np.float32)

    def add_seasonal(df):
        key = list(zip(df["region_id"], df["month"].astype(int)))
        df["regional_seasonal_mean"] = np.array(
            [seas.get(k, region_means.get(k[0], 0.0)) for k in key], dtype=np.float32
        )
    add_seasonal(train_feat)
    add_seasonal(test_feat)

    labeled_feat = train_feat[train_feat["score"].notna()].copy()
    weekly = pd.concat(
        [_daily_to_weekly(g) for _, g in labeled_feat.groupby("region_id", sort=False)],
        ignore_index=True,
    )
    del labeled_feat

    return weekly, test_feat, region_means, seas


def filter_recent(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("region_id", sort=False):
        cutoff = int(g["ordinal"].max()) - RECENT_YEARS * ORDINAL_PER_YEAR
        parts.append(g[g["ordinal"] >= cutoff])
    return pd.concat(parts, ignore_index=True)


# ── Normalisierung ─────────────────────────────────────────────────────────────
class FeatureScaler:
    """Einfacher Z-Score Scaler, der auf Trainings-Features passt."""
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.nanmean(X, axis=0, keepdims=True).astype(np.float32)
        self.std_  = np.nanstd(X,  axis=0, keepdims=True).astype(np.float32)
        self.std_  = np.where(self.std_ < 1e-6, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ── Dataset ───────────────────────────────────────────────────────────────────
class DroughtDataset(Dataset):
    """
    Gibt (X_seq, y) zurück:
      X_seq: (SEQ_LEN, n_features) — Sequenz der letzten SEQ_LEN Wochen
      y:     (PRED_LEN,)           — nächste 5 Wochen Scores
    """
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.X = torch.from_numpy(sequences)  # (N, SEQ_LEN, n_feat)
        self.y = torch.from_numpy(targets)    # (N, PRED_LEN)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_sequences(weekly: pd.DataFrame, step_features: list,
                    skip_regions: set = set(), stride: int = 1,
                    last_n_windows: int = None):
    """
    Baut (N, SEQ_LEN, n_feat) Sequenzen und (N, 5) Targets.
    last_n_windows: wenn gesetzt, nur die letzten N Fenster pro Region (Val-Schema).
    """
    Xs, ys = [], []
    for region, g in weekly.groupby("region_id", sort=False):
        if region in skip_regions:
            continue
        g  = g.sort_values("ordinal")
        Xn = g[step_features].to_numpy(np.float32)
        sc = g["score"].to_numpy(np.float32)
        n  = len(g)
        if n < SEQ_LEN + PRED_LEN:
            continue
        limit = n - SEQ_LEN - PRED_LEN + 1

        if last_n_windows is not None:
            # Nur die letzten N Fenster — repräsentativ für Kaggle-Testszenario
            start = max(0, limit - last_n_windows)
            indices = range(start, limit)
        else:
            indices = range(0, limit, stride)

        for i in indices:
            y_vec = sc[i + SEQ_LEN: i + SEQ_LEN + PRED_LEN]
            if not np.any(np.isnan(y_vec)):
                Xs.append(Xn[i: i + SEQ_LEN])
                ys.append(y_vec)

    return np.stack(Xs).astype(np.float32), np.stack(ys).astype(np.float32)


def build_test_sequences(test_feat: pd.DataFrame, step_features: list,
                         sample_sub_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Baut Test-Sequenzen in exakt der Reihenfolge der sample_submission.csv.
    Garantiert: test_ids[i] == sample_submission.region_id[i]
    """
    test_weekly = pd.concat(
        [_daily_to_weekly(g) for _, g in test_feat.groupby("region_id", sort=False)],
        ignore_index=True,
    )
    # region_id Typ normalisieren — NPZ gibt manchmal bytes statt str
    test_weekly["region_id"] = test_weekly["region_id"].astype(str).str.strip()

    # Reihenfolge aus sample_submission
    submission_order = pd.read_csv(sample_sub_path)["region_id"].astype(str).str.strip().tolist()

    # Lookup: region → Feature-Matrix
    region_to_rows = {}
    for region, g in test_weekly.groupby("region_id", sort=False):
        g    = g.sort_values("ordinal")
        rows = g[step_features].to_numpy(np.float32)
        if len(rows) < SEQ_LEN:
            # Erste Zeile wiederholen statt Nullen: Nullen → (0-mean)/std = Extremwerte nach Z-Score
            pad  = np.tile(rows[0:1], (SEQ_LEN - len(rows), 1))
            rows = np.vstack([pad, rows])
        else:
            rows = rows[-SEQ_LEN:]
        region_to_rows[str(region).strip()] = rows

    # Sequenzen in submission_order aufbauen
    Xs, regions = [], []
    missing = []
    for region in submission_order:
        if region in region_to_rows:
            Xs.append(region_to_rows[region])
        else:
            # Region nicht in Test-Daten → Nullen
            Xs.append(np.zeros((SEQ_LEN, len(step_features)), dtype=np.float32))
            missing.append(region)
        regions.append(region)

    if missing:
        print(f"  WARNUNG: {len(missing)} Regionen nicht in test.npz gefunden")

    return np.stack(Xs).astype(np.float32), np.array(regions)


# ── Transformer Modell ────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Standard sinusoidales Positional Encoding."""
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class DroughtTransformer(nn.Module):
    """
    Architektur:
      Input:  (batch, SEQ_LEN, n_features)
      Linear Projection → d_model
      Positional Encoding
      Transformer Encoder (N_LAYERS × Multi-Head Attention + FFN)
      Pooling (Mean über Sequenz)
      FC Head → PRED_LEN Outputs
    """
    def __init__(self, n_features: int):
        super().__init__()

        # 1. Parallel Feature Projection (pro Zeitschritt)
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
        )

        # 2. Positional Encoding
        self.pos_enc = PositionalEncoding(D_MODEL, max_len=SEQ_LEN + 10, dropout=DROPOUT)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,   # (batch, seq, feat) — kein Transponieren nötig
            norm_first=True,    # Pre-LN: stabileres Training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # 4. Fully Connected Head
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 2, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT / 2),
            nn.Linear(D_MODEL, PRED_LEN),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, SEQ_LEN, n_features)
        x = self.input_proj(x)          # (batch, SEQ_LEN, D_MODEL)
        x = self.pos_enc(x)             # + Positional Encoding
        x = self.transformer(x)         # (batch, SEQ_LEN, D_MODEL)
        x = x[:, -1, :]                 # Letzter Zeitschritt: kennt via Attention die ganze History
        x = self.head(x)                # (batch, PRED_LEN)
        return x

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gibt Attention-Gewichte der ersten Encoder-Schicht zurueck.
        Output: (batch, N_HEADS, SEQ_LEN, SEQ_LEN)
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        # Direkt auf den ersten TransformerEncoderLayer zugreifen
        layer = self.transformer.layers[0]
        attn_out, attn_weights = layer.self_attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        return attn_weights  # (batch, N_HEADS, SEQ_LEN, SEQ_LEN)


# ── Training & Evaluation ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler_amp):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds, targets = [], []
    for X_batch, y_batch in loader:
        p = model(X_batch.to(DEVICE)).cpu().numpy()
        preds.append(np.clip(p, 0, 5))
        targets.append(y_batch.numpy())
    preds   = np.vstack(preds)
    targets = np.vstack(targets)
    return mae_np(targets, preds), preds, targets


@torch.no_grad()
def predict(model, X_seq: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X_seq), batch_size):
        batch = torch.from_numpy(X_seq[i: i + batch_size]).to(DEVICE)
        preds.append(model(batch).cpu().numpy())
    return np.clip(np.vstack(preds), 0, 5)


def train_transformer(X_tr_seq, y_tr, X_va_seq, y_va, n_features: int, seed: int):
    """Trainiert einen Transformer mit Early Stopping auf Val-Holdout."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model      = DroughtTransformer(n_features).to(DEVICE)
    n_epochs   = EPOCHS
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=LR / 10)
    criterion  = weighted_l1_loss
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    tr_ds = DroughtDataset(X_tr_seq, y_tr)
    va_ds = DroughtDataset(X_va_seq, y_va)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True)

    best_val_mae  = float("inf")
    best_state    = None
    patience_cnt  = 0
    best_epoch_nr = n_epochs

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, criterion, scaler_amp)
        val_mae, _, _ = evaluate(model, va_loader, criterion)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae  = val_mae
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch_nr = epoch
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}/{n_epochs}  train_loss={train_loss:.4f}  "
                  f"val_mae={val_mae:.4f}  best={best_val_mae:.4f}  lr={lr_now:.2e}")

        if patience_cnt >= PATIENCE:
            print(f"    Early Stopping bei Epoch {epoch}  (best val_mae={best_val_mae:.4f})")
            break

    model.load_state_dict(best_state)
    model._best_epoch_ = best_epoch_nr  # fuer Final-Training merken
    return model, best_val_mae


def train_fixed_epochs(X_tr_seq: np.ndarray, y_tr: np.ndarray,
                       n_features: int, seed: int, n_epochs: int):
    """
    Trainiert für exakt n_epochs ohne Early Stopping.
    Für Final Training auf allen Daten (Train + Val) nach der Hyperparameter-Suche.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model      = DroughtTransformer(n_features).to(DEVICE)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=LR / 10)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    tr_ds     = DroughtDataset(X_tr_seq, y_tr)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True)

    for epoch in range(1, n_epochs + 1):
        train_one_epoch(model, tr_loader, optimizer, scheduler, weighted_l1_loss, scaler_amp)
        scheduler.step()
        if epoch % 10 == 0 or epoch == n_epochs:
            print(f"    Final Epoch {epoch:3d}/{n_epochs}")

    return model


# ── Analyse-Funktionen ────────────────────────────────────────────────────────

def analyze_feature_importance(model: nn.Module, X_va_sc: np.ndarray,
                                y_va: np.ndarray, all_features: list,
                                n_samples: int = 500) -> pd.DataFrame:
    """
    Gradient-based Feature Importance (Integrated Gradients, vereinfacht).
    Misst wie stark sich der MAE aendert wenn ein Feature auf seinen Mittelwert gesetzt wird.
    Auch bekannt als: Permutation Importance auf Feature-Ebene.
    """
    print("\n  [Analyse] Feature Importance (Gradient x Input) ...")
    model.eval()

    # Subset fuer Geschwindigkeit
    idx = np.random.default_rng(42).choice(len(X_va_sc), min(n_samples, len(X_va_sc)), replace=False)
    X_sub = torch.from_numpy(X_va_sc[idx]).to(DEVICE)
    y_sub = y_va[idx]

    n_feat = len(all_features)
    importance = np.zeros(n_feat, dtype=np.float32)

    # Baseline MAE
    with torch.no_grad():
        base_pred = np.clip(model(X_sub).cpu().numpy(), 0, 5)
    base_mae = mae_np(y_sub, base_pred)

    # Pro Feature: auf Mittelwert setzen, MAE-Differenz messen
    for fi in range(n_feat):
        X_masked = X_sub.clone()
        X_masked[:, :, fi] = 0.0  # bereits z-normalisiert, 0 = Mittelwert
        with torch.no_grad():
            masked_pred = np.clip(model(X_masked).cpu().numpy(), 0, 5)
        masked_mae = mae_np(y_sub, masked_pred)
        importance[fi] = masked_mae - base_mae  # positiv = Feature hilft

    df = pd.DataFrame({
        "feature": all_features,
        "importance": importance,
        "importance_pct": 100 * importance / (importance.sum() + 1e-8),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Gruppen
    groups = {
        "Rolling Stats": lambda f: "roll" in f,
        "VPD":           lambda f: "vpd" in f,
        "Duerreindizes": lambda f: any(k in f for k in ["deficit", "anomaly", "dry_days", "trend"]),
        "Zeit (zyklisch)": lambda f: any(k in f for k in ["sin", "cos"]),
        "Regional":      lambda f: "regional" in f,
        "Wetter direkt": lambda f: f in ["prec","surf_pre","humidity","tmp","dp_tmp",
                                          "wb_tmp","tmp_max","tmp_min","tmp_range",
                                          "surf_tmp","wind","wind_max","wind_min","wind_range"],
    }
    print(f"\n  {'─'*58}")
    print(f"  FEATURE IMPORTANCE  (Masked-Feature, n={n_samples})")
    print(f"  {'─'*58}")
    print(f"  {'Rang':<5} {'Feature':<38} {'ΔMAE':>8} {'%':>6}")
    for i, row in df.head(20).iterrows():
        print(f"  {i+1:<5d} {row.feature:<38} {row.importance:>+8.5f} {row.importance_pct:>5.1f}%")

    print(f"\n  Gruppen-Summe (ΔMAE):")
    for gname, gfunc in groups.items():
        mask = [gfunc(f) for f in df["feature"]]
        g_imp = df.loc[mask, "importance"].sum()
        print(f"    {gname:<20} {g_imp:>+8.5f}")
    print(f"  {'─'*58}\n")

    return df


def analyze_attention(model: nn.Module, X_va_sc: np.ndarray,
                      n_samples: int = 200) -> np.ndarray:
    """
    Mittlere Attention-Gewichte ueber Val-Samples.
    Zeigt: Welche Wochen im 13-Wochen-Fenster sind fuer die Vorhersage wichtig?
    Output: (SEQ_LEN,) — Attention pro Zeitschritt (gemittelt ueber Heads und Keys)
    """
    print("  [Analyse] Attention Weights (welche Wochen sind wichtig?) ...")
    model.eval()
    idx   = np.random.default_rng(42).choice(len(X_va_sc), min(n_samples, len(X_va_sc)), replace=False)
    X_sub = torch.from_numpy(X_va_sc[idx]).to(DEVICE)

    with torch.no_grad():
        attn = model.get_attention_weights(X_sub)  # (batch, heads, seq, seq)

    # Mitteln: ueber batch, heads, und Query-Dimension → (SEQ_LEN,) = wer wird beachtet
    attn_mean = attn.cpu().numpy().mean(axis=(0, 1, 2))  # (SEQ_LEN,)
    attn_mean = attn_mean / (attn_mean.sum() + 1e-8)

    print(f"\n  Attention-Gewichte pro Woche (Woche 1 = aelteste, Woche 13 = aktuellste):")
    print(f"  {'Woche':<8} {'Aufmerksamkeit':>16} {'Bar'}")
    for w in range(SEQ_LEN):
        bar = "█" * int(attn_mean[w] * 200)
        print(f"  W{w+1:<6d} {attn_mean[w]:>15.4f}  {bar}")

    most_important = int(np.argmax(attn_mean)) + 1
    print(f"\n  → Wichtigste Woche: W{most_important} "
          f"({'aktuellste' if most_important == SEQ_LEN else f'{SEQ_LEN - most_important} Wochen zurueck'})")
    return attn_mean


def analyze_errors(y_va: np.ndarray, final_val_preds: np.ndarray) -> None:
    """
    Fehleranalyse: MAE pro Woche, Bias, Score-Verteilung.
    Zeigt wo das Modell systematisch falsch liegt.
    """
    print("  [Analyse] Fehleranalyse ...")
    print(f"\n  {'─'*58}")
    print(f"  FEHLERANALYSE (Val-Set)")
    print(f"  {'─'*58}")

    # MAE + Bias pro Woche
    print(f"  {'Woche':<8} {'MAE':>8} {'Bias':>8} {'Corr':>8}  Interpretation")
    for wk in range(PRED_LEN):
        y_w = y_va[:, wk]
        p_w = final_val_preds[:, wk]
        wk_mae  = mae_np(y_w, p_w)
        wk_bias = float(np.mean(p_w - y_w))   # positiv = Modell ueberschaetzt
        # Pearson Korrelation
        if y_w.std() > 1e-6 and p_w.std() > 1e-6:
            corr = float(np.corrcoef(y_w, p_w)[0, 1])
        else:
            corr = 0.0
        bias_str = "Ueberschaetzt" if wk_bias > 0.05 else ("Unterschaetzt" if wk_bias < -0.05 else "Gut")
        print(f"  Woche {wk+1:<3d} {wk_mae:>8.4f} {wk_bias:>+8.4f} {corr:>8.4f}  {bias_str}")

    # Score-Bucket Analyse: wo ist der Fehler am groessten?
    print(f"\n  MAE nach echtem Score-Bucket:")
    print(f"  {'Score':<12} {'N':>6} {'MAE':>8}  {'ΔMAE vs. Gesamt':>16}")
    overall_mae = mae_np(y_va, final_val_preds)
    for bucket_lo, bucket_hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5.1)]:
        mask = (y_va >= bucket_lo) & (y_va < bucket_hi)
        n = mask.sum()
        if n > 0:
            bucket_mae = mae_np(y_va[mask], final_val_preds[mask])
            delta = bucket_mae - overall_mae
            label = f"{bucket_lo:.0f}–{min(bucket_hi, 5):.0f}"
            bar   = "+" * int(abs(delta) * 20)
            print(f"  Score {label:<7} {n:>6,} {bucket_mae:>8.4f}  {delta:>+8.4f}  {bar}")

    print(f"\n  Gesamt Val MAE: {overall_mae:.4f}")
    print(f"  {'─'*58}\n")


def save_plots(feat_imp_df: pd.DataFrame, attn_weights: np.ndarray,
               y_va: np.ndarray, final_val_preds: np.ndarray,
               all_features: list) -> None:
    """Speichert Analyse-Plots als PNG nach /kaggle/working/."""
    if not HAS_MPL:
        print("  matplotlib nicht verfuegbar — keine Plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Transformer Analyse", fontsize=14, fontweight="bold")

    # Plot 1: Top-20 Feature Importance
    ax = axes[0]
    top = feat_imp_df.head(20)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top["importance"]]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔMAE (Feature maskiert)")
    ax.set_title("Feature Importance (Top 20)")
    ax.tick_params(axis="y", labelsize=7)

    # Plot 2: Attention Weights pro Woche
    ax = axes[1]
    weeks = [f"W{i+1}" for i in range(SEQ_LEN)]
    bars  = ax.bar(weeks, attn_weights, color="#2ecc71", edgecolor="white")
    ax.set_xlabel("Woche (W1=aelteste, W13=aktuellste)")
    ax.set_ylabel("Mittlere Attention")
    ax.set_title("Welche Wochen beachtet der Transformer?")
    for bar, val in zip(bars, attn_weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Plot 3: Predicted vs. Actual (Scatter, erste 1000 Punkte)
    ax = axes[2]
    y_flat = y_va[:1000].flatten()
    p_flat = final_val_preds[:1000].flatten()
    ax.scatter(y_flat, p_flat, alpha=0.15, s=5, color="#9b59b6")
    ax.plot([0, 5], [0, 5], "r--", linewidth=1.5, label="Perfekt")
    ax.set_xlabel("Echter Score")
    ax.set_ylabel("Vorhergesagter Score")
    ax.set_title("Predicted vs. Actual (Val)")
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 5.2)
    ax.legend(fontsize=8)
    overall_mae = mae_np(y_va, final_val_preds)
    ax.text(0.05, 0.95, f"MAE={overall_mae:.4f}", transform=ax.transAxes,
            fontsize=9, va="top", color="darkred")

    plt.tight_layout()
    plot_path = "/kaggle/working/transformer_analysis.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plots gespeichert: {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 66)
    print(f"  kaggle_transformer  |  SEQ_LEN={SEQ_LEN}  |  RECENT_YEARS={RECENT_YEARS}")
    print(f"  D_MODEL={D_MODEL}  N_HEADS={N_HEADS}  N_LAYERS={N_LAYERS}  D_FF={D_FF}")
    print(f"  Device: {DEVICE}  |  Seeds: {SEEDS}")
    print("=" * 66)

    # 1. Daten laden
    print(f"\n[1/6] Daten laden ...  [{elapsed(t0)}]")
    train_raw = load_npz(TRAIN_NPZ)
    test_raw  = load_npz(TEST_NPZ)
    print(f"   Train: {len(train_raw):,}  |  Test: {len(test_raw):,}")
    print(f"   Regionen: Train={train_raw['region_id'].nunique()}, Test={test_raw['region_id'].nunique()}")

    # 2. Feature Engineering
    print(f"\n[2/6] Feature Engineering ...  [{elapsed(t0)}]")
    weekly, test_feat, region_means, seas = compute_features(train_raw, test_raw, t0)
    del train_raw, test_raw

    STEP_FEATURES = build_step_features()
    # Fehlende Features mit 0 füllen
    for f in STEP_FEATURES + ["regional_mean_score", "regional_seasonal_mean"]:
        if f not in weekly.columns:
            weekly[f] = np.float32(0)
    all_features = STEP_FEATURES + ["regional_mean_score", "regional_seasonal_mean"]
    N_FEAT = len(all_features)
    print(f"   Features pro Zeitschritt: {N_FEAT}")

    # 3. Recent-Filter
    print(f"\n[3/6] Recent-Filter (letzte {RECENT_YEARS} Jahre) ...  [{elapsed(t0)}]")
    weekly_recent = filter_recent(weekly)
    pct = 100 * len(weekly_recent) / len(weekly)
    print(f"   {len(weekly_recent):,} / {len(weekly):,} Zeilen ({pct:.0f}%)")
    del weekly

    # 4. Sequenzen bauen
    print(f"\n[4/6] Sequenzen bauen ...  [{elapsed(t0)}]")
    rng = np.random.default_rng(RANDOM_STATE)
    all_reg     = sorted(weekly_recent["region_id"].unique())
    val_regions = set(rng.choice(all_reg, max(1, int(len(all_reg) * VAL_REGION_FRAC)), replace=False))
    tr_regions  = set(all_reg) - val_regions

    X_tr_raw, y_tr = build_sequences(weekly_recent, all_features, skip_regions=val_regions, stride=1)
    # Val: letzte 5 Fenster pro Holdout-Region — näher am Kaggle-Testszenario als alle Stride-1-Fenster
    X_va_raw, y_va = build_sequences(weekly_recent, all_features, skip_regions=tr_regions,
                                     last_n_windows=5)
    X_te_raw, test_ids = build_test_sequences(test_feat, all_features, SAMPLE_SUB)

    print(f"   Train: {len(X_tr_raw):,}  |  Val: {len(X_va_raw):,}  |  Test: {len(X_te_raw):,}")
    print(f"   Shape: ({len(X_tr_raw)}, {SEQ_LEN}, {N_FEAT})")

    # 5. Normalisierung — Scaler auf Train-Daten fitten
    print(f"\n[5/6] Normalisierung ...  [{elapsed(t0)}]")
    # Reshape für Scaler: (N × SEQ_LEN, N_FEAT)
    flat_tr = X_tr_raw.reshape(-1, N_FEAT)
    scaler  = FeatureScaler().fit(flat_tr)

    X_tr_sc = scaler.transform(flat_tr).reshape(len(X_tr_raw), SEQ_LEN, N_FEAT)
    X_va_sc = scaler.transform(X_va_raw.reshape(-1, N_FEAT)).reshape(len(X_va_raw), SEQ_LEN, N_FEAT)
    X_te_sc = scaler.transform(X_te_raw.reshape(-1, N_FEAT)).reshape(len(X_te_raw), SEQ_LEN, N_FEAT)

    # NaN → 0 nach Skalierung
    X_tr_sc = np.nan_to_num(X_tr_sc, nan=0.0)
    X_va_sc = np.nan_to_num(X_va_sc, nan=0.0)
    X_te_sc = np.nan_to_num(X_te_sc, nan=0.0)
    print(f"   Scaler: mean range [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")

    # Persistence-Baseline — letzter bekannter Score pro Val-Window
    # y_va hat Shape (N_windows, 5): pro Window den letzten Score des Input-Fensters nehmen
    # Das ist die letzte Spalte von X_va_raw (score ist nicht drin), also
    # nehmen wir den Mittelwert von y_va selbst als naive Baseline
    # Korrekt: fuer jedes Val-Window den letzten Score aus weekly_recent holen
    val_reg_list = sorted(val_regions)
    last_score_map = (weekly_recent.sort_values("ordinal")
                      .groupby("region_id")["score"].last()
                      .to_dict())

    # Persistence-Baseline: letztes Fenster pro Holdout-Region (identisch zu Val-Schema)
    baseline_vals = []
    for region, g in weekly_recent.groupby("region_id", sort=False):
        if region not in val_regions:
            continue
        g   = g.sort_values("ordinal")
        sc  = g["score"].to_numpy(np.float32)
        n   = len(g)
        if n < SEQ_LEN + PRED_LEN:
            continue
        limit = n - SEQ_LEN - PRED_LEN + 1
        start = max(0, limit - 5)  # letzte 5 Fenster — identisch zu Val-Schema
        for i in range(start, limit):
            y_vec = sc[i + SEQ_LEN: i + SEQ_LEN + PRED_LEN]
            if not np.any(np.isnan(y_vec)):
                baseline_vals.append(sc[i + SEQ_LEN - 1])

    baseline_vals = np.array(baseline_vals, dtype=np.float32)
    baseline_pred = np.column_stack([baseline_vals] * PRED_LEN)
    print(f"   Persistence-Baseline Val MAE: {mae_np(y_va, baseline_pred):.4f}")

    # 6. Multi-Seed Training
    print(f"\n[6/6] Training (Multi-Seed: {SEEDS}) ...  [{elapsed(t0)}]")
    all_val_preds = []
    best_epochs   = []
    last_val_model = None

    # Phase A: Val-Training (80% Regionen) → Early Stopping kalibriert n_epochs
    for seed in SEEDS:
        print(f"\n  ── Val-Training Seed {seed} ──")
        val_model, best_val_mae = train_transformer(X_tr_sc, y_tr, X_va_sc, y_va, N_FEAT, seed)
        val_preds = predict(val_model, X_va_sc)
        print(f"  Seed {seed} → Val MAE = {mae_np(y_va, val_preds):.4f}  "
              f"(best epoch {val_model._best_epoch_})")
        all_val_preds.append(val_preds)
        best_epochs.append(val_model._best_epoch_)
        last_val_model = val_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    final_val_preds = np.mean(all_val_preds, axis=0)
    ensemble_val_mae = mae_np(y_va, final_val_preds)
    n_final_epochs   = max(5, int(np.mean(best_epochs)))
    print(f"\n  Ensemble Val MAE ({len(SEEDS)} Seeds): {ensemble_val_mae:.4f}")
    print(f"  Beste Epochen pro Seed: {best_epochs}  → Final Training: {n_final_epochs} Epochen")
    for wk in range(PRED_LEN):
        wk_mae = mae_np(y_va[:, wk], final_val_preds[:, wk])
        print(f"    Woche {wk+1}: MAE = {wk_mae:.4f}")

    # Phase B: Final Training auf allen Daten (Train + Val) → keine Regionen ausgelassen
    print(f"\n  ── Final Training auf allen {len(all_reg)} Regionen ──")
    X_all_sc = np.vstack([X_tr_sc, X_va_sc])
    y_all    = np.vstack([y_tr,    y_va])

    all_test_preds = []
    for seed in SEEDS:
        print(f"\n  Seed {seed} — Final Training ({n_final_epochs} Epochs) ...")
        final_model = train_fixed_epochs(X_all_sc, y_all, N_FEAT, seed, n_final_epochs)
        test_preds  = predict(final_model, X_te_sc)
        print(f"  Seed {seed} → Test Pred mean={test_preds.mean():.4f}  "
              f"min={test_preds.min():.4f}  max={test_preds.max():.4f}")
        all_test_preds.append(test_preds)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    final_test_preds = np.mean(all_test_preds, axis=0)

    # ── Analyse (nutzt Val-Modell vom letzten Seed — kein extra Training) ───────
    print(f"\n{'─'*66}")
    print("  ANALYSE")
    print(f"{'─'*66}")

    feat_imp_df  = analyze_feature_importance(
        last_val_model, X_va_sc, y_va, all_features, n_samples=500
    )
    attn_weights = analyze_attention(last_val_model, X_va_sc, n_samples=300)
    analyze_errors(y_va, final_val_preds)
    save_plots(feat_imp_df, attn_weights, y_va, final_val_preds, all_features)

    # Feature Importance CSV speichern
    feat_imp_path = "/kaggle/working/feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_path, index=False)
    print(f"  Feature Importance CSV: {feat_imp_path}")

    # Submission erstellen — Format exakt wie sample_submission.csv
    print(f"\n  Submission erstellen ...")

    # Reihenfolge bereits korrekt (build_test_sequences folgt sample_submission)
    out = pd.DataFrame({"region_id": test_ids.astype(str)})
    for k in range(PRED_LEN):
        out[f"pred_week{k+1}"] = np.clip(final_test_preds[:, k], 0, 5)

    # Sanity Checks
    assert len(out) == 2248, f"Zeilenzahl falsch: {len(out)}"
    pred_mean = out[[f"pred_week{k+1}" for k in range(PRED_LEN)]].values.mean()
    assert pred_mean > 0.01, f"Predictions nahe 0 — etwas stimmt nicht (mean={pred_mean:.4f})"

    out.to_csv(OUT_PATH, index=False)
    print(f"  Spalten:   {list(out.columns)}")
    print(f"  Zeilen:    {len(out):,}")
    print(f"  Pred mean: {pred_mean:.4f}  min: {out['pred_week1'].min():.4f}  max: {out['pred_week1'].max():.4f}")
    print(f"  Beispiel:  {dict(out.iloc[0])}")

    print(f"\n{'═'*66}")
    print(f"  FERTIG")
    print(f"  Ensemble Val MAE:  {ensemble_val_mae:.4f}")
    print(f"  Submission:        {OUT_PATH}  ({len(out):,} Zeilen)")
    print(f"  Laufzeit:          {elapsed(t0)}")
    print(f"{'═'*66}")


if __name__ == "__main__":
    main()