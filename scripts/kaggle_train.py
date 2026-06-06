"""
kaggle_train.py  –  Schnelles Training auf Kaggle (~5-10 Min. mit GPU).

Erfordert: data/precomputed/ Ordner als Kaggle Dataset hochgeladen.
Lädt alles vorberechnete und trainiert nur noch LGB + DL.

Input Kaggle Dataset Pfad: /kaggle/input/precomputed-drought/
(Name anpassen je nachdem wie du es hochgeladen hast)

Usage auf Kaggle:
    Einfach Notebook-Zelle mit dem Code ausführen.
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

# ─── Paths ────────────────────────────────────────────────────────────────────
_ON_KAGGLE = Path("/kaggle/input").exists()

if _ON_KAGGLE:
    # Sucht automatisch nach dem precomputed Ordner
    _candidates = list(Path("/kaggle/input").rglob("meta.npz"))
    if not _candidates:
        raise FileNotFoundError("meta.npz nicht gefunden. Precomputed Dataset hochgeladen?")
    PRECOMP_DIR = _candidates[0].parent
    OUT_PATH    = Path("/kaggle/working/submission_kaggle.csv")
    SAMPLE_SUB_CANDIDATES = list(Path("/kaggle/input").rglob("sample_submission.csv"))
    SAMPLE_SUB  = SAMPLE_SUB_CANDIDATES[0] if SAMPLE_SUB_CANDIDATES else None
else:
    PRECOMP_DIR = Path(__file__).parent.parent / "data" / "precomputed"
    OUT_PATH    = Path(__file__).parent.parent / "outputs" / "submission_kaggle.csv"
    SAMPLE_SUB  = Path(__file__).parent.parent / "resources" / "sample_submission.csv"
    OUT_PATH.parent.mkdir(exist_ok=True)

print(f"Precomputed dir: {PRECOMP_DIR}")
print(f"Device: {DEVICE if TORCH_AVAILABLE else 'CPU (no torch)'}")

# ─── Config ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_LGB_EST    = 1000
SEQ_LEN      = 26
D_MODEL      = 64
N_HEADS      = 4
N_TF_LAYERS  = 2
DL_EPOCHS    = 60
DL_BATCH     = 512
DL_LR        = 3e-4
DL_WD        = 1e-3
DL_PATIENCE  = 15

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]
N_WEATHER = len(WEATHER_COLS)

LGB_PARAMS = dict(
    objective="regression", metric="mae",
    n_estimators=N_LGB_EST, learning_rate=0.04, num_leaves=127,
    min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1,
)


def elapsed(t0):
    s = time.time() - t0
    return f"{s/60:.1f} Min." if s >= 60 else f"{s:.0f}s"

def mae_np(yt, yp):
    return float(np.mean(np.abs(np.clip(yp, 0, 5) - yt)))

def show_mae(name, yt, yp):
    print(f"  {name:<50s}  MAE = {mae_np(yt, yp):.4f}")


# ─── DL Model ────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    def _sinusoidal_pos_enc(seq_len, d_model):
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe  = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    class ParallelDroughtModel(nn.Module):
        def __init__(self, n_features, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_TF_LAYERS):
            super().__init__()
            self.seq_embed = nn.Linear(N_WEATHER, d_model)
            self.register_buffer("pos_enc", _sinusoidal_pos_enc(SEQ_LEN, d_model))
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, 0.1, batch_first=True, norm_first=True),
                num_layers=n_layers,
            )
            self.temporal_norm = nn.LayerNorm(d_model)
            self.feat_mlp = nn.Sequential(
                nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, d_model),
            )
            self.fusion = nn.Sequential(
                nn.Linear(d_model*2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 5),
            )

        def forward(self, seq, feat):
            x_t = self.seq_embed(seq) + self.pos_enc
            x_t = self.temporal_norm(self.transformer(x_t)).mean(dim=1)
            x_f = self.feat_mlp(feat)
            return torch.clamp(self.fusion(torch.cat([x_t, x_f], dim=1)), 0.0, 5.0)

    class DroughtDataset(Dataset):
        def __init__(self, seqs, feats, ys):
            self.s = torch.from_numpy(seqs)
            self.f = torch.from_numpy(feats)
            self.y = torch.from_numpy(ys)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.s[i], self.f[i], self.y[i]


def train_dl(tr_seqs, tr_feats, tr_ys, va_seqs, va_feats, va_ys, n_features):
    if not TORCH_AVAILABLE: return None
    print(f"  DL: device={DEVICE}  train={len(tr_ys):,}  val={len(va_ys)}")
    model     = ParallelDroughtModel(n_features).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DL_LR, weight_decay=DL_WD)
    steps     = math.ceil(len(tr_ys)/DL_BATCH) * DL_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, DL_LR, total_steps=steps, pct_start=0.1, final_div_factor=100)
    criterion = nn.HuberLoss(delta=1.0)
    loader    = DataLoader(DroughtDataset(tr_seqs, tr_feats, tr_ys), DL_BATCH, shuffle=True, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    va_s = torch.from_numpy(va_seqs).to(DEVICE)
    va_f = torch.from_numpy(va_feats).to(DEVICE)
    va_y = torch.from_numpy(va_ys)
    best_mae, best_state, patience = 999.0, None, 0
    for epoch in range(1, DL_EPOCHS+1):
        model.train()
        total = 0.0
        for sb, fb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(sb.to(DEVICE), fb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            vm = mae_np(va_y.numpy(), model(va_s, va_f).cpu().numpy())
        if epoch % 5 == 0 or epoch <= 3:
            print(f"    Epoch {epoch:3d}/{DL_EPOCHS}  loss={total/len(loader):.4f}  val_mae={vm:.4f}")
        if vm < best_mae:
            best_mae = vm; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= DL_PATIENCE:
                print(f"    Early stop @ {epoch}  best={best_mae:.4f}"); break
    model.load_state_dict(best_state)
    return model


def predict_dl(model, seqs, feats):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 2048):
            out.append(model(torch.from_numpy(seqs[i:i+2048]).to(DEVICE),
                             torch.from_numpy(feats[i:i+2048]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ─── LGB ─────────────────────────────────────────────────────────────────────

def train_lgb(X_tr, y_tr, r_tr, X_va, y_va, r_va, n_trees=None):
    X_df = pd.DataFrame(X_tr); X_df["region_id"] = pd.Categorical(r_tr)
    X_va_df = pd.DataFrame(X_va); X_va_df["region_id"] = pd.Categorical(r_va)
    models = []
    for week in range(5):
        n = (n_trees[week] if n_trees else None) or N_LGB_EST
        p = dict(LGB_PARAMS, random_state=RANDOM_STATE+week, n_estimators=n)
        m = lgb.LGBMRegressor(**p)
        kw = dict(categorical_feature=["region_id"])
        if X_va is not None:
            kw["eval_set"] = [(X_va_df, y_va[:,week].ravel())]
            kw["eval_metric"] = "mae"
            kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        m.fit(X_df, y_tr[:,week].ravel(), **kw)
        models.append(m)
    return models, X_df, X_va_df

def predict_lgb(models, X_df):
    return np.clip(np.column_stack([m.predict(X_df) for m in models]), 0, 5).astype(np.float32)

def blend_search(yt, pa, pb):
    best_mae, best_a = 999.0, 0.5
    for a in [round(x*0.05,2) for x in range(1,20)]:
        m = mae_np(yt, a*pa + (1-a)*pb)
        if m < best_mae: best_mae, best_a = m, a
    return best_a, best_mae


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 64)
    print("  Kaggle Fast Training  (pre-computed features)")
    print("=" * 64)

    # 1. Load pre-computed data
    print("\n[1/4] Loading pre-computed arrays ...")
    full = np.load(PRECOMP_DIR / "features_full.npz", allow_pickle=True)
    redu = np.load(PRECOMP_DIR / "features_reduced.npz")
    tgt  = np.load(PRECOMP_DIR / "targets.npz")
    seqs = np.load(PRECOMP_DIR / "sequences.npz")
    meta = np.load(PRECOMP_DIR / "meta.npz", allow_pickle=True)

    X_tr   = full["X_tr"];  r_tr = full["r_tr"]
    X_va   = full["X_va"];  r_va = full["r_va"]
    X_all  = full["X_all"]; r_all= full["r_all"]
    X_test = full["X_test"]; test_region_ids = full["test_region_ids"]

    y_tr  = tgt["y_tr"]; y_va = tgt["y_va"]; y_all = tgt["y_all"]

    tr_seqs  = seqs["tr_seqs"];  tr_feats  = seqs["tr_feats"];  tr_ys   = seqs["tr_ys"]
    va_seqs  = seqs["va_seqs"];  va_feats  = seqs["va_feats"]
    te_seqs  = seqs["te_seqs"];  te_feats_tab = seqs["te_feats_tab"]

    feat_names_reduced = list(meta["feature_names_reduced"])
    n_features_reduced = len(feat_names_reduced)

    print(f"   X_tr={X_tr.shape}  X_va={X_va.shape}  reduced={redu['X_tr'].shape[1]} features  [{elapsed(t0)}]")

    # 2. LGB training
    print("\n[2/4] Training LightGBM ...")
    lgb_models, X_tr_df, X_va_df = train_lgb(X_tr, y_tr, r_tr, X_va, y_va, r_va)
    lgb_val = predict_lgb(lgb_models, X_va_df)
    show_mae("LightGBM (val)", y_va, lgb_val)

    # 3. DL training (if torch available)
    dl_model = None
    if TORCH_AVAILABLE:
        print("\n[3/4] Training DL Transformer+MLP ...")
        dl_model = train_dl(tr_seqs, tr_feats, tr_ys, va_seqs, va_feats, y_va, n_features_reduced)
        if dl_model:
            dl_val = predict_dl(dl_model, va_seqs, va_feats)
            show_mae("DL Hybrid (val)", y_va, dl_val)
            lgb_w, blend_mae = blend_search(y_va, lgb_val, dl_val)
            dl_w = round(1-lgb_w, 2)
            print(f"  Blend: LGB={lgb_w:.2f}  DL={dl_w:.2f}  MAE={blend_mae:.4f}")
        else:
            lgb_w, dl_w = 1.0, 0.0
    else:
        lgb_w, dl_w = 1.0, 0.0
        print("\n[3/4] DL skipped (no torch).")

    # 4. Final training + predictions
    print("\n[4/4] Final training + predictions ...")
    X_all_df = pd.DataFrame(X_all); X_all_df["region_id"] = pd.Categorical(r_all)
    n_lgb = [int(getattr(m,"best_iteration_",None) or N_LGB_EST) for m in lgb_models]
    final_lgb_models, X_all_df, _ = train_lgb(X_all, y_all, r_all, None, None, None, n_lgb)

    X_test_df = pd.DataFrame(X_test); X_test_df["region_id"] = pd.Categorical(test_region_ids)
    lgb_test = predict_lgb(final_lgb_models, X_test_df)

    if dl_model is not None and dl_w > 0:
        print("  Retraining DL on all data ...")
        all_seqs_data = np.load(PRECOMP_DIR / "sequences.npz")
        final_dl = train_dl(
            all_seqs_data["tr_seqs"], all_seqs_data["tr_feats"], all_seqs_data["tr_ys"],
            va_seqs, va_feats, y_va, n_features_reduced
        )
        dl_test    = predict_dl(final_dl, te_seqs, te_feats_tab)
        test_preds = lgb_w * lgb_test + dl_w * dl_test
    else:
        test_preds = lgb_test

    # Build submission
    sub = pd.DataFrame({"region_id": test_region_ids})
    for k in range(5):
        sub[f"pred_week{k+1}"] = test_preds[:, k]

    if SAMPLE_SUB and Path(SAMPLE_SUB).exists():
        template = pd.read_csv(SAMPLE_SUB)
        sub = template[["region_id"]].merge(sub, on="region_id", how="left")
        for col in [f"pred_week{k+1}" for k in range(5)]:
            sub[col] = sub[col].fillna(0.0)

    sub.to_csv(OUT_PATH, index=False)
    print(f"\n{'='*64}")
    print(f"  Saved: {OUT_PATH}")
    print(f"  Rows={len(sub):,}  Total={elapsed(t0)}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
