"""
5-week disaster severity prediction (Kaggle format).

Task: after 91 days of weather per region, predict scores for the next 5 weeks.
Submission columns: pred_week1 .. pred_week5 (one row per region).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WEEK_COLS = [f"pred_week{k}" for k in range(1, 6)]


def clip_scores(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 5.0)


def build_sliding_samples(
    labeled: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    From weekly labeled rows: X at week i → y = scores at weeks i+1..i+5.
    Returns X_df, y (n, 5), meta with region_id.
    """
    xs, ys, meta = [], [], []
    for region, g in labeled.groupby("region_id", sort=False):
        g = g.sort_values("ordinal")
        if len(g) < 6:
            continue
        for i in range(len(g) - 5):
            xs.append(g.iloc[i][feature_cols])
            ys.append(g.iloc[i + 1 : i + 6]["score"].to_numpy(dtype=float))
            meta.append({"region_id": region, "anchor_ordinal": int(g.iloc[i]["ordinal"])})

    if not xs:
        raise ValueError("Keine Sliding-Window-Samples — train_labeled zu klein?")

    X_df = pd.DataFrame(xs).reset_index(drop=True)
    X_df["region_id"] = X_df["region_id"].astype("category")
    return X_df, np.vstack(ys), pd.DataFrame(meta)


def build_region_holdout(
    labeled: pd.DataFrame,
    feature_cols: list[str],
    val_region_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    """
    Train: sliding windows on train regions.
    Val: last 5 weekly scores per held-out region (Kaggle-style block).
    """
    regions = sorted(labeled["region_id"].unique())
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(regions) * val_region_frac))
    val_regions = set(rng.choice(regions, size=n_val, replace=False))
    train_regions = [r for r in regions if r not in val_regions]

    tr_parts = [labeled[labeled["region_id"] == r] for r in train_regions]
    train_sub = pd.concat(tr_parts, ignore_index=True) if tr_parts else labeled.iloc[0:0]
    X_tr, y_tr, _ = build_sliding_samples(train_sub, feature_cols)

    vx, vy, v_regions = [], [], []
    for region in val_regions:
        g = labeled[labeled["region_id"] == region].sort_values("ordinal")
        if len(g) < 6:
            continue
        vx.append(g.iloc[-6][feature_cols])
        vy.append(g.iloc[-5:]["score"].to_numpy(dtype=float))
        v_regions.append(region)

    X_va = pd.DataFrame(vx).reset_index(drop=True)
    X_va["region_id"] = X_va["region_id"].astype("category")
    y_va = np.vstack(vy)
    return X_tr, y_tr, X_va, y_va, sorted(val_regions)


def test_features_last_row(test_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One feature row per region: last day of the 91-day test window."""
    X = (
        test_features.sort_values(["region_id", "ordinal"])
        .groupby("region_id", sort=False)
        .tail(1)[feature_cols]
        .reset_index(drop=True)
    )
    X["region_id"] = X["region_id"].astype("category")
    return X


def mae_kaggle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE over all regions × 5 weeks (Kaggle-style scalar)."""
    return float(np.mean(np.abs(clip_scores(y_pred) - y_true)))


def predict_week_columns(models: list, X: pd.DataFrame) -> np.ndarray:
    """Shape (n_regions, 5)."""
    preds = np.column_stack([clip_scores(m.predict(X)) for m in models])
    return preds


def submission_frame(region_ids: pd.Series, preds: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"region_id": region_ids.values})
    for k, col in enumerate(WEEK_COLS):
        out[col] = preds[:, k]
    return out


def find_sample_submission(project_root: Path, data_dir: Path | None = None) -> Path | None:
    """Locate Kaggle template CSV (repo, Drive data, or legacy folder)."""
    candidates = [
        project_root / "resources" / "sample_submission.csv",
        project_root / "data-mining-2026-final-project" / "sample_submission.csv",
        project_root / "data" / "sample_submission.csv",
    ]
    if data_dir is not None:
        candidates.insert(0, data_dir / "sample_submission.csv")
    for path in candidates:
        if path.exists():
            return path
    return None


def build_submission_template(region_ids: pd.Series) -> pd.DataFrame:
    """Kaggle columns with one row per region (sorted), preds filled with 0."""
    regions = region_ids.drop_duplicates().sort_values().reset_index(drop=True)
    out = pd.DataFrame({"region_id": regions})
    for col in WEEK_COLS:
        out[col] = 0.0
    return out


def align_to_sample_submission(
    submission: pd.DataFrame,
    template: Path | pd.DataFrame,
) -> pd.DataFrame:
    """Same region order and columns as sample_submission.csv."""
    if isinstance(template, Path):
        base = pd.read_csv(template)
    else:
        base = template.copy()
    merged = base[["region_id"]].merge(submission, on="region_id", how="left")
    for col in WEEK_COLS:
        merged[col] = merged[col].fillna(0.0)
    return merged
