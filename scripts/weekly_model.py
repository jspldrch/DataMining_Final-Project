"""
5-week disaster severity prediction (Kaggle format).

Why weekly aggregation in 04 (not in 03)?
- 03 writes *daily* labeled rows (~782/region) with full weather+lags from the panel.
- Kaggle asks for pred_week1..5 → targets are weekly; daily sliding windows explode RAM (~1.3M+ samples).
- We collapse to one row per (region, ordinal//7) here so 03 stays streaming-safe and 04 matches the task.

Submission: pred_week1..pred_week5, one row per region, MAE metric.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.parallel_util import default_workers, run_parallel_map

WEEK_COLS = [f"pred_week{k}" for k in range(1, 6)]
WEEK_BUCKET = 7  # ordinal days per bucket (matches ~7-day score rhythm in EDA)


def clip_scores(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 5.0)


def slim_for_modeling(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Keep only columns needed for modeling (lower RAM after parquet load)."""
    cols = list(dict.fromkeys(feature_cols + ["score", "ordinal"]))
    return df[[c for c in cols if c in df.columns]].copy()


def daily_to_weekly(labeled: pd.DataFrame) -> pd.DataFrame:
    """
    One row per region per 7-day ordinal bucket (last labeled day in bucket).

    Features come from that day (already computed on the full panel in 03).
    """
    df = labeled.sort_values(["region_id", "ordinal"])
    df = df.assign(_week=df["ordinal"] // WEEK_BUCKET)
    idx = df.groupby(["region_id", "_week"], sort=False)["ordinal"].idxmax()
    weekly = df.loc[idx].drop(columns="_week")
    return weekly.reset_index(drop=True)


def _numeric_and_cat_cols(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    num = [c for c in feature_cols if c != "region_id"]
    cat = ["region_id"] if "region_id" in feature_cols else []
    return num, cat


def _windows_from_weekly_group(
    g: pd.DataFrame,
    num_cols: list[str],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Per region: X[i] → scores at weeks i+1..i+5. Returns float32 arrays."""
    g = g.sort_values("ordinal")
    n = len(g)
    if n < 6:
        return None, None
    X_num = g[num_cols].to_numpy(dtype=np.float32)
    scores = g["score"].to_numpy(dtype=np.float32)
    n_win = n - 5
    # y[j] = scores[j+1 : j+6]
    y_out = np.lib.stride_tricks.sliding_window_view(scores[1:], 5)[:n_win]
    X_out = X_num[:n_win]
    return X_out, y_out


def _assemble_X(
    X_num: np.ndarray,
    regions: list,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    X_df = pd.DataFrame(X_num, columns=num_cols)
    if cat_cols:
        X_df["region_id"] = pd.Categorical(regions)
    return X_df


def _sliding_region_worker(
    args: tuple[object, pd.DataFrame, list[str]],
) -> tuple[np.ndarray, np.ndarray, object, list[dict]] | None:
    region, g, num_cols = args
    X_out, y_out = _windows_from_weekly_group(g, num_cols)
    if X_out is None:
        return None
    n_win = len(y_out)
    ordinals = g["ordinal"].to_numpy()
    meta = [{"region_id": region, "anchor_ordinal": int(ordinals[j])} for j in range(n_win)]
    return X_out, y_out, region, meta


def _collect_sliding_results(
    results: list,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    regions: list = []
    meta_rows: list[dict] = []

    for item in results:
        if item is None:
            continue
        X_out, y_out, region, meta = item
        n_win = len(y_out)
        X_parts.append(X_out)
        y_parts.append(y_out)
        regions.extend([region] * n_win)
        meta_rows.extend(meta)

    if not X_parts:
        raise ValueError("Keine Sliding-Window-Samples — zu wenig Wochen pro Region?")

    X_num = np.vstack(X_parts)
    y_all = np.vstack(y_parts)
    X_df = _assemble_X(X_num, regions, num_cols, cat_cols)
    return X_df, y_all, pd.DataFrame(meta_rows)


def build_sliding_samples(
    labeled: pd.DataFrame,
    feature_cols: list[str],
    *,
    already_weekly: bool = False,
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    From weekly rows: feature vector at week i → y = scores at weeks i+1..i+5.
    """
    w = labeled if already_weekly else daily_to_weekly(labeled)
    num_cols, cat_cols = _numeric_and_cat_cols(feature_cols)
    n_workers = n_workers if n_workers is not None else default_workers()

    groups = [(r, g.sort_values("ordinal"), num_cols) for r, g in w.groupby("region_id", sort=False)]

    if n_workers <= 1:
        results = [_sliding_region_worker(t) for t in groups]
    else:
        results = run_parallel_map(_sliding_region_worker, groups, n_workers=n_workers)

    return _collect_sliding_results(results, num_cols, cat_cols)


def build_region_holdout(
    labeled: pd.DataFrame,
    feature_cols: list[str],
    val_region_frac: float = 0.2,
    seed: int = 42,
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    """
    Train: sliding windows on train regions (weekly rows).
    Val: last 5 weekly scores per held-out region (one feature row each).
    """
    w = daily_to_weekly(labeled)
    regions = sorted(w["region_id"].unique())
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(regions) * val_region_frac))
    val_regions = set(rng.choice(regions, size=n_val, replace=False))

    train_sub = w[~w["region_id"].isin(val_regions)]
    X_tr, y_tr, _ = build_sliding_samples(
        train_sub, feature_cols, already_weekly=True, n_workers=n_workers
    )

    num_cols, cat_cols = _numeric_and_cat_cols(feature_cols)
    vx_num: list[np.ndarray] = []
    vy: list[np.ndarray] = []
    v_regions: list = []

    val_sorted = sorted(val_regions)
    for region in val_sorted:
        g = w.loc[w["region_id"] == region].sort_values("ordinal")
        if len(g) < 6:
            continue
        vx_num.append(g.iloc[-6][num_cols].to_numpy(dtype=np.float32))
        vy.append(g.iloc[-5:]["score"].to_numpy(dtype=np.float32))
        v_regions.append(region)

    X_va = pd.DataFrame(np.vstack(vx_num), columns=num_cols)
    if cat_cols:
        X_va["region_id"] = pd.Categorical(v_regions)
    y_va = np.vstack(vy)
    return X_tr, y_tr, X_va, y_va, val_sorted


def weekly_summary(daily_labeled: pd.DataFrame) -> dict:
    """Diagnostics after daily_to_weekly (for notebook prints)."""
    w = daily_to_weekly(daily_labeled)
    per_region = w.groupby("region_id", sort=False).size()
    return {
        "daily_rows": len(daily_labeled),
        "weekly_rows": len(w),
        "regions": w["region_id"].nunique(),
        "median_weeks_per_region": float(per_region.median()),
    }


def test_features_last_row(test_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One feature row per region: last day of the 91-day test window (predict next 5 weeks)."""
    X = (
        test_features.sort_values(["region_id", "ordinal"])
        .groupby("region_id", sort=False)
        .tail(1)[feature_cols]
        .reset_index(drop=True)
    )
    if "region_id" in X.columns:
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
