"""
Feature engineering v2 — extends v1 for Kaggle MAE ≤ 0.8 push.

See docs/10_PREPROCESSING_V2.md for diff vs scripts/features.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.features import (
    WEATHER_COLS,
    add_persistence_baseline,
    build_features,
    feature_columns,
    parse_dates,
)

# Extra score history (weekly rhythm + 5-week horizon)
SCORE_LAGS = [7, 14, 21, 28, 35]

# 91-day test-window summaries (constant per region on test rows)
TEST91_STATS = [
    ("tmp_range", "mean"),
    ("tmp_range", "max"),
    ("tmp_max", "mean"),
    ("tmp_max", "max"),
    ("prec", "mean"),
    ("prec", "max"),
    ("surf_pre", "mean"),
    ("surf_pre", "min"),
    ("wind_max", "mean"),
    ("wind_max", "max"),
]

REGION_STAT_COLS = [
    "region_score_mean",
    "region_score_median",
    "region_score_std",
]


def add_score_lag_features(df: pd.DataFrame, region_col: str = "region_id") -> pd.DataFrame:
    """Lags of forward-filled score (train history visible into test panel)."""
    df = df.copy()
    filled = df.groupby(region_col, sort=False)["score"].ffill()
    grouped = filled.groupby(df[region_col], sort=False)
    for lag in SCORE_LAGS:
        df[f"score_lag{lag}"] = grouped.shift(lag)
    return df


def merge_region_stats(df: pd.DataFrame, region_stats: pd.DataFrame) -> pd.DataFrame:
    return df.merge(region_stats, on="region_id", how="left")


def add_test91_summary(test_feat: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    """Broadcast 91-day weather aggregates to all test rows of this region."""
    if test_raw.empty or test_feat.empty:
        return test_feat
    test_feat = test_feat.copy()
    for col, stat in TEST91_STATS:
        name = f"test91_{stat}_{col}"
        if col not in test_raw.columns:
            continue
        val = getattr(test_raw[col], stat)()
        test_feat[name] = float(val) if pd.notna(val) else np.nan
    return test_feat


def build_features_v2(df: pd.DataFrame, region_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """v1 weather/lags + score lags + persistence + optional region stats."""
    df = build_features(df)
    df = add_persistence_baseline(df, lag_days=7)
    df = add_score_lag_features(df)
    if region_stats is not None:
        df = merge_region_stats(df, region_stats)
    return df


def feature_columns_v2(include_region: bool = True) -> list[str]:
    """All model columns for v2 (includes v1 + new blocks)."""
    score_cols = ["score_persist7"] + [f"score_lag{lag}" for lag in SCORE_LAGS]
    test91_cols = [f"test91_{stat}_{col}" for col, stat in TEST91_STATS]
    base = feature_columns(include_region=include_region)
    extra = score_cols + REGION_STAT_COLS + test91_cols
    return list(dict.fromkeys(base + extra))


def meta_train_cols_v2() -> list[str]:
    return [
        "region_id",
        "date",
        "year",
        "month",
        "day",
        "ordinal",
        "score",
        "score_persist7",
    ]


def save_columns_v2(df: pd.DataFrame, *, labeled: bool) -> list[str]:
    meta = meta_train_cols_v2() if labeled else [
        "region_id",
        "date",
        "year",
        "month",
        "day",
        "ordinal",
    ]
    feats = feature_columns_v2()
    return [c for c in list(dict.fromkeys(meta + feats)) if c in df.columns]
