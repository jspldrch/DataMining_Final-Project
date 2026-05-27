"""
Feature engineering for weather score forecasting.

Used by notebooks/03_preprocessing.ipynb and 04_modeling.ipynb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLS = [
    "prec",
    "surf_pre",
    "humidity",
    "tmp",
    "dp_tmp",
    "wb_tmp",
    "tmp_max",
    "tmp_min",
    "tmp_range",
    "surf_tmp",
    "wind",
    "wind_max",
    "wind_min",
    "wind_range",
]

LAG_COLS = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre"]
LAGS = [1, 3, 7, 14, 21]

ROLL_COLS = ["prec", "wind", "tmp"]
ROLL_WINDOWS = [7, 14, 30, 60, 90]
ROLL_MIN_PERIODS = 3


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Split date string into year, month, day (no pd.to_datetime)."""
    df = df.copy()
    if "year" not in df.columns:
        parts = df["date"].astype(str).str.split("-", expand=True)
        df["year"] = parts[0].astype(int)
        df["month"] = parts[1].astype(int)
        df["day"] = parts[2].astype(int)
    return df


def add_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ordinal"] = df["year"] * 372 + df["month"] * 31 + df["day"]
    return df


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    if "ordinal" not in df.columns:
        df = add_ordinal(df)
    return df.sort_values(["region_id", "ordinal", "date"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    return df


def add_lag_features(df: pd.DataFrame, region_col: str = "region_id") -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(region_col, sort=False)
    for col in LAG_COLS:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = grouped[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    region_col: str = "region_id",
    min_periods: int = ROLL_MIN_PERIODS,
) -> pd.DataFrame:
    """Rolling stats use prior days only (shift(1) before window)."""
    df = df.copy()
    grouped = df.groupby(region_col, sort=False)
    for col in ROLL_COLS:
        prior = grouped[col].shift(1)
        for window in ROLL_WINDOWS:
            roller = prior.groupby(df[region_col], sort=False)
            df[f"{col}_roll{window}_mean"] = roller.transform(
                lambda s: s.rolling(window, min_periods=min_periods).mean()
            )
            df[f"{col}_roll{window}_std"] = roller.transform(
                lambda s: s.rolling(window, min_periods=min_periods).std()
            )
            df[f"{col}_roll{window}_max"] = roller.transform(
                lambda s: s.rolling(window, min_periods=min_periods).max()
            )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline on a sorted panel (train + test combined)."""
    df = parse_dates(df)
    df = add_ordinal(df)
    df = sort_panel(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def feature_columns(include_region: bool = True) -> list[str]:
    """Column names for model training (excludes target and metadata)."""
    lag_names = [f"{c}_lag{lag}" for c in LAG_COLS for lag in LAGS]
    roll_names = []
    for col in ROLL_COLS:
        for window in ROLL_WINDOWS:
            roll_names.extend(
                [
                    f"{col}_roll{window}_mean",
                    f"{col}_roll{window}_std",
                    f"{col}_roll{window}_max",
                ]
            )
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos"]
    cols = list(WEATHER_COLS) + lag_names + roll_names + calendar
    if include_region:
        cols = ["region_id"] + cols
    return cols


def add_persistence_baseline(df: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
    """Score from ~7 days ago (forward-filled within region)."""
    df = df.copy()
    filled = df.groupby("region_id", sort=False)["score"].ffill()
    df["score_persist7"] = filled.groupby(df["region_id"], sort=False).shift(lag_days)
    return df
