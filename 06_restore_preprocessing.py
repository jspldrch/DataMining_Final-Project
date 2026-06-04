import json

with open("notebooks/preprocessing.ipynb", "r") as f:
    prep_nb = json.load(f)

for cell in prep_nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    
    if "def build_features_v2" in source and "def feature_columns_v2" in source:
        # This is the corrupted cell, replace it with the full version
        new_source = """# --- NATIVE FEATURE ENGINEERING (v3) ---
import numpy as np
import pandas as pd

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp",
    "wb_tmp", "tmp_max", "tmp_min", "tmp_range",
    "surf_tmp", "wind", "wind_max", "wind_min", "wind_range",
]

LAG_COLS = ["tmp_range", "tmp_max", "tmp", "prec", "wind", "surf_pre"]
LAGS = [1, 3, 7, 14, 21]

ROLL_COLS = ["prec", "wind", "tmp"]
ROLL_WINDOWS = [7, 14, 28]

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
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

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("region_id", sort=False)
    for col in LAG_COLS:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = grouped[col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("region_id", sort=False)
    
    # 1. Standard rolling means/max
    for col in ROLL_COLS:
        prior = grouped[col].shift(1)
        for window in ROLL_WINDOWS:
            roller = prior.groupby(df["region_id"], sort=False)
            df[f"{col}_roll{window}_mean"] = roller.transform(lambda s: s.rolling(window, min_periods=3).mean())
            df[f"{col}_roll{window}_max"] = roller.transform(lambda s: s.rolling(window, min_periods=3).max())
            
            # Special: Accumulated Precipitation
            if col == "prec":
                df[f"prec_roll{window}_sum"] = roller.transform(lambda s: s.rolling(window, min_periods=3).sum())
    
    # 2. 90-day precipitation and temp anomaly
    prior_prec = grouped["prec"].shift(1)
    prior_tmp = grouped["tmp"].shift(1)
    
    roller_prec = prior_prec.groupby(df["region_id"], sort=False)
    roller_tmp = prior_tmp.groupby(df["region_id"], sort=False)
    
    df["prec_roll90_sum"] = roller_prec.transform(lambda s: s.rolling(90, min_periods=10).sum())
    df["tmp_roll90_mean"] = roller_tmp.transform(lambda s: s.rolling(90, min_periods=10).mean())
    df["tmp_mean_diff_90"] = df["tmp"] - df["tmp_roll90_mean"]
    
    # 3. Cross Features
    df["storm_proxy"] = df["wind_max"] * df["prec"]
    
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_dates(df)
    df = add_ordinal(df)
    df = sort_panel(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df

def feature_columns(include_region: bool = True) -> list[str]:
    lag_names = [f"{c}_lag{lag}" for c in LAG_COLS for lag in LAGS]
    roll_names = []
    for col in ROLL_COLS:
        for window in ROLL_WINDOWS:
            roll_names.extend([f"{col}_roll{window}_mean", f"{col}_roll{window}_max"])
            if col == "prec":
                roll_names.append(f"prec_roll{window}_sum")
                
    roll_names.extend(["prec_roll90_sum", "tmp_roll90_mean", "tmp_mean_diff_90", "storm_proxy"])
    
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos"]
    cols = list(WEATHER_COLS) + lag_names + roll_names + calendar
    if include_region:
        cols = ["region_id"] + cols
    return cols

def add_persistence_baseline(df: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
    df = df.copy()
    filled = df.groupby("region_id", sort=False)["score"].ffill()
    df["score_persist7"] = filled.groupby(df["region_id"], sort=False).shift(lag_days)
    return df

def build_features_v2(df: pd.DataFrame, region_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    df = build_features(df)
    df = add_persistence_baseline(df, lag_days=7)
    if region_stats is not None:
        df = pd.merge(df, region_stats, on="region_id", how="left")
    return df

def feature_columns_v2(include_region: bool = True) -> list[str]:
    score_cols = ["score_persist7", "region_score_mean", "region_score_median", "region_score_std"]
    base = feature_columns(include_region=include_region)
    extra = score_cols
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
"""
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

with open("notebooks/preprocessing.ipynb", "w") as f:
    json.dump(prep_nb, f, indent=1)

print("Restored cell in preprocessing.ipynb")
