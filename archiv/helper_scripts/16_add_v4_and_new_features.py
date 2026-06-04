import json
import re

with open("notebooks/pipeline_combined.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    source = "".join(cell["source"])
    
    if 'def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:' in source:
        # We will replace the entire cell content up to feature_columns
        # Since this cell contains add_rolling_features, build_features, feature_columns
        new_source = """def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("region_id", sort=False)
    
    # 1. Standard rolling means/max
    for col in ROLL_COLS:
        prior = grouped[col].shift(1)
        for window in ROLL_WINDOWS:
            roller = prior.groupby(df["region_id"], sort=False)
            df[f"{col}_roll{window}_mean"] = roller.transform(lambda s: s.rolling(window, min_periods=3).mean())
            df[f"{col}_roll{window}_max"] = roller.transform(lambda s: s.rolling(window, min_periods=3).max())
            if col == "prec":
                df[f"prec_roll{window}_sum"] = roller.transform(lambda s: s.rolling(window, min_periods=3).sum())
    
    # 2. V4 & New Meteorological Features (Shifted to prevent leak)
    prior_prec = grouped["prec"].shift(1)
    prior_tmp = grouped["tmp"].shift(1)
    prior_hum = grouped["humidity"].shift(1)
    
    roller_prec = prior_prec.groupby(df["region_id"], sort=False)
    roller_tmp = prior_tmp.groupby(df["region_id"], sort=False)
    roller_hum = prior_hum.groupby(df["region_id"], sort=False)
    
    # Base anomalies
    df["prec_roll90_mean"] = roller_prec.transform(lambda s: s.rolling(90, min_periods=10).mean())
    df["prec_roll365_mean"] = roller_prec.transform(lambda s: s.rolling(365, min_periods=60).mean())
    df["tmp_roll90_mean"] = roller_tmp.transform(lambda s: s.rolling(90, min_periods=10).mean())
    df["tmp_roll365_mean"] = roller_tmp.transform(lambda s: s.rolling(365, min_periods=60).mean())
    
    # Drought Indices (V4)
    df["prec_deficit_90d"] = df["prec_roll90_mean"] - df["prec_roll365_mean"]
    df["tmp_anomaly_90d"] = df["tmp_roll90_mean"] - df["tmp_roll365_mean"]
    
    p7 = roller_prec.transform(lambda s: s.rolling(7, min_periods=3).mean())
    p30 = roller_prec.transform(lambda s: s.rolling(30, min_periods=10).mean())
    p30_std = roller_prec.transform(lambda s: s.rolling(30, min_periods=10).std().clip(lower=0.01))
    df["prec_trend_30d"] = (p7 - p30) / p30_std
    
    hum_90 = roller_hum.transform(lambda s: s.rolling(90, min_periods=30).mean())
    hum_365 = roller_hum.transform(lambda s: s.rolling(365, min_periods=60).mean())
    df["humidity_deficit_90d"] = hum_90 - hum_365
    
    df["heat_drought_idx"] = df["prec_deficit_90d"] * df["tmp_anomaly_90d"].clip(lower=0)
    
    # Dry days counts
    is_dry = (prior_prec < 1.0).astype(float)
    dry_roller = is_dry.groupby(df["region_id"], sort=False)
    df["dry_days_14d"] = dry_roller.transform(lambda s: s.rolling(14, min_periods=3).sum())
    df["dry_days_30d"] = dry_roller.transform(lambda s: s.rolling(30, min_periods=7).sum())
    
    # NEW Domain Features: VPD (Vapor Pressure Deficit)
    # SVP = 0.6108 * exp(17.27 * T / (T + 237.3))
    # AVP = SVP * (RH / 100)
    # VPD = SVP - AVP
    svp = 0.6108 * np.exp((17.27 * prior_tmp) / (prior_tmp + 237.3))
    avp = svp * (prior_hum / 100.0)
    vpd = svp - avp
    vpd_roller = vpd.groupby(df["region_id"], sort=False)
    df["vpd_mean_14d"] = vpd_roller.transform(lambda s: s.rolling(14, min_periods=3).mean())
    df["vpd_mean_30d"] = vpd_roller.transform(lambda s: s.rolling(30, min_periods=7).mean())
    
    # NEW Domain Features: DTR (Diurnal Temp Range)
    prior_dtr = grouped["tmp_range"].shift(1)
    dtr_roller = prior_dtr.groupby(df["region_id"], sort=False)
    df["dtr_mean_14d"] = dtr_roller.transform(lambda s: s.rolling(14, min_periods=3).mean())
    df["dtr_mean_30d"] = dtr_roller.transform(lambda s: s.rolling(30, min_periods=7).mean())
    
    # NEW Domain Features: EMA Prec
    df["prec_ema_14d"] = roller_prec.transform(lambda s: s.ewm(span=14, min_periods=3).mean())
    
    # NEW Domain Features: AET Proxy (Apparent Evapotranspiration)
    prior_wind = grouped["wind"].shift(1)
    aet = (prior_tmp * prior_wind) / (prior_hum + 1.0)
    aet_roller = aet.groupby(df["region_id"], sort=False)
    df["aet_proxy_30d"] = aet_roller.transform(lambda s: s.rolling(30, min_periods=7).mean())
    
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
                
    drought_indices = [
        "prec_deficit_90d", "tmp_anomaly_90d", "prec_trend_30d",
        "humidity_deficit_90d", "heat_drought_idx",
        "dry_days_14d", "dry_days_30d",
        "vpd_mean_14d", "vpd_mean_30d",
        "dtr_mean_14d", "dtr_mean_30d",
        "prec_ema_14d", "aet_proxy_30d"
    ]
    
    calendar = ["month_sin", "month_cos", "day_sin", "day_cos"]
    cols = list(WEATHER_COLS) + lag_names + roll_names + drought_indices + calendar
    if include_region:
        cols = ["region_id"] + cols
    return cols
"""
        
        # Split the replacement correctly
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

with open("notebooks/pipeline_combined.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Added V4 and new Domain features to pipeline_combined.ipynb")
