import json

# --- 1. UPDATE PREPROCESSING ---
with open("notebooks/preprocessing.ipynb", "r") as f:
    prep_nb = json.load(f)

for cell in prep_nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    
    if "def build_features_v2" in source:
        new_source = """def build_features_v2(df: pd.DataFrame, region_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    df = build_features(df)
    df = add_persistence_baseline(df, lag_days=7)
    if region_stats is not None:
        # Merge the region statistics into the dataframe
        df = pd.merge(df, region_stats, on="region_id", how="left")
    return df

def feature_columns_v2(include_region: bool = True) -> list[str]:
    # Add the new region stats to the official features list
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

# --- 2. UPDATE MODELING ---
with open("notebooks/modeling.ipynb", "r") as f:
    mod_nb = json.load(f)

for cell in mod_nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    
    if 'META =' in source and 'FEATURES =' in source:
        new_source = """# Meta-Spalten zaehlen nicht als numerisches Modell-Feature.
# 'region_id' wurde wieder in META aufgenommen (wir nutzen stattdessen die region_stats)
META = {"region_id", "date", "year", "month", "day", "ordinal", "score", "score_persist7"}
FEATURES = [c for c in train_df.columns if c not in META and c in test_df.columns]

train_df[FEATURES] = train_df[FEATURES].astype(np.float32)
test_df[FEATURES] = test_df[FEATURES].astype(np.float32)

print(f"Modus: {MODE}")
print(f"Train: {len(train_df):,} Zeilen · {train_df['region_id'].nunique():,} Regionen")
print(f"Test:  {len(test_df):,} Zeilen · {test_df['region_id'].nunique():,} Regionen")
print(f"Features ({len(FEATURES)}): {FEATURES[:6]} …")
train_df.head(3)
"""
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

with open("notebooks/modeling.ipynb", "w") as f:
    json.dump(mod_nb, f, indent=1)

print("Stats update applied successfully.")
