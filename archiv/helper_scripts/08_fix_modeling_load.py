import json

with open("notebooks/modeling.ipynb", "r") as f:
    mod_nb = json.load(f)

for cell in mod_nb["cells"]:
    if cell["cell_type"] != "code": continue
    source = "".join(cell["source"])
    if 'META = {"region_id", "date"' in source or "train_df[FEATURES]" in source:
        new_source = """# --- Daten laden ---
MODE = "sample" if "sample" in TRAIN_PARQUET.name else "full"

print("Lade Parquet-Dateien...")
train_df = pd.read_parquet(TRAIN_PARQUET)
test_df = pd.read_parquet(TEST_PARQUET)

# Meta-Spalten zaehlen nicht als numerisches Modell-Feature.
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

print("Fixed train_df loading in modeling.ipynb")
