import json

with open("notebooks/modeling.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    source = "".join(cell["source"])
    
    # Update META and FEATURES logic
    if 'META = {"region_id"' in source:
        new_source = """# Meta-Spalten zaehlen nicht als numerisches Modell-Feature.
# 'region_id' wurde aus META entfernt, damit es als Feature nutzbar ist!
META = {"date", "year", "month", "day", "ordinal", "score", "score_persist7"}
FEATURES = [c for c in train_df.columns if c not in META and c in test_df.columns]

if "region_id" in FEATURES:
    # region_id muss vom Typ 'category' sein, damit LightGBM es versteht
    train_df["region_id"] = train_df["region_id"].astype("category")
    test_df["region_id"] = test_df["region_id"].astype("category")
    
    num_feats = [c for c in FEATURES if c != "region_id"]
    train_df[num_feats] = train_df[num_feats].astype(np.float32)
    test_df[num_feats] = test_df[num_feats].astype(np.float32)
else:
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
    json.dump(nb, f, indent=1)

print("Region update applied.")
