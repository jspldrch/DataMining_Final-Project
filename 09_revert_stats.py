import json
import re

# 1. Update preprocessing.ipynb
with open("notebooks/preprocessing.ipynb", "r") as f:
    prep_nb = json.load(f)

for cell in prep_nb["cells"]:
    if cell["cell_type"] != "code": continue
    source = "".join(cell["source"])
    
    if "def build_features_v2" in source and "def feature_columns_v2" in source:
        # Revert build_features_v2 and feature_columns_v2 to NOT use region_stats
        new_source = re.sub(
            r'def build_features_v2.*?return df',
            """def build_features_v2(df: pd.DataFrame, region_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    df = build_features(df)
    df = add_persistence_baseline(df, lag_days=7)
    return df""",
            source, flags=re.DOTALL
        )
        new_source = re.sub(
            r'def feature_columns_v2.*?return list\(dict\.fromkeys\(base \+ extra\)\)',
            """def feature_columns_v2(include_region: bool = True) -> list[str]:
    score_cols = ["score_persist7"]
    base = feature_columns(include_region=include_region)
    extra = score_cols
    return list(dict.fromkeys(base + extra))""",
            new_source, flags=re.DOTALL
        )
        
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n": lines.pop()
        cell["source"] = lines

with open("notebooks/preprocessing.ipynb", "w") as f:
    json.dump(prep_nb, f, indent=1)


# 2. Update modeling.ipynb
with open("notebooks/modeling.ipynb", "r") as f:
    mod_nb = json.load(f)

for cell in mod_nb["cells"]:
    if cell["cell_type"] != "code": continue
    source = "".join(cell["source"])
    
    if "LGB_PARAMS = dict(" in source:
        # Add objective="mae" to LGB_PARAMS
        if 'objective="mae"' not in source and "objective='mae'" not in source:
            new_source = source.replace("LGB_PARAMS = dict(", "LGB_PARAMS = dict(\n    objective=\"mae\",")
            lines = [line + "\n" for line in new_source.split("\n")]
            if lines[-1] == "\n": lines.pop()
            cell["source"] = lines

with open("notebooks/modeling.ipynb", "w") as f:
    json.dump(mod_nb, f, indent=1)

print("Reverted region stats and added objective='mae' to modeling.")
