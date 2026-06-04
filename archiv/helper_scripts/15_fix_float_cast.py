import json

with open("notebooks/pipeline_combined.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    source = "".join(cell["source"])
    
    if 'train_df[FEATURES] = train_df[FEATURES].astype(np.float32)' in source:
        # We need to cast only numerical features to float32, skipping region_id
        replacement = """
# Cast only numeric features to float32 to save memory
num_features = [c for c in FEATURES if c != "region_id"]
train_df[num_features] = train_df[num_features].astype(np.float32)
test_df[num_features] = test_df[num_features].astype(np.float32)
"""
        source = source.replace('train_df[FEATURES] = train_df[FEATURES].astype(np.float32)\ntest_df[FEATURES] = test_df[FEATURES].astype(np.float32)', replacement.strip())
        
        lines = [line + "\n" for line in source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

with open("notebooks/pipeline_combined.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Fixed float cast error in pipeline_combined.ipynb")
