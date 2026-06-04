import json
import re

with open("notebooks/pipeline_combined.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    source = "".join(cell["source"])
    
    # Replace META definition
    if 'META = {"region_id"' in source:
        source = re.sub(r'META = \{"region_id"[^\}]+\}', 'META = {"date", "year", "month", "day", "ordinal", "score", "score_persist7"}', source)
    
    # Cast region_id to category
    if 'test_df = pd.read_parquet(TEST_PARQUET)' in source:
        replacement = 'test_df = pd.read_parquet(TEST_PARQUET)\n\n# Convert region_id to category for LightGBM\ntrain_df["region_id"] = train_df["region_id"].astype("category")\ntest_df["region_id"] = test_df["region_id"].astype("category")'
        source = source.replace('test_df = pd.read_parquet(TEST_PARQUET)', replacement)
        
    lines = [line + "\n" for line in source.split("\n")]
    if lines[-1] == "\n":
        lines.pop()
    cell["source"] = lines

with open("notebooks/pipeline_combined.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Applied V5 fixes to pipeline_combined.ipynb")
