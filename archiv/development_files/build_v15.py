import json
import re

path_in = "arthur_scripts/arthur_v12_SPATIAL.ipynb"
path_out = "arthur_scripts/arthur_v15_ULTIMATE.ipynb"

with open(path_in, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "markdown":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            source[i] = line.replace("Arthur v12 - Spatial Magic", "Arthur v15 - ULTIMATE (Spatial + Slow Learning)")
            source[i] = source[i].replace("v12", "v15")
            
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            # Paths
            source[i] = line.replace("v12_SPATIAL", "v15_ULTIMATE")
            
            # N_ESTIMATORS
            if "N_ESTIMATORS  = 1000" in line or "N_ESTIMATORS = 1000" in line:
                source[i] = "N_ESTIMATORS  = 4000\n"
                
            # Learning Rate
            if "learning_rate=0.04" in line:
                source[i] = line.replace("learning_rate=0.04", "learning_rate=0.01")
                
with open(path_out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Created arthur_v15_ULTIMATE.ipynb successfully!")
