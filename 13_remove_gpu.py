import json
import re

with open("notebooks/pipeline_combined.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    source = "".join(cell["source"])
    
    if 'LGB_PARAMS = dict(' in source and 'device="gpu"' in source:
        # Remove device="gpu",
        source = re.sub(r'\s*device="gpu",\n', '\n', source)
        
        lines = [line + "\n" for line in source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

with open("notebooks/pipeline_combined.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Removed device='gpu' from pipeline_combined.ipynb")
