import json
import re

with open("notebooks/pipeline_combined.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    source = "".join(cell["source"])
    
    if 'LGB_PARAMS = dict(' in source and 'objective="mae"' in source:
        # Update learning rate
        source = re.sub(r'learning_rate=0\.03,', 'learning_rate=0.1,', source)
        
        # Add device="gpu"
        if 'device="gpu"' not in source:
            source = re.sub(r'verbose=-1,\n\)', 'verbose=-1,\n    device="gpu",\n)', source)
            
        # Update FOLDS
        source = re.sub(r'FOLDS = \[0, 5, 10\]', 'FOLDS = [0]', source)
        
        lines = [line + "\n" for line in source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

with open("notebooks/pipeline_combined.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Updated parameters in pipeline_combined.ipynb")
