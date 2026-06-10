import json

path = "arthur_scripts/arthur_v14_HYBRID.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        
        # 1. Update build_sequence_data definition
        for i, line in enumerate(source):
            if "def build_sequence_data(" in line:
                # Find the val_regions parameter and replace it
                for j in range(i, i+10):
                    if "val_regions: set = None" in source[j]:
                        source[j] = source[j].replace("val_regions: set = None", "val_weeks: int = 0")
                        break
            
            # 2. Update logic inside build_sequence_data
            if "if val_regions is not None and region in val_regions:" in line:
                source[i] = ""
                source[i+1] = "" # continue
            
            if "n_windows = n - seq_len - N_WEEKS + 1" in line:
                source[i] = line.replace("n_windows = n - seq_len", "train_len = n - val_weeks\n        if train_len < seq_len + N_WEEKS: continue\n        n_windows = train_len - seq_len")

            # 3. Update the call
            if "X_dl_train, y_dl_train = build_sequence_data" in line:
                source[i] = line.replace("val_regions=val_regions", "val_weeks=VAL_WEEKS")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Patch applied for Temporal Validation!")
