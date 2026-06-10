import json

path = "arthur_scripts/arthur_v14_HYBRID.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "X_dl_train, y_dl_train = build_sequence_data(train_weekly, DL_FEATURES, SEQ_LEN, stride=1)" in line:
                source[i] = line.replace("stride=1", "val_regions=val_regions, stride=1")
                print("Patched build_sequence_data call!")
            
            # Change loss to SmoothL1Loss to prevent zero-collapse
            if "loss = torch.mean(torch.abs(pred.clamp(0.0, 5.0) - yb))" in line:
                source[i] = "            loss_fn = torch.nn.SmoothL1Loss()\n            loss = loss_fn(pred.clamp(0.0, 5.0), yb)\n"
                print("Patched PyTorch Loss function!")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Done patching ipynb.")
