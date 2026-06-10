import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

print("Loading full train.csv...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
print("Loading full test.csv...")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Get just 3 regions to make it very fast
regions = train_df["region_id"].unique()[:3]
print(f"Selecting regions: {regions}")

train_small = train_df[train_df["region_id"].isin(regions)]
test_small = test_df[test_df["region_id"].isin(regions)]

train_small.to_csv(DATA_DIR / "train_small.csv", index=False)
test_small.to_csv(DATA_DIR / "test_small.csv", index=False)

print("Created train_small.csv and test_small.csv!")
