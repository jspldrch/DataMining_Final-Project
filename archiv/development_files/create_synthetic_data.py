import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Generate synthetic train data
np.random.seed(42)
regions = [100, 200]
dates = pd.date_range("2020-01-01", periods=100, freq="D")

train_rows = []
for r in regions:
    for i, d in enumerate(dates):
        train_rows.append({
            "region_id": r,
            "ordinal": i,
            "date": d.strftime("%Y-%m-%d"),
            "prec": np.random.rand(),
            "surf_pre": np.random.rand() * 1000,
            "tmp": np.random.rand() * 30,
            "tmp_max": np.random.rand() * 35,
            "tmp_min": np.random.rand() * 20,
            "tmp_range": np.random.rand() * 15,
            "dp_tmp": np.random.rand() * 20,
            "wb_tmp": np.random.rand() * 20,
            "surf_tmp": np.random.rand() * 30,
            "wind": np.random.rand() * 10,
            "wind_max": np.random.rand() * 15,
            "wind_min": np.random.rand() * 5,
            "wind_range": np.random.rand() * 10,
            "humidity": np.random.rand() * 100,
            "score": np.random.rand() * 5.0
        })

train_df = pd.DataFrame(train_rows)
train_df.to_csv(DATA_DIR / "train.csv", index=False)

# Generate synthetic test data
test_rows = []
test_dates = pd.date_range("2020-04-10", periods=14, freq="D")
for r in regions:
    for i, d in enumerate(test_dates, start=100):
        test_rows.append({
            "region_id": r,
            "ordinal": i,
            "date": d.strftime("%Y-%m-%d"),
            "prec": np.random.rand(),
            "surf_pre": np.random.rand() * 1000,
            "tmp": np.random.rand() * 30,
            "tmp_max": np.random.rand() * 35,
            "tmp_min": np.random.rand() * 20,
            "tmp_range": np.random.rand() * 15,
            "dp_tmp": np.random.rand() * 20,
            "wb_tmp": np.random.rand() * 20,
            "surf_tmp": np.random.rand() * 30,
            "wind": np.random.rand() * 10,
            "wind_max": np.random.rand() * 15,
            "wind_min": np.random.rand() * 5,
            "wind_range": np.random.rand() * 10,
            "humidity": np.random.rand() * 100,
            "score": np.nan
        })

test_df = pd.DataFrame(test_rows)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print("Created synthetic train.csv and test.csv in data/ directory!")
