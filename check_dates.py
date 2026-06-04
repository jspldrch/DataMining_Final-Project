import pandas as pd

train = pd.read_csv("data/train.csv", usecols=["region_id", "date"])
train_r1 = train[train["region_id"] == "R1"]
print("Train R1 max date:", train_r1["date"].max())
print("Train R1 min date:", train_r1["date"].min())
print("Train R1 len:", len(train_r1))

test = pd.read_csv("data/test.csv", usecols=["region_id", "date"])
test_r1 = test[test["region_id"] == "R1"]
print("Test R1 max date:", test_r1["date"].max())
print("Test R1 min date:", test_r1["date"].min())
print("Test R1 len:", len(test_r1))
