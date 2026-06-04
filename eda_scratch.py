import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", usecols=["region_id", "date", "score"])
test = pd.read_csv("data/test.csv", usecols=["region_id", "date"])

print("--- Train ---")
train_grouped = train.groupby("region_id")["date"].agg(["min", "max"])
print("Unique train max dates:", train_grouped["max"].unique())
print("Unique train min dates:", train_grouped["min"].unique())

print("--- Test ---")
test_grouped = test.groupby("region_id")["date"].agg(["min", "max"])
print("Unique test max dates:", test_grouped["max"].unique())
print("Unique test min dates:", test_grouped["min"].unique())

print("--- Gap ---")
train_max = pd.to_datetime(train_grouped["max"])
test_min = pd.to_datetime(test_grouped["min"])
gaps = (test_min - train_max).dt.days
print("Unique gaps (in days):", gaps.unique())

print("--- Scores ---")
scores_notna = train.dropna(subset=["score"]).groupby("region_id")["date"].agg(["max"])
print("Unique last score dates:", scores_notna["max"].unique())

