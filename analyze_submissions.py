import pandas as pd
import numpy as np

# Load the submissions
v2_5 = pd.read_csv("submission_full_v2-5.csv")
v4 = pd.read_csv("submission_v4.csv")

# Ensure they are sorted by region_id
v2_5 = v2_5.sort_values("region_id").reset_index(drop=True)
v4 = v4.sort_values("region_id").reset_index(drop=True)

# Select only the prediction columns
pred_cols = [f"pred_week{i}" for i in range(1, 6)]
v2_5_preds = v2_5[pred_cols].values
v4_preds = v4[pred_cols].values

print("=== Basic Statistics ===")
print(f"V2-5 (Score 0.87) - Mean: {v2_5_preds.mean():.4f}, Std: {v2_5_preds.std():.4f}, Min: {v2_5_preds.min():.4f}, Max: {v2_5_preds.max():.4f}")
print(f"V4 (Score 0.85)   - Mean: {v4_preds.mean():.4f}, Std: {v4_preds.std():.4f}, Min: {v4_preds.min():.4f}, Max: {v4_preds.max():.4f}")
print()

# Check variance across weeks (how much does the model predict changes over time vs static prediction)
v2_5_week_var = np.var(v2_5_preds, axis=1).mean()
v4_week_var = np.var(v4_preds, axis=1).mean()

print("=== Mean Variance across the 5 weeks (Per Region) ===")
print(f"V2-5: {v2_5_week_var:.6f} (Higher means model predicts changes from week 1 to 5)")
print(f"V4  : {v4_week_var:.6f}")
print()

# Check Mean Absolute Difference between the two submissions
mae_diff = np.abs(v2_5_preds - v4_preds).mean()
print(f"=== Mean Absolute Difference between V2-5 and V4 ===")
print(f"MAD: {mae_diff:.4f}")
print()

# Check correlation between the submissions
corrs = []
for i in range(5):
    c = np.corrcoef(v2_5_preds[:, i], v4_preds[:, i])[0, 1]
    corrs.append(c)
print(f"=== Correlation between V2-5 and V4 per week ===")
for i, c in enumerate(corrs):
    print(f"Week {i+1}: {c:.4f}")
print(f"Average Correlation: {np.mean(corrs):.4f}")

