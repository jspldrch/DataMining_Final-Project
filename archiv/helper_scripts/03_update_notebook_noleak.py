import json

with open("notebooks/modeling.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    source = "".join(cell["source"])
    
    # 1. Update META columns to exclude score_persist7
    if 'META = {"region_id"' in source:
        new_source = source.replace(
            'META = {"region_id", "date", "year", "month", "day", "ordinal", "score"}',
            'META = {"region_id", "date", "year", "month", "day", "ordinal", "score", "score_persist7"}'
        )
        # Also remove BLEND_PERSIST definition
        new_source = "\n".join([line for line in new_source.split("\n") if "BLEND_PERSIST =" not in line])
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

    # 2. Update Validation Loop
    elif 'FOLDS = [0, 5, 10]' in source:
        new_source = """LGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=11,
    max_depth=5,
    min_child_samples=300,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_alpha=3.0,
    reg_lambda=5.0,
    verbose=-1,
)

FOLDS = [0, 5, 10]
fold_maes = []
best_iters_per_week = {w: [] for w in range(N_WEEKS)}

for fold, offset in enumerate(FOLDS):
    print(f"\\n{'='*40}\\nFold {fold+1} (Offset {offset} Wochen)\\n{'='*40}")
    
    X_tr, y_tr, _, _ = build_samples(train_df, mode="train", val_offset_weeks=offset)
    X_va, y_va, _, _ = build_samples(train_df, mode="val", val_offset_weeks=offset)
    
    if len(X_va) == 0:
        print("Nicht genug Daten für diesen Fold!")
        continue
        
    print(f"Train-Fenster: {len(X_tr):,} | Val-Fenster: {len(X_va):,}")
    
    val_preds = np.zeros_like(y_va, dtype=np.float64)
    for w in range(N_WEEKS):
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_STATE + w + fold*100, n_jobs=-1)
        m.fit(
            X_tr, week_target(y_tr, w),
            eval_set=[(X_va, week_target(y_va, w))],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)],
        )
        val_preds[:, w] = clip_scores(m.predict(X_va))
        best_iters_per_week[w].append(m.best_iteration_)
        
    lgb_mae = mae(y_va, val_preds)
    fold_maes.append(lgb_mae)
    
    print(f"LightGBM MAE: {lgb_mae:.4f}")

print(f"\\n{'='*40}\\nDURCHSCHNITT ÜBER ALLE FOLDS\\n{'='*40}")
print(f"LightGBM MAE (nur Wetterdaten): {np.mean(fold_maes):.4f}")
"""
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

    # 3. Update Final Training
    elif '# Neu-Training auf ALLEN Regionen' in source:
        new_source = """# Neu-Training auf ALLEN Regionen
X_all, y_all, _, _ = build_samples(train_df, mode="all")

final_models = []
for w in range(N_WEEKS):
    avg_best_iter = int(np.mean(best_iters_per_week[w])) if best_iters_per_week[w] else LGB_PARAMS["n_estimators"]
    
    m = lgb.LGBMRegressor(
        **{**LGB_PARAMS, "n_estimators": avg_best_iter},
        random_state=RANDOM_STATE + w,
        n_jobs=-1,
    )
    m.fit(X_all, week_target(y_all, w))
    final_models.append(m)
print(f"Finales Training: {N_WEEKS} Modelle auf {len(X_all):,} Fenstern")

# Eine Feature-Zeile je Region: letzter Tag des 91-Tage-Testfensters.
last_rows = test_df.sort_values(["region_id", "ordinal"]).groupby("region_id").tail(1)
test_regions = last_rows["region_id"].to_numpy()
X_test = last_rows[FEATURES]

test_preds = np.column_stack([clip_scores(m.predict(X_test)) for m in final_models])

# Keine Persistence mehr, reines Modell!
print("Submission: 100% LightGBM (Wetterdaten-Modell)")

submission = pd.DataFrame({"region_id": test_regions})
for k in range(N_WEEKS):
    submission[f"pred_week{k + 1}"] = test_preds[:, k]
submission = submission.sort_values("region_id").reset_index(drop=True)

out_path = SUBMISSION_DIR / f"submission_{MODE}_v2.csv"
submission.to_csv(out_path, index=False)
print(f"Gespeichert: {out_path}  ({len(submission):,} Zeilen)")
submission.head(10)
"""
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

with open("notebooks/modeling.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Leakage removal applied.")
