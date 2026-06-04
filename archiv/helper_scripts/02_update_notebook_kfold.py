import json

with open("notebooks/modeling.ipynb", "r") as f:
    nb = json.load(f)

# Combine cells for validation into one to make the loop easier
new_cells = []
skip = False

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        new_cells.append(cell)
        continue
        
    source = "".join(cell["source"])
    
    # Update build_samples
    if "def build_samples" in source:
        new_source = """def clip_scores(arr):
    \"\"\"Auf den Wettbewerbs-Score-Bereich begrenzen.\"\"\"
    return np.clip(arr, 0.0, 5.0)


def mae(y_true, y_pred):
    \"\"\"MAE ueber alle Regionen x 5 Wochen (Kaggle-Metrik).\"\"\"
    return float(np.mean(np.abs(clip_scores(y_pred) - np.asarray(y_true))))


def week_target(y, w):
    \"\"\"Labels fuer Woche w als 1D-float64-Array (LightGBM 4.x erwartet 1D).\"\"\"
    return np.ascontiguousarray(y[:, w], dtype=np.float64)


def build_samples(weekly, mode="train", val_offset_weeks=0):
    \"\"\"
    Sliding-Window je Region.
    val_offset_weeks: 0 = letztes Fenster, 5 = Fenster 5 Wochen davor, etc.
    mode='train': Alle Fenster VOR dem Validierungsfenster.
    mode='val': Genau das Validierungsfenster.
    mode='all': Alle verfuegbaren Fenster.
    \"\"\"
    X_parts, y_parts, p_parts, r_parts = [], [], [], []
    for region, g in weekly.groupby("region_id", sort=True):
        g = g.sort_values("ordinal")
        scores = g["score"].to_numpy(np.float32)
        n = len(g)
        
        if n <= 2 * N_WEEKS + val_offset_weeks:
            continue
            
        y = np.lib.stride_tricks.sliding_window_view(scores[1:], N_WEEKS)
        X = g.iloc[: n - N_WEEKS][FEATURES]
        persist = scores[: n - N_WEEKS]
        
        target_val_idx = -1 - val_offset_weeks
        
        if mode == "val":
            # Das Validierungsfenster
            X, y, persist = X.iloc[[target_val_idx]], y[[target_val_idx]], persist[[target_val_idx]]
        elif mode == "train":
            # Alle Fenster davor (ohne Target-Ueberschneidung)
            # Das Fenster bei target_val_idx ueberschneidet sich die naechsten N_WEEKS.
            # Train darf hoechstens bis target_val_idx - N_WEEKS gehen.
            max_idx = len(X) + target_val_idx - N_WEEKS + 1
            if max_idx <= 0:
                continue
            X, y, persist = X.iloc[:max_idx], y[:max_idx], persist[:max_idx]
        elif mode == "all":
            pass # Behalte alles
            
        X_parts.append(X)
        y_parts.append(y)
        p_parts.append(persist)
        r_parts.append(np.full(len(persist), region, dtype=object))

    if not X_parts:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([])
        
    X = pd.concat(X_parts, ignore_index=True)
    return X, np.vstack(y_parts), np.concatenate(p_parts), np.concatenate(r_parts)
"""
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines
        new_cells.append(cell)

    elif 'X_tr, y_tr, persist_tr, _ = build_samples(train_df, mode="train")' in source:
        # We will replace this and the following LightGBM cell with a single K-Fold loop cell.
        skip = True
        
    elif 'LGB_PARAMS =' in source and 'for w in range(N_WEEKS):' in source:
        # This is the LightGBM cell. Replace both with the new combined cell.
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
fold_persist_maes = []
best_iters_per_week = {w: [] for w in range(N_WEEKS)}

for fold, offset in enumerate(FOLDS):
    print(f"\\n{'='*40}\\nFold {fold+1} (Offset {offset} Wochen)\\n{'='*40}")
    
    X_tr, y_tr, persist_tr, _ = build_samples(train_df, mode="train", val_offset_weeks=offset)
    X_va, y_va, persist_va, val_sample_regions = build_samples(train_df, mode="val", val_offset_weeks=offset)
    
    if len(X_va) == 0:
        print("Nicht genug Daten für diesen Fold!")
        continue
        
    print(f"Train-Fenster: {len(X_tr):,} | Val-Fenster: {len(X_va):,}")
    
    # Persistence
    persist_preds = np.repeat(persist_va[:, None], N_WEEKS, axis=1)
    persist_mae = mae(y_va, persist_preds)
    fold_persist_maes.append(persist_mae)
    
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
    blend_val = clip_scores((1 - BLEND_PERSIST) * val_preds + BLEND_PERSIST * persist_preds)
    blend_mae = mae(y_va, blend_val)
    fold_maes.append(blend_mae)
    
    print(f"Persist MAE: {persist_mae:.4f}")
    print(f"LightGBM MAE: {lgb_mae:.4f}")
    print(f"Blend MAE:   {blend_mae:.4f}")

print(f"\\n{'='*40}\\nDURCHSCHNITT ÜBER ALLE FOLDS\\n{'='*40}")
print(f"Persist MAE: {np.mean(fold_persist_maes):.4f}")
print(f"Blend MAE:   {np.mean(fold_maes):.4f}")
"""
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines
        new_cells.append(cell)
        skip = False
        
    elif 'X_all, y_all, _, _ = build_samples(train_df, mode="all")' in source:
        new_source = """# Neu-Training auf ALLEN Regionen
X_all, y_all, _, _ = build_samples(train_df, mode="all")

final_models = []
for w in range(N_WEEKS):
    # Nimm den durchschnittlichen best_iteration der Folds fuer diese Woche
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

# Persistence fuer Test = letzter gelabelter Train-Score je Region.
last_score = train_df.sort_values("ordinal").groupby("region_id")["score"].last()
persist_test = last_score.reindex(test_regions).fillna(0.0).to_numpy()
test_preds = clip_scores(
    (1 - BLEND_PERSIST) * test_preds + BLEND_PERSIST * persist_test[:, None]
)
print(f"Submission-Blend: {1 - BLEND_PERSIST:.0%} Modell + {BLEND_PERSIST:.0%} Persistence")

# Submission schreiben.
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
        new_cells.append(cell)
        
    elif not skip:
        new_cells.append(cell)

nb["cells"] = new_cells

with open("notebooks/modeling.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("K-Fold update applied.")
