import json

with open("notebooks/modeling.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    source = "".join(cell["source"])
    
    # 1. Update build_samples and holdout logic
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


def build_samples(weekly, mode="train"):
    \"\"\"
    Sliding-Window je Region.
    mode='train': Nutzt alle Fenster, bei denen das Target VOR dem Validierungs-Start liegt.
    mode='val': Nutzt NUR das allerletzte Fenster (simuliert den Test auf Kaggle).
    mode='all': Nutzt alle verfuegbaren Fenster (fuer finales Training).
    \"\"\"
    X_parts, y_parts, p_parts, r_parts = [], [], [], []
    for region, g in weekly.groupby("region_id", sort=True):
        g = g.sort_values("ordinal")
        scores = g["score"].to_numpy(np.float32)
        n = len(g)
        
        # Wir brauchen mindestens 2*N_WEEKS fuer Train + Val Split
        if n <= 2 * N_WEEKS:
            continue
            
        y = np.lib.stride_tricks.sliding_window_view(scores[1:], N_WEEKS)
        X = g.iloc[: n - N_WEEKS][FEATURES]
        persist = scores[: n - N_WEEKS]
        
        if mode == "val":
            # Das allerletzte Fenster fuer Validierung
            X, y, persist = X.iloc[[-1]], y[[-1]], persist[[-1]]
        elif mode == "train":
            # Alle Fenster, deren Targets sich NICHT mit den Val-Targets ueberschneiden.
            # Val-Features sind am Index -1. Train-Features duerfen maximal am Index -1 - N_WEEKS sein.
            max_idx = len(X) - N_WEEKS
            if max_idx <= 0:
                continue
            X, y, persist = X.iloc[:max_idx], y[:max_idx], persist[:max_idx]
        elif mode == "all":
            pass # Behalte alles
            
        X_parts.append(X)
        y_parts.append(y)
        p_parts.append(persist)
        r_parts.append(np.full(len(persist), region, dtype=object))

    X = pd.concat(X_parts, ignore_index=True)
    return X, np.vstack(y_parts), np.concatenate(p_parts), np.concatenate(r_parts)
"""
        # Split by lines and add back newlines
        lines = [line + "\n" for line in new_source.split("\n")]
        # Remove last empty newline if exists
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

    # 2. Update split execution cell
    elif "train_regions, val_regions = region_holdout" in source:
        new_source = """X_tr, y_tr, persist_tr, _ = build_samples(train_df, mode="train")
X_va, y_va, persist_va, val_sample_regions = build_samples(train_df, mode="val")

print(f"Train-Fenster: {len(X_tr):,}  ({len(train_df['region_id'].unique()):,} Regionen)")
print(f"Val-Fenster:   {len(X_va):,}  ({len(val_sample_regions):,} Regionen)")
print(f"y-Shapes: train {y_tr.shape} · val {y_va.shape}")
"""
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

    # 3. Update LightGBM params
    elif "LGB_PARAMS =" in source:
        new_source = """LGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.03,      # Etwas langsamer fuer mehr Stabilitaet
    num_leaves=11,           # Staerkere Regularisierung (flachere Baeume)
    max_depth=5,             # Harte Grenze fuer Baumtiefe
    min_child_samples=300,   # Mehr Samples pro Blatt erzwungen
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,    # Weniger Features pro Baum (gegen Overfitting)
    reg_alpha=3.0,           # Staerkere L1 Regularisierung
    reg_lambda=5.0,          # Staerkere L2 Regularisierung
    verbose=-1,
)

models = []
val_preds = np.zeros_like(y_va, dtype=np.float64)

for w in range(N_WEEKS):
    m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_STATE + w, n_jobs=-1)
    m.fit(
        X_tr, week_target(y_tr, w),
        eval_set=[(X_va, week_target(y_va, w))],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)],
    )
    models.append(m)
    val_preds[:, w] = clip_scores(m.predict(X_va))
    print(f"  Woche {w + 1}: best_iteration = {m.best_iteration_}")

blend_val = clip_scores((1 - BLEND_PERSIST) * val_preds + BLEND_PERSIST * persist_preds)
blend_label = f"Blend ({1 - BLEND_PERSIST:.0%} Modell + {BLEND_PERSIST:.0%} Persist)"
print()
print(f"{'LightGBM (5 Modelle)':32s} MAE={mae(y_va, val_preds):.4f}")
print(f"{blend_label:32s} MAE={mae(y_va, blend_val):.4f}")
"""
        lines = [line + "\n" for line in new_source.split("\n")]
        if lines[-1] == "\n":
            lines.pop()
        cell["source"] = lines

    # 4. Update Final Training
    elif "X_all, y_all, _, _ = build_samples(train_df)" in source:
        new_source = """# Neu-Training auf ALLEN Regionen, Baum-Anzahl je Woche aus der Validierung.
X_all, y_all, _, _ = build_samples(train_df, mode="all")

final_models = []
for w in range(N_WEEKS):
    n_est = models[w].best_iteration_ or LGB_PARAMS["n_estimators"]
    m = lgb.LGBMRegressor(
        **{**LGB_PARAMS, "n_estimators": n_est},
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


with open("notebooks/modeling.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
