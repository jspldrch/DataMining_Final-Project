import json
from pathlib import Path

# Load Notebooks
with open("notebooks/preprocessing.ipynb", "r") as f:
    prep_nb = json.load(f)

with open("notebooks/modeling.ipynb", "r") as f:
    mod_nb = json.load(f)

# 1. Fix Leakage in Preprocessing
for cell in prep_nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    
    # Remove region_stats from features and pipeline
    if "def build_features_v2" in source:
        source = source.replace("def build_features_v2(df: pd.DataFrame, region_stats: pd.DataFrame | None = None) -> pd.DataFrame:", "def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:")
        source = source.replace("panel = build_features_v2(panel, region_stats=region_stats)", "panel = build_features_v2(panel)")
        source = source.replace("def process_region_v2_core(\n    train_part: pd.DataFrame,\n    test_part: pd.DataFrame,\n    region_stats: pd.DataFrame,\n)", "def process_region_v2_core(\n    train_part: pd.DataFrame,\n    test_part: pd.DataFrame,\n)")
        source = source.replace("def _region_worker_v2(\n    args: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],\n)", "def _region_worker_v2(\n    args: tuple[pd.DataFrame, pd.DataFrame],\n)")
        source = source.replace("train_r, test_r, region_stats = args\n    return process_region_v2_core(train_r, test_r, region_stats)", "train_r, test_r = args\n    return process_region_v2_core(train_r, test_r)")
        
        # fix iter_region_tasks
        source = source.replace("def _iter_region_tasks(\n    train_path: Path,\n    test_by_region: dict,\n    region_stats: pd.DataFrame,", "def _iter_region_tasks(\n    train_path: Path,\n    test_by_region: dict,")
        source = source.replace("yield (train_r, test_r, region_stats)", "yield (train_r, test_r)")
        
        # fix preprocess_by_region_v2
        source = source.replace("def preprocess_by_region_v2(\n    train_path: Path,\n    test_path: Path,\n    out_train: Path,\n    out_test: Path,\n    chunk_size: int = 500_000,\n    n_workers: int | None = None,\n) -> dict:\n    n_workers = n_workers if n_workers is not None else default_workers()\n    \n    print(\"Berechne region_stats vorab...\")\n    train_scores = pd.read_csv(train_path, usecols=[\"region_id\", \"score\"])\n    labeled = train_scores[train_scores[\"score\"].notna()]\n    region_stats = (\n        labeled.groupby(\"region_id\", sort=False)[\"score\"]\n        .agg(mean=\"mean\", median=\"median\", std=\"std\")\n        .rename(\n            columns={\n                \"mean\": \"region_score_mean\",\n                \"median\": \"region_score_median\",\n                \"std\": \"region_score_std\",\n            }\n        )\n        .reset_index()\n    )\n    region_stats[\"region_score_std\"] = region_stats[\"region_score_std\"].fillna(0.0)\n    del train_scores, labeled\n    print(f\"  -> {len(region_stats)} Regionen berechnet.\")", "def preprocess_by_region_v2(\n    train_path: Path,\n    test_path: Path,\n    out_train: Path,\n    out_test: Path,\n    chunk_size: int = 500_000,\n    n_workers: int | None = None,\n) -> dict:\n    n_workers = n_workers if n_workers is not None else default_workers()")
        
        source = source.replace("tasks = _iter_region_tasks(train_path, test_by_region, region_stats, chunk_size)", "tasks = _iter_region_tasks(train_path, test_by_region, chunk_size)")
        
        lines = [line + "\n" for line in source.split("\n")]
        if lines[-1] == "\n": lines.pop()
        cell["source"] = lines

# 2. Fix META in Modeling
for cell in mod_nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if 'META = {"region_id"' in source:
        source = source.replace(
            'META = {"region_id", "date", "year", "month", "day", "ordinal", "score", "score_persist7"}',
            'META = {"region_id", "date", "year", "month", "day", "ordinal", "score"}'
        )
        lines = [line + "\n" for line in source.split("\n")]
        if lines[-1] == "\n": lines.pop()
        cell["source"] = lines

# 3. Combine Notebooks
combined_nb = {
    "cells": prep_nb["cells"] + mod_nb["cells"],
    "metadata": prep_nb["metadata"],
    "nbformat": prep_nb["nbformat"],
    "nbformat_minor": prep_nb["nbformat_minor"]
}

# Update titles to avoid confusion
combined_nb["cells"].insert(0, {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# Pipeline Combined (Preprocess + Modeling)\n\nEnd-to-End Pipeline without leakage, including `score_persist7` feature."]
})

with open("notebooks/pipeline_combined.ipynb", "w") as f:
    json.dump(combined_nb, f, indent=1)

print("Created notebooks/pipeline_combined.ipynb")
