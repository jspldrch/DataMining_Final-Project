"""
Region-level aggregates from train labels (one pass over train.csv).
Used by preprocessing v2 — no label leakage from test.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def compute_region_score_stats(
    train_path: Path,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Per region: mean/median/std of score on labeled rows only (~1.76M rows).
    """
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(train_path, usecols=["region_id", "score"], chunksize=chunk_size):
        labeled = chunk[chunk["score"].notna()]
        if not labeled.empty:
            pieces.append(labeled)

    if not pieces:
        raise ValueError("Keine gelabelten Zeilen in train.csv gefunden.")

    labeled_all = pd.concat(pieces, ignore_index=True)
    stats = labeled_all.groupby("region_id", sort=False)["score"].agg(
        mean="mean",
        median="median",
        std="std",
        count="count",
    )
    stats = stats.rename(
        columns={
            "mean": "region_score_mean",
            "median": "region_score_median",
            "std": "region_score_std",
            "count": "region_label_count",
        }
    )
    stats["region_score_std"] = stats["region_score_std"].fillna(0.0)
    return stats.reset_index()
