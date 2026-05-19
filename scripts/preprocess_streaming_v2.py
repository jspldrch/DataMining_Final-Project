"""
Preprocessing v2 — same streaming layout as v1, enhanced features.

Outputs (default names):
  train_labeled_v2.parquet
  test_features_v2.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.features import parse_dates
from scripts.features_v2 import (
    add_test91_summary,
    build_features_v2,
    feature_columns_v2,
    save_columns_v2,
)
from scripts.preprocess_streaming import _ParquetAppender
from scripts.region_stats import compute_region_score_stats

OUT_TRAIN_V2 = "train_labeled_v2.parquet"
OUT_TEST_V2 = "test_features_v2.parquet"


def _process_region_v2(
    train_part: pd.DataFrame,
    test_part: pd.DataFrame,
    region_stats: pd.DataFrame,
    train_writer: _ParquetAppender,
    test_writer: _ParquetAppender,
) -> None:
    train_part = train_part.copy()
    test_part = test_part.copy()
    train_part["_split"] = "train"
    test_part["_split"] = "test"
    if "score" not in test_part.columns:
        test_part["score"] = float("nan")

    panel = pd.concat([train_part, test_part], ignore_index=True)
    panel = build_features_v2(panel, region_stats=region_stats)

    train_feat = panel[panel["_split"] == "train"]
    test_feat = panel[panel["_split"] == "test"]
    test_raw = parse_dates(test_part.drop(columns=["_split"], errors="ignore"))
    test_feat = add_test91_summary(test_feat, test_raw)

    train_labeled = train_feat[train_feat["score"].notna()]
    train_writer.write(train_labeled[save_columns_v2(train_labeled, labeled=True)])
    test_writer.write(test_feat[save_columns_v2(test_feat, labeled=False)])


def preprocess_by_region_v2(
    train_path: Path,
    test_path: Path,
    out_train: Path,
    out_test: Path,
    chunk_size: int = 500_000,
) -> dict:
    print("v2: Region-Score-Statistiken aus train.csv …")
    region_stats = compute_region_score_stats(train_path, chunk_size=chunk_size)
    print(f"  → {len(region_stats):,} Regionen")

    test = parse_dates(pd.read_csv(test_path))
    test_by_region = {r: g for r, g in test.groupby("region_id", sort=False)}

    for path in (out_train, out_test):
        if path.exists():
            path.unlink()

    train_writer = _ParquetAppender(out_train)
    test_writer = _ParquetAppender(out_test)

    buffer_parts: list[pd.DataFrame] = []
    current_region = None
    n_regions = 0

    try:
        for chunk in pd.read_csv(train_path, chunksize=chunk_size):
            chunk = parse_dates(chunk)
            for region, g in chunk.groupby("region_id", sort=False):
                if current_region is None:
                    current_region = region
                    buffer_parts = [g]
                    continue

                if region == current_region:
                    buffer_parts.append(g)
                    continue

                train_r = pd.concat(buffer_parts, ignore_index=True)
                test_r = test_by_region.get(current_region, pd.DataFrame())
                _process_region_v2(
                    train_r, test_r, region_stats, train_writer, test_writer
                )
                n_regions += 1
                if n_regions % 200 == 0:
                    print(f"  … {n_regions} Regionen verarbeitet")

                current_region = region
                buffer_parts = [g]

        if current_region is not None and buffer_parts:
            train_r = pd.concat(buffer_parts, ignore_index=True)
            test_r = test_by_region.get(current_region, pd.DataFrame())
            _process_region_v2(
                train_r, test_r, region_stats, train_writer, test_writer
            )
            n_regions += 1

    finally:
        train_writer.close()
        test_writer.close()

    train_rows = pq.read_metadata(out_train).num_rows if out_train.exists() else 0
    test_rows = pq.read_metadata(out_test).num_rows if out_test.exists() else 0

    return {
        "version": 2,
        "regions": n_regions,
        "train_labeled_rows": train_rows,
        "test_rows": test_rows,
        "out_train": out_train,
        "out_test": out_test,
        "feature_count": len(feature_columns_v2()),
    }


def preprocess_panel_v2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    region_stats: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample-mode path (fits in RAM)."""
    if region_stats is None:
        labeled = train[train["score"].notna()][["region_id", "score"]]
        region_stats = labeled.groupby("region_id", sort=False)["score"].agg(
            mean="mean", median="median", std="std"
        ).rename(
            columns={
                "mean": "region_score_mean",
                "median": "region_score_median",
                "std": "region_score_std",
            }
        ).reset_index()
        region_stats["region_score_std"] = region_stats["region_score_std"].fillna(0.0)

    train = train.copy()
    test = test.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    test["score"] = float("nan")

    panel = pd.concat([train, test], ignore_index=True)
    panel = build_features_v2(panel, region_stats=region_stats)

    train_feat = panel[panel["_split"] == "train"]
    test_feat = panel[panel["_split"] == "test"]
    test_raw = parse_dates(test.drop(columns=["_split"], errors="ignore"))
    test_feat = add_test91_summary(test_feat, test_raw)

    train_labeled = train_feat[train_feat["score"].notna()]
    return (
        train_labeled[save_columns_v2(train_labeled, labeled=True)],
        test_feat[save_columns_v2(test_feat, labeled=False)],
    )
