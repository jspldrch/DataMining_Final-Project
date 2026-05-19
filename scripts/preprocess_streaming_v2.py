"""
Preprocessing v2 — streaming + optional multi-core per region.

Outputs:
  train_labeled_v2.parquet
  test_features_v2.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.features import parse_dates
from scripts.features_v2 import (
    add_test91_summary,
    build_features_v2,
    feature_columns_v2,
    save_columns_v2,
)
from scripts.parallel_util import default_workers, run_parallel_consume
from scripts.preprocess_streaming import _ParquetAppender
from scripts.region_stats import compute_region_score_stats

OUT_TRAIN_V2 = "train_labeled_v2.parquet"
OUT_TEST_V2 = "test_features_v2.parquet"


def process_region_v2_core(
    train_part: pd.DataFrame,
    test_part: pd.DataFrame,
    region_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature engineering for one region (no I/O)."""
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
    return (
        train_labeled[save_columns_v2(train_labeled, labeled=True)],
        test_feat[save_columns_v2(test_feat, labeled=False)],
    )


def _region_worker_v2(
    args: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_r, test_r, region_stats = args
    return process_region_v2_core(train_r, test_r, region_stats)


def _process_region_v2(
    train_part: pd.DataFrame,
    test_part: pd.DataFrame,
    region_stats: pd.DataFrame,
    train_writer: _ParquetAppender,
    test_writer: _ParquetAppender,
) -> None:
    train_out, test_out = process_region_v2_core(train_part, test_part, region_stats)
    train_writer.write(train_out)
    test_writer.write(test_out)


def _iter_region_tasks(
    train_path: Path,
    test_by_region: dict,
    region_stats: pd.DataFrame,
    chunk_size: int,
):
    """Yield (train_r, test_r, region_stats) per completed region."""
    buffer_parts: list[pd.DataFrame] = []
    current_region = None

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
            yield (train_r, test_r, region_stats)
            current_region = region
            buffer_parts = [g]

    if current_region is not None and buffer_parts:
        train_r = pd.concat(buffer_parts, ignore_index=True)
        test_r = test_by_region.get(current_region, pd.DataFrame())
        yield (train_r, test_r, region_stats)


def preprocess_by_region_v2(
    train_path: Path,
    test_path: Path,
    out_train: Path,
    out_test: Path,
    chunk_size: int = 500_000,
    n_workers: int | None = None,
) -> dict:
    n_workers = n_workers if n_workers is not None else default_workers()
    print(f"v2: Region-Score-Statistiken aus train.csv … (workers={n_workers})")
    region_stats = compute_region_score_stats(train_path, chunk_size=chunk_size)
    print(f"  → {len(region_stats):,} Regionen")

    test = parse_dates(pd.read_csv(test_path))
    test_by_region = {r: g for r, g in test.groupby("region_id", sort=False)}

    for path in (out_train, out_test):
        if path.exists():
            path.unlink()

    train_writer = _ParquetAppender(out_train)
    test_writer = _ParquetAppender(out_test)
    finished_regions: set[str] = set()
    duplicate_test_skipped = 0

    def _consume(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        nonlocal duplicate_test_skipped
        train_out, test_out = result
        if train_out.empty and test_out.empty:
            return

        rid = str(
            test_out["region_id"].iloc[0]
            if not test_out.empty
            else train_out["region_id"].iloc[0]
        )

        # train.csv can have the same region_id in two non-contiguous blocks (unsorted
        # file). We still append train labels from each block, but test must be written
        # only once per region (91 rows) — otherwise Kaggle features are duplicated.
        if rid in finished_regions:
            duplicate_test_skipped += 1
            if not train_out.empty:
                train_writer.write(train_out)
            return

        finished_regions.add(rid)
        if not train_out.empty:
            train_writer.write(train_out)
        if not test_out.empty:
            test_writer.write(test_out)

        if len(finished_regions) % 200 == 0:
            print(f"  … {len(finished_regions)} Regionen verarbeitet")

    try:
        tasks = _iter_region_tasks(train_path, test_by_region, region_stats, chunk_size)
        run_parallel_consume(
            _region_worker_v2,
            tasks,
            _consume,
            n_workers=n_workers,
        )
    finally:
        train_writer.close()
        test_writer.close()

    train_rows = pq.read_metadata(out_train).num_rows if out_train.exists() else 0
    test_rows = pq.read_metadata(out_test).num_rows if out_test.exists() else 0

    if duplicate_test_skipped:
        print(
            f"  Hinweis: {duplicate_test_skipped} doppelte Region-Durchläufe "
            "(test nur 1× geschrieben). train.csv ggf. nach region_id sortieren."
        )

    return {
        "version": 2,
        "regions": len(finished_regions),
        "duplicate_region_passes": duplicate_test_skipped,
        "train_labeled_rows": train_rows,
        "test_rows": test_rows,
        "out_train": out_train,
        "out_test": out_test,
        "feature_count": len(feature_columns_v2()),
        "n_workers": n_workers,
    }


def preprocess_panel_v2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    region_stats: pd.DataFrame | None = None,
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample-mode path; parallel over regions when n_workers > 1."""
    if region_stats is None:
        labeled = train[train["score"].notna()][["region_id", "score"]]
        region_stats = (
            labeled.groupby("region_id", sort=False)["score"]
            .agg(mean="mean", median="median", std="std")
            .rename(
                columns={
                    "mean": "region_score_mean",
                    "median": "region_score_median",
                    "std": "region_score_std",
                }
            )
            .reset_index()
        )
        region_stats["region_score_std"] = region_stats["region_score_std"].fillna(0.0)

    n_workers = n_workers if n_workers is not None else default_workers()
    regions = train["region_id"].unique()
    test_by = {r: test[test["region_id"] == r] for r in regions}

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    def _consume(res: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        tr, te = res
        if not tr.empty:
            train_parts.append(tr)
        if not te.empty:
            test_parts.append(te)

    tasks = (
        (train[train["region_id"] == r], test_by.get(r, pd.DataFrame()), region_stats)
        for r in regions
    )
    run_parallel_consume(_region_worker_v2, tasks, _consume, n_workers=n_workers)

    train_out = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_out = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return train_out, test_out
