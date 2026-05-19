"""
Memory-safe preprocessing: one region at a time (~5.5k rows in RAM).

Requires train.csv sorted by region_id (true for this dataset).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.features import (
    add_persistence_baseline,
    build_features,
    feature_columns,
    parse_dates,
)


def _meta_train_cols() -> list[str]:
    return [
        "region_id",
        "date",
        "year",
        "month",
        "day",
        "ordinal",
        "score",
        "score_persist7",
    ]


def _save_cols(df: pd.DataFrame, labeled: bool) -> list[str]:
    meta = _meta_train_cols() if labeled else [
        "region_id",
        "date",
        "year",
        "month",
        "day",
        "ordinal",
    ]
    feats = feature_columns()
    return [c for c in list(dict.fromkeys(meta + feats)) if c in df.columns]


class _ParquetAppender:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._writer: pq.ParquetWriter | None = None

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _process_region(
    train_part: pd.DataFrame,
    test_part: pd.DataFrame,
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
    panel = build_features(panel)
    panel = add_persistence_baseline(panel, lag_days=7)

    train_feat = panel[panel["_split"] == "train"]
    test_feat = panel[panel["_split"] == "test"]
    train_labeled = train_feat[train_feat["score"].notna()]

    train_writer.write(train_labeled[_save_cols(train_labeled, labeled=True)])
    test_writer.write(test_feat[_save_cols(test_feat, labeled=False)])


def preprocess_by_region(
    train_path: Path,
    test_path: Path,
    out_train: Path,
    out_test: Path,
    chunk_size: int = 500_000,
) -> dict:
    """
    Stream train.csv (sorted by region_id) and write parquet incrementally.
    Peak RAM: one region (~5.5k rows) + test slice + chunk buffer.
    """
    test = parse_dates(pd.read_csv(test_path))
    test_by_region = {r: g for r, g in test.groupby("region_id", sort=False)}

    for p in (out_train, out_test):
        if p.exists():
            p.unlink()

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

                # Region wechsel → vorherige Region ist vollständig (sortierte Datei)
                train_r = pd.concat(buffer_parts, ignore_index=True)
                test_r = test_by_region.get(current_region, pd.DataFrame())
                _process_region(train_r, test_r, train_writer, test_writer)
                n_regions += 1
                if n_regions % 200 == 0:
                    print(f"  … {n_regions} Regionen verarbeitet")

                current_region = region
                buffer_parts = [g]

        if current_region is not None and buffer_parts:
            train_r = pd.concat(buffer_parts, ignore_index=True)
            test_r = test_by_region.get(current_region, pd.DataFrame())
            _process_region(train_r, test_r, train_writer, test_writer)
            n_regions += 1

    finally:
        train_writer.close()
        test_writer.close()

    train_rows = pq.read_metadata(out_train).num_rows if out_train.exists() else 0
    test_rows = pq.read_metadata(out_test).num_rows if out_test.exists() else 0

    return {
        "regions": n_regions,
        "train_labeled_rows": train_rows,
        "test_rows": test_rows,
        "out_train": out_train,
        "out_test": out_test,
    }
