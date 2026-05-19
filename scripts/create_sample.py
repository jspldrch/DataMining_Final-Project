#!/usr/bin/env python3
"""Create train_sample.csv (first 10k rows) for local development."""

import sys
from pathlib import Path

import pandas as pd

# Allow imports from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.paths import resolve_data_dir  # noqa: E402

DATA_DIR = resolve_data_dir()
INPUT_PATH = DATA_DIR / "train.csv"
OUTPUT_PATH = DATA_DIR / "train_sample.csv"
NROWS = 10_000


def main() -> None:
    print(f"Reading first {NROWS:,} rows from {INPUT_PATH}...")
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        print("Place train.csv in data/ (see docs/DATA_SETUP.md).")
        sys.exit(1)

    df_sample = pd.read_csv(INPUT_PATH, nrows=NROWS)
    df_sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Created: {OUTPUT_PATH} ({len(df_sample):,} rows)")


if __name__ == "__main__":
    main()
