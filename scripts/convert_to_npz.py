"""
convert_to_npz.py  –  CSV → compressed NPZ converter

Converts train.csv and test.csv to numpy compressed NPZ format.
NPZ files are typically 60-80% smaller than CSV because:
  - float32 binary storage instead of text
  - numpy zlib compression
  - integer codes for region_id (no repeated strings)

Run once locally, then upload the NPZ files to Kaggle.

Usage:
    python scripts/convert_to_npz.py

Output:
    data/train.npz
    data/test.npz
    data/sample_submission.npz   (small, but consistent)

Then modify DATA_DIR in your training script to point to the Kaggle input path,
e.g. /kaggle/input/your-dataset-name/ and use the load_npz() helper below.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]


def convert_csv_to_npz(csv_path: Path, npz_path: Path, has_score: bool = True) -> None:
    """Read a CSV and save as compressed NPZ."""
    print(f"  Reading {csv_path.name} ...")
    t0  = time.time()
    df  = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}  Columns: {df.columns.tolist()}")

    # ── Parse dates → separate int arrays (smaller than storing date strings) ──
    parts = df["date"].str.split("-", expand=True)
    year  = parts[0].astype(np.int32).values
    month = parts[1].astype(np.int32).values
    day   = parts[2].astype(np.int32).values

    # ── Encode region_id as integers (store mapping separately) ──────────────
    unique_regions = np.sort(df["region_id"].unique())          # sorted for consistency
    r2i            = {r: i for i, r in enumerate(unique_regions)}
    region_ids     = np.array([r2i[r] for r in df["region_id"]], dtype=np.int32)

    # ── Build arrays dict ─────────────────────────────────────────────────────
    arrays: dict = {
        "region_id":    region_ids,           # int32 codes
        "region_names": unique_regions,        # string → int mapping (dtype=object)
        "year":         year,
        "month":        month,
        "day":          day,
    }

    # Weather features
    for col in WEATHER_COLS:
        if col in df.columns:
            arrays[col] = df[col].values.astype(np.float32)
        else:
            print(f"  WARNING: column '{col}' not found in {csv_path.name}")

    # Score (only in train)
    if has_score and "score" in df.columns:
        arrays["score"] = pd.to_numeric(df["score"], errors="coerce").values.astype(np.float32)

    # ── Save compressed ───────────────────────────────────────────────────────
    np.savez_compressed(npz_path, **arrays)
    elapsed = time.time() - t0
    csv_mb  = csv_path.stat().st_size  / 1e6
    npz_mb  = npz_path.stat().st_size  / 1e6
    ratio   = (1 - npz_mb / csv_mb) * 100
    print(f"  Saved → {npz_path.name}  "
          f"({csv_mb:.0f} MB → {npz_mb:.1f} MB, -{ratio:.0f}%)  "
          f"[{elapsed:.1f}s]")


def convert_submission_to_npz(csv_path: Path, npz_path: Path) -> None:
    """Convert sample_submission.csv to NPZ."""
    df = pd.read_csv(csv_path)
    unique_regions = np.sort(df["region_id"].unique())
    r2i = {r: i for i, r in enumerate(unique_regions)}
    arrays = {
        "region_id":    np.array([r2i[r] for r in df["region_id"]], dtype=np.int32),
        "region_names": unique_regions,
    }
    for col in df.columns:
        if col.startswith("pred_week"):
            arrays[col] = df[col].values.astype(np.float32)
    np.savez_compressed(npz_path, **arrays)
    print(f"  Saved → {npz_path.name}  ({npz_path.stat().st_size/1e3:.0f} KB)")


def load_npz(npz_path: Path, weather_cols: list[str] = WEATHER_COLS) -> pd.DataFrame:
    """
    Load an NPZ file and reconstruct a pandas DataFrame.
    Compatible with DataFrames produced by pd.read_csv on the original CSV.
    """
    d             = np.load(npz_path, allow_pickle=True)
    region_names  = d["region_names"]
    region_ids    = d["region_id"]

    df = pd.DataFrame({
        "region_id": region_names[region_ids],
        "year":      d["year"].astype(np.int32),
        "month":     d["month"].astype(np.int32),
        "day":       d["day"].astype(np.int32),
    })

    # Reconstruct date string (needed by scripts that call date.str.split)
    df["date"] = (
        df["year"].astype(str) + "-"
        + df["month"].astype(str).str.zfill(2) + "-"
        + df["day"].astype(str).str.zfill(2)
    )

    for col in weather_cols:
        if col in d:
            df[col] = d[col].astype(np.float32)

    if "score" in d:
        df["score"] = d["score"].astype(np.float32)

    return df


def smart_load(data_dir: Path, filename_stem: str,
               weather_cols: list[str] = WEATHER_COLS) -> pd.DataFrame:
    """
    Load data preferring NPZ (faster, smaller) over CSV.
    Works on both local machine and Kaggle.

    Usage in training script:
        train = smart_load(DATA_DIR, "train")
        test  = smart_load(DATA_DIR, "test")
    """
    npz_path = data_dir / f"{filename_stem}.npz"
    csv_path = data_dir / f"{filename_stem}.csv"

    if npz_path.exists():
        return load_npz(npz_path, weather_cols)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, dtype={c: "float32" for c in weather_cols})
        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce").astype("float32")
        return df
    else:
        raise FileNotFoundError(f"Neither {npz_path} nor {csv_path} found.")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_total = time.time()
    print("=" * 60)
    print("  CSV → NPZ Converter")
    print("=" * 60)

    # train.csv
    train_csv = DATA_DIR / "train.csv"
    train_npz = DATA_DIR / "train.npz"
    if train_csv.exists():
        print(f"\n[1/3] Converting train.csv ...")
        convert_csv_to_npz(train_csv, train_npz, has_score=True)
    else:
        print(f"  SKIP: {train_csv} not found")

    # test.csv
    test_csv = DATA_DIR / "test.csv"
    test_npz = DATA_DIR / "test.npz"
    if test_csv.exists():
        print(f"\n[2/3] Converting test.csv ...")
        convert_csv_to_npz(test_csv, test_npz, has_score=False)
    else:
        print(f"  SKIP: {test_csv} not found")

    # sample_submission.csv
    sub_csv = ROOT / "resources" / "sample_submission.csv"
    sub_npz = DATA_DIR / "sample_submission.npz"
    if sub_csv.exists():
        print(f"\n[3/3] Converting sample_submission.csv ...")
        convert_submission_to_npz(sub_csv, sub_npz)
    else:
        print(f"  SKIP: {sub_csv} not found")

    print(f"\n{'='*60}")
    print(f"  Done in {time.time()-t_total:.1f}s")
    print(f"  Upload these files to Kaggle:")
    for f in [train_npz, test_npz, sub_npz]:
        if f.exists():
            print(f"    {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
    print(f"{'='*60}\n")

    print("How to use in training scripts:")
    print("  from scripts.convert_to_npz import smart_load")
    print("  DATA_DIR = Path('/kaggle/input/your-dataset-name')")
    print("  train = smart_load(DATA_DIR, 'train')")
    print("  test  = smart_load(DATA_DIR, 'test')")
