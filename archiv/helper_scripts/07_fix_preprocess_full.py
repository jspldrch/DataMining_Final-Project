import json

with open("notebooks/preprocessing.ipynb", "r") as f:
    prep_nb = json.load(f)

for cell in prep_nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    
    if "def preprocess_by_region_v2" in source:
        new_source = """def preprocess_by_region_v2(
    train_path: Path,
    test_path: Path,
    out_train: Path,
    out_test: Path,
    chunk_size: int = 500_000,
    n_workers: int | None = None,
) -> dict:
    n_workers = n_workers if n_workers is not None else default_workers()
    
    print("Berechne region_stats vorab...")
    train_scores = pd.read_csv(train_path, usecols=["region_id", "score"])
    labeled = train_scores[train_scores["score"].notna()]
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
    del train_scores, labeled
    print(f"  -> {len(region_stats)} Regionen berechnet.")

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
            "(test nur 1× geschrieben)."
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
"""
        # Replace only the function definition, preserve the rest of the cell
        parts = source.split("def preprocess_by_region_v2(")
        if len(parts) == 2:
            new_cell_source = parts[0] + new_source
            lines = [line + "\n" for line in new_cell_source.split("\n")]
            if lines[-1] == "\n":
                lines.pop()
            cell["source"] = lines
        else:
            # Fallback if multiple matches or parsing error
            import re
            new_cell_source = re.sub(r'def preprocess_by_region_v2\(.*?return \{.*?\}', new_source, source, flags=re.DOTALL)
            lines = [line + "\n" for line in new_cell_source.split("\n")]
            if lines[-1] == "\n":
                lines.pop()
            cell["source"] = lines

with open("notebooks/preprocessing.ipynb", "w") as f:
    json.dump(prep_nb, f, indent=1)

print("Fixed preprocess_by_region_v2 in preprocessing.ipynb")
