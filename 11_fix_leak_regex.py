import json
import re

with open("notebooks/pipeline_combined.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    
    # 1. build_features_v2
    source = re.sub(r'def build_features_v2\(df: pd.DataFrame, region_stats: pd.DataFrame \| None = None\) -> pd.DataFrame:', 'def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:', source)
    
    # 2. panel = build_features_v2
    source = re.sub(r'panel = build_features_v2\(panel, region_stats=region_stats\)', 'panel = build_features_v2(panel)', source)
    
    # 3. process_region_v2_core signature
    source = re.sub(r'def process_region_v2_core\([\s\S]*?train_part: pd.DataFrame,[\s\S]*?test_part: pd.DataFrame,[\s\S]*?region_stats: pd.DataFrame,[\s\S]*?\)', 'def process_region_v2_core(\n    train_part: pd.DataFrame,\n    test_part: pd.DataFrame,\n)', source)
    
    # 4. _region_worker_v2 signature
    source = re.sub(r'def _region_worker_v2\([\s\S]*?args: tuple\[pd.DataFrame, pd.DataFrame, pd.DataFrame\],[\s\S]*?\)', 'def _region_worker_v2(\n    args: tuple[pd.DataFrame, pd.DataFrame],\n)', source)
    
    # 5. unpack args
    source = re.sub(r'train_r, test_r, region_stats = args', 'train_r, test_r = args', source)
    
    # 6. call process_region_v2_core
    source = re.sub(r'process_region_v2_core\(train_r, test_r, region_stats\)', 'process_region_v2_core(train_r, test_r)', source)
    source = re.sub(r'process_region_v2_core\(train_part, test_part, region_stats\)', 'process_region_v2_core(train_part, test_part)', source)
    
    # 7. _process_region_v2 signature
    source = re.sub(r'def _process_region_v2\([\s\S]*?train_part: pd.DataFrame,[\s\S]*?test_part: pd.DataFrame,[\s\S]*?region_stats: pd.DataFrame,[\s\S]*?train_writer: _ParquetAppender,[\s\S]*?test_writer: _ParquetAppender,[\s\S]*?\)', 'def _process_region_v2(\n    train_part: pd.DataFrame,\n    test_part: pd.DataFrame,\n    train_writer: _ParquetAppender,\n    test_writer: _ParquetAppender,\n)', source)
    
    # 8. _iter_region_tasks signature
    source = re.sub(r'def _iter_region_tasks\([\s\S]*?train_path: Path,[\s\S]*?test_by_region: dict,[\s\S]*?region_stats: pd.DataFrame,[\s\S]*?chunk_size: int,[\s\S]*?\)', 'def _iter_region_tasks(\n    train_path: Path,\n    test_by_region: dict,\n    chunk_size: int,\n)', source)
    
    # 9. yield
    source = re.sub(r'yield \(train_r, test_r, region_stats\)', 'yield (train_r, test_r)', source)
    
    # 10. preprocess_by_region_v2 signature & body
    if "def preprocess_by_region_v2" in source:
        # replace signature
        source = re.sub(r'def preprocess_by_region_v2\([\s\S]*?n_workers: int \| None = None,[\s\S]*?\) -> dict:', 'def preprocess_by_region_v2(\n    train_path: Path,\n    test_path: Path,\n    out_train: Path,\n    out_test: Path,\n    chunk_size: int = 500_000,\n    n_workers: int | None = None,\n) -> dict:', source)
        
        # remove region_stats calculation
        source = re.sub(r'print\("Berechne region_stats vorab\.\.\."\)[\s\S]*?del train_scores, labeled\s*print\(f"  -> \{len\(region_stats\)\} Regionen berechnet\."\)', '', source)
        
        # remove from tasks call
        source = re.sub(r'tasks = _iter_region_tasks\(train_path, test_by_region, region_stats, chunk_size\)', 'tasks = _iter_region_tasks(train_path, test_by_region, chunk_size)', source)
    
    lines = [line + "\n" for line in source.split("\n")]
    if lines[-1] == "\n":
        lines.pop()
    cell["source"] = lines

with open("notebooks/pipeline_combined.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Applied strict regex cleanup to remove region_stats!")
