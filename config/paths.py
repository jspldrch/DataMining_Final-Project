"""
Central path configuration — delegates to scripts.project_env (single source of truth).
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
COLAB_DATA_DIR = "/content/drive/MyDrive/DataMining/DataMining_Final-Project/data"
COLAB_DATA_CANDIDATES = [
    "/content/drive/MyDrive/DataMining/DataMining_Final-Project/data",
    "/content/drive/MyDrive/DataMining/data",
]
CANONICAL_DATA_DIR = PROJECT_ROOT / "data"
LEGACY_DATA_DIR = PROJECT_ROOT / "data-mining-2026-final-project" / "data"


def is_colab() -> bool:
    from scripts.project_env import is_colab as _is_colab
    return _is_colab()


def resolve_data_dir() -> Path:
    from scripts.project_env import find_local_root, resolve_data_dir as _resolve
    root = find_local_root() if not is_colab() else PROJECT_ROOT
    return _resolve(root, colab=is_colab())


def resolve_colab_data_dir(project_root: Path | None = None) -> Path:
    from scripts.project_env import resolve_data_dir as _resolve
    return _resolve(project_root or PROJECT_ROOT, colab=True)


def setup_environment(*, install_deps: bool = False) -> dict:
    """
    For notebooks 01/02. Same data paths as 03/04; optional pip install.
    """
    from scripts.project_env import bootstrap_notebook
    env = bootstrap_notebook(install_deps=install_deps)
    return {
        "is_colab": env["is_colab"],
        "project_root": env["project_root"],
        "data_dir": env["data_dir"],
        "train_path": env["train_path"],
        "test_path": env["test_path"],
        "figures_dir": env["figures_dir"],
        "use_chunked_train": env["mode"] == "full",
        "chunk_size": 500_000,
        "env_label": f"{'Colab' if env['is_colab'] else 'Lokal'} ({env['mode']})",
        "mode": env["mode"],
        "outputs_dir": env["outputs_dir"],
    }
