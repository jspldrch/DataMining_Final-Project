"""
Central path configuration for local development and Google Colab.

Usage in notebooks:
    from config.paths import setup_environment
    env = setup_environment()
    TRAIN_PATH = env["train_path"]
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Legacy competition folder name (still supported)
LEGACY_DATA_DIR = PROJECT_ROOT / "data-mining-2026-final-project" / "data"
CANONICAL_DATA_DIR = PROJECT_ROOT / "data"

# Colab: first match with train.csv wins (see resolve_colab_data_dir)
COLAB_DATA_DIR = "/content/drive/MyDrive/DataMining/DataMining_Final-Project/data"
COLAB_DATA_CANDIDATES = [
    "/content/drive/MyDrive/DataMining/DataMining_Final-Project/data",
    "/content/drive/MyDrive/DataMining/data",
]

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def resolve_data_dir() -> Path:
    """Return the first existing data directory (canonical or legacy)."""
    if CANONICAL_DATA_DIR.exists():
        return CANONICAL_DATA_DIR
    if LEGACY_DATA_DIR.exists():
        return LEGACY_DATA_DIR
    return CANONICAL_DATA_DIR


def resolve_colab_data_dir(project_root: Path | None = None) -> Path:
    """Pick first Colab Drive path that contains train.csv."""
    root = project_root or PROJECT_ROOT
    candidates = [
        str(root / "data"),
        *COLAB_DATA_CANDIDATES,
        str(CANONICAL_DATA_DIR),
    ]
    for raw in candidates:
        path = Path(raw)
        if (path / "train.csv").exists():
            return path
    return Path(COLAB_DATA_DIR)


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def setup_environment() -> dict:
    """
    Detect environment and return paths + flags.

    Returns dict with keys:
        is_colab, data_dir, train_path, test_path, figures_dir,
        use_chunked_train, chunk_size, env_label
    """
    colab = is_colab()
    figures_dir = FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    if colab:
        data_dir = resolve_colab_data_dir()
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"
        use_chunked = True
        env_label = "Colab (voller Trainingsdatensatz)"
    else:
        data_dir = resolve_data_dir()
        sample_path = data_dir / "train_sample.csv"
        full_path = data_dir / "train.csv"
        train_path = sample_path if sample_path.exists() else full_path
        test_path = data_dir / "test.csv"
        use_chunked = False
        env_label = "Lokal (train_sample.csv)" if sample_path.exists() else "Lokal (train.csv)"

    return {
        "is_colab": colab,
        "project_root": PROJECT_ROOT,
        "data_dir": data_dir,
        "train_path": train_path,
        "test_path": test_path,
        "figures_dir": figures_dir,
        "use_chunked_train": use_chunked,
        "chunk_size": 500_000,
        "env_label": env_label,
    }
