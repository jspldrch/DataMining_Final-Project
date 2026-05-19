"""
Unified notebook environment (local + Google Colab).

Only differences between environments:
  - Colab: git clone/pull → /content/DataMining_Final-Project; CSVs on Drive
  - Local: repo root from cwd; CSVs in data/ (same filenames as Colab)

Same MODE rules, same outputs/ layout, same preprocessing & modeling code path.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

GITHUB_REPO_URL = "https://github.com/jspldrch/DataMining_Final-Project.git"
GITHUB_BRANCH = "main"
COLAB_REPO_DIR = Path("/content/DataMining_Final-Project")
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/DataMining/DataMining_Final-Project")

DRIVE_DATA_CANDIDATES = [
    DRIVE_PROJECT_DIR / "data",
    Path("/content/drive/MyDrive/DataMining/data"),
]


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def find_local_root() -> Path:
    """Repo root when running from notebooks/ or project root."""
    candidates = [Path.cwd(), Path.cwd().parent]
    for path in candidates:
        if (path / "scripts" / "features.py").exists():
            return path.resolve()
    raise FileNotFoundError(
        "Projektroot nicht gefunden (scripts/features.py). "
        "Notebook aus notebooks/ starten oder ins Repo-Verzeichnis wechseln."
    )


def git_clone_or_pull(repo_dir: Path = COLAB_REPO_DIR) -> Path:
    if (repo_dir / ".git").exists():
        subprocess.run(["git", "pull", "origin", GITHUB_BRANCH], cwd=repo_dir, check=True)
        print(f"git pull OK → {repo_dir}")
    else:
        if repo_dir.exists():
            import shutil
            shutil.rmtree(repo_dir)
        subprocess.run(
            ["git", "clone", "--branch", GITHUB_BRANCH, GITHUB_REPO_URL, str(repo_dir)],
            check=True,
        )
        print(f"git clone OK → {repo_dir}")
    return repo_dir


def _local_data_candidates(project_root: Path) -> list[Path]:
    return [
        project_root / "data",
        project_root / "data-mining-2026-final-project" / "data",
    ]


def resolve_data_dir(project_root: Path, *, colab: bool) -> Path:
    """First directory that contains test.csv and train.csv or train_sample.csv."""
    candidates = DRIVE_DATA_CANDIDATES if colab else _local_data_candidates(project_root)
    for path in candidates:
        if not (path / "test.csv").exists():
            continue
        if (path / "train.csv").exists() or (path / "train_sample.csv").exists():
            return path
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Datenordner nicht gefunden. Erwartet train.csv + test.csv\n"
        f"Geprüft:\n{tried}"
    )


def resolve_mode(data_dir: Path) -> str:
    """
    full  = train.csv (same as Colab)
    sample = only if train.csv missing but train_sample.csv exists
    """
    force = os.environ.get("DM_MODE", "").strip().lower()
    if force in ("full", "sample"):
        return force
    if (data_dir / "train.csv").exists():
        return "full"
    if (data_dir / "train_sample.csv").exists():
        return "sample"
    raise FileNotFoundError(f"Weder train.csv noch train_sample.csv in {data_dir}")


def _pip_install(project_root: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(project_root / "requirements.txt")],
        cwd=project_root,
        check=True,
    )


def bootstrap_notebook(*, install_deps: bool = True) -> dict:
    """
    One entry cell for notebooks 03 & 04.

    Colab: mount Drive → git pull → pip → Drive data + Drive outputs
    Local: repo root → pip → data/ → outputs/
    """
    colab = is_colab()

    if colab:
        from google.colab import drive
        drive.mount("/content/drive")
        project_root = git_clone_or_pull(COLAB_REPO_DIR)
        data_dir = resolve_data_dir(project_root, colab=True)
        outputs_dir = DRIVE_PROJECT_DIR / "outputs"
    else:
        project_root = find_local_root()
        data_dir = resolve_data_dir(project_root, colab=False)
        outputs_dir = project_root / "outputs"

    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if install_deps:
        _pip_install(project_root)

    mode = resolve_mode(data_dir)
    processed_dir = outputs_dir / "processed"
    submission_dir = outputs_dir / "submissions"
    figures_dir = outputs_dir / "figures"
    for directory in (processed_dir, submission_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / ("train.csv" if mode == "full" else "train_sample.csv")
    test_path = data_dir / "test.csv"

    print(f"Umgebung: {'Colab' if colab else 'Lokal'} | Modus: {mode}")
    print(f"  Code:    {project_root}")
    print(f"  Daten:   {data_dir}")
    print(f"  Train:   {train_path} ({'OK' if train_path.exists() else 'FEHLT'})")
    print(f"  Test:    {test_path} ({'OK' if test_path.exists() else 'FEHLT'})")
    print(f"  Outputs: {outputs_dir}")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train- oder Test-CSV fehlt — siehe Pfade oben.")

    return {
        "is_colab": colab,
        "mode": mode,
        "project_root": project_root,
        "data_dir": data_dir,
        "train_path": train_path,
        "test_path": test_path,
        "processed_dir": processed_dir,
        "submission_dir": submission_dir,
        "outputs_dir": outputs_dir,
        "figures_dir": figures_dir,
    }


def load_script(name: str, path: Path) -> ModuleType:
    """Import a .py file from scripts/ without package install."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Required for @dataclass and other decorators on Python 3.12+ (Colab).
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
