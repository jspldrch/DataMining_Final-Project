"""
Colab / local path setup. Used by notebooks 03 and 04.

Notebooks must run git clone/pull first so this file exists under REPO_DIR.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

GITHUB_REPO_URL = "https://github.com/jspldrch/DataMining_Final-Project.git"
GITHUB_BRANCH = "main"
COLAB_REPO_DIR = Path("/content/DataMining_Final-Project")

# CSVs + Outputs auf Drive (persistent)
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


def git_clone_or_pull(repo_dir: Path = COLAB_REPO_DIR) -> Path:
    """Clone repo to /content or pull latest. Returns repo path."""
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


def find_local_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent]
    for p in candidates:
        if (p / "scripts" / "features.py").exists():
            return p.resolve()
    return Path.cwd().resolve()


def resolve_drive_data_dir() -> Path:
    for path in DRIVE_DATA_CANDIDATES:
        if (path / "train.csv").exists() and (path / "test.csv").exists():
            return path
    tried = "\n".join(f"  - {p}" for p in DRIVE_DATA_CANDIDATES)
    raise FileNotFoundError(f"train.csv/test.csv nicht auf Drive.\nGeprüft:\n{tried}")


def init_environment(
    *,
    install_deps: bool = True,
    skip_mount: bool = False,
    skip_git: bool = False,
) -> dict:
    """
    Colab: mount Drive, git pull, pip install.
    Local: use repo next to notebooks/.

    Returns dict with project_root, data_dir, train_path, test_path,
    processed_dir, submission_dir, is_colab, mode.
    """
    colab = is_colab()

    if colab:
        if not skip_mount:
            from google.colab import drive

            drive.mount("/content/drive")
        if not skip_git:
            project_root = git_clone_or_pull(COLAB_REPO_DIR)
        else:
            project_root = COLAB_REPO_DIR
        data_dir = resolve_drive_data_dir()
        outputs_dir = DRIVE_PROJECT_DIR / "outputs"
        mode = "full"
    else:
        project_root = find_local_root()
        data_dir = project_root / "data"
        if not (data_dir / "train.csv").exists() and (data_dir / "train_sample.csv").exists():
            mode = "sample"
        else:
            mode = "full"
        outputs_dir = project_root / "outputs"

    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if install_deps:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            cwd=project_root,
            check=True,
        )

    processed_dir = outputs_dir / "processed"
    submission_dir = outputs_dir / "submissions"
    processed_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    if mode == "sample":
        train_path = data_dir / "train_sample.csv"
    else:
        train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    print(f"Umgebung: {'Colab' if colab else 'Lokal'} | Modus: {mode}")
    print(f"  Code:    {project_root}")
    print(f"  Daten:   {data_dir}")
    print(f"  Train:   {train_path} ({'OK' if train_path.exists() else 'FEHLT'})")
    print(f"  Test:    {test_path} ({'OK' if test_path.exists() else 'FEHLT'})")
    print(f"  Outputs: {processed_dir}")

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
    }
