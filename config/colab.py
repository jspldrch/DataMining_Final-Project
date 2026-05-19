"""
Google Colab bootstrap: code from GitHub, CSVs only on Drive.

Usage in Colab (first cell of any notebook):

    from config.colab import bootstrap
    env = bootstrap()
    # env["project_root"], env["data_dir"], env["train_path"], ...
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --- Anpassen falls anderes Repo / Branch ---
GITHUB_REPO_URL = "https://github.com/jspldrch/DataMining_Final-Project.git"
GITHUB_BRANCH = "main"

# Code wird hier geklont (RAM in /content, nicht auf Drive)
COLAB_REPO_DIR = Path("/content/DataMining_Final-Project")

# Nur CSVs liegen dauerhaft auf Drive (einmal hochladen)
COLAB_DRIVE_DATA_CANDIDATES = [
    Path("/content/drive/MyDrive/DataMining/DataMining_Final-Project/data"),
    Path("/content/drive/MyDrive/DataMining/data"),
]
COLAB_DRIVE_DATA_DIR = COLAB_DRIVE_DATA_CANDIDATES[0]

# Optional: Outputs auf Drive behalten (Parquet, Submissions)
COLAB_DRIVE_OUTPUTS_DIR = Path("/content/drive/MyDrive/DataMining/outputs")


def resolve_drive_data_dir() -> Path:
    """First Drive folder that contains train.csv."""
    for path in COLAB_DRIVE_DATA_CANDIDATES:
        if (path / "train.csv").exists() and (path / "test.csv").exists():
            return path
    tried = "\n".join(f"  - {p}" for p in COLAB_DRIVE_DATA_CANDIDATES)
    raise FileNotFoundError(
        f"train.csv / test.csv auf Drive nicht gefunden.\n"
        f"Geprüft:\n{tried}\n"
        "Lege CSVs in einen der Ordner (nur einmal nötig)."
    )


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def bootstrap(
    repo_url: str = GITHUB_REPO_URL,
    branch: str = GITHUB_BRANCH,
    drive_data_dir: Path | None = None,
    link_outputs_to_drive: bool = True,
    install_requirements: bool = True,
) -> dict:
    """
    1. Mount Google Drive
    2. git clone or git pull → /content/DataMining_Final-Project
    3. pip install -r requirements.txt
    4. Optional: symlink outputs/ → Drive
  5. Return paths (data always from Drive, not from git)
    """
    try:
        from google.colab import drive  # noqa: F401
    except ImportError as e:
        raise RuntimeError("bootstrap() only works in Google Colab") from e

    from google.colab import drive as drive_mod

    drive_mod.mount("/content/drive")

    repo_dir = COLAB_REPO_DIR
    if (repo_dir / ".git").exists():
        print(f"git pull in {repo_dir}")
        _run(["git", "pull", "origin", branch], cwd=repo_dir)
    else:
        if repo_dir.exists():
            import shutil

            shutil.rmtree(repo_dir)
        print(f"git clone → {repo_dir}")
        _run(["git", "clone", "--branch", branch, repo_url, str(repo_dir)])

    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    if install_requirements:
        print("pip install …")
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=repo_dir)

    data_dir = drive_data_dir or resolve_drive_data_dir()

    project_outputs = repo_dir / "outputs"
    project_outputs.mkdir(parents=True, exist_ok=True)

    if link_outputs_to_drive:
        COLAB_DRIVE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("processed", "submissions", "figures", "regional"):
            drive_sub = COLAB_DRIVE_OUTPUTS_DIR / name
            drive_sub.mkdir(parents=True, exist_ok=True)
            link = project_outputs / name
            if link.is_symlink():
                link.unlink()
            elif link.exists() and not link.is_symlink():
                pass  # keep existing local folder
            else:
                if not link.exists():
                    link.symlink(drive_sub, target_is_directory=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    print("✓ Setup fertig")
    print(f"  Code (git):  {repo_dir}")
    print(f"  Daten:       {data_dir}")
    print(f"  Train:       {train_path} ({train_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  Test:        {test_path}")

    return {
        "project_root": repo_dir,
        "data_dir": data_dir,
        "train_path": train_path,
        "test_path": test_path,
        "outputs_dir": project_outputs,
        "is_colab": True,
        "use_chunked_train": True,
        "chunk_size": 500_000,
        "env_label": "Colab (git + Drive-Daten)",
    }
