"""
First import helper for notebooks (Colab cold-start + local path).

Usage in notebook cell 1:
    from scripts.notebook_init import setup
    env = setup()
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.project_env import GITHUB_BRANCH, GITHUB_REPO_URL, COLAB_REPO_DIR, bootstrap_notebook, is_colab


def ensure_repo_on_syspath() -> None:
    """Colab: clone repo if missing. Local: add project root to sys.path."""
    if is_colab():
        if not (COLAB_REPO_DIR / "scripts" / "project_env.py").exists():
            if COLAB_REPO_DIR.exists():
                import shutil
                shutil.rmtree(COLAB_REPO_DIR)
            subprocess.run(
                ["git", "clone", "--branch", GITHUB_BRANCH, GITHUB_REPO_URL, str(COLAB_REPO_DIR)],
                check=True,
            )
            print(f"git clone OK → {COLAB_REPO_DIR}")
        if str(COLAB_REPO_DIR) not in sys.path:
            sys.path.insert(0, str(COLAB_REPO_DIR))
        return

    for candidate in (Path.cwd(), Path.cwd().parent):
        if (candidate / "scripts" / "project_env.py").exists():
            root = candidate.resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return
    raise FileNotFoundError("scripts/project_env.py nicht gefunden — vom Projektroot oder notebooks/ starten.")


def setup(*, install_deps: bool = True) -> dict:
    ensure_repo_on_syspath()
    return bootstrap_notebook(install_deps=install_deps)
