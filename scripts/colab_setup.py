"""Backward-compatible alias — use scripts.project_env.bootstrap_notebook."""
from scripts.project_env import (  # noqa: F401
    bootstrap_notebook,
    bootstrap_notebook as init_environment,
    find_local_root,
    git_clone_or_pull,
    is_colab,
    resolve_data_dir,
    resolve_mode,
)
