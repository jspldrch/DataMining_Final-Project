"""Parallel LightGBM training (one process per forecast week)."""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

# (week, X_tr, y_tr, X_va, y_va, params, categorical_feature, early_stopping_rounds)
WeekSpec = tuple[
    int,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    dict[str, Any],
    list[str] | None,
    int,
]


def labels_for_week(y: np.ndarray, week: int) -> list[float]:
    """
    LightGBM 4.x: no ``y[:, week]`` slices (train *and* eval_set labels).

    Plain ``list`` is accepted reliably; 2D ``y_va`` in eval_set causes:
    TypeError: Wrong type(ndarray) for label.
    """
    return np.asarray(y, dtype=np.float64)[:, week].ravel().tolist()


def _fit_one_week(spec: WeekSpec) -> lgb.LGBMRegressor:
    week, X_tr, y_tr, X_va, y_va, params, categorical_feature, es_rounds = spec
    p = dict(params)
    p["random_state"] = p.get("random_state", 42) + week
    p["n_jobs"] = 1
    m = lgb.LGBMRegressor(**p)
    fit_kw: dict = {}
    if categorical_feature:
        fit_kw["categorical_feature"] = categorical_feature
    if X_va is not None and len(X_va):
        fit_kw["eval_set"] = [(X_va, labels_for_week(y_va, week))]
        fit_kw["eval_metric"] = "mae"
        fit_kw["callbacks"] = [lgb.early_stopping(es_rounds, verbose=False)]
    m.fit(X_tr, labels_for_week(y_tr, week), **fit_kw)
    return m


def _fit_one_week_final(spec: WeekSpec) -> lgb.LGBMRegressor:
    week, X_tr, y_tr, _X_va, _y_va, params, categorical_feature, _es = spec
    p = dict(params)
    p["random_state"] = p.get("random_state", 42) + week
    p["n_jobs"] = 1
    m = lgb.LGBMRegressor(**p)
    fit_kw: dict = {}
    if categorical_feature:
        fit_kw["categorical_feature"] = categorical_feature
    m.fit(X_tr, labels_for_week(y_tr, week), **fit_kw)
    return m


def train_week_models_parallel(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    params: dict,
    *,
    categorical_feature: list[str] | None = None,
    early_stopping_rounds: int = 50,
    n_workers: int = 5,
    final_fit: bool = False,
) -> list[lgb.LGBMRegressor]:
    """Train weeks 0..4 in parallel (5 processes by default)."""
    from scripts.parallel_util import run_parallel_map

    n_workers = min(n_workers, 5)
    fit_fn = _fit_one_week_final if final_fit else _fit_one_week
    specs: list[WeekSpec] = [
        (
            w,
            X_tr,
            y_tr,
            X_va,
            y_va,
            params,
            categorical_feature,
            early_stopping_rounds,
        )
        for w in range(5)
    ]
    if n_workers <= 1:
        return [fit_fn(s) for s in specs]
    return run_parallel_map(fit_fn, specs, n_workers=n_workers)
