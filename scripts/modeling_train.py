"""Parallel LightGBM training (one process per forecast week)."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class WeekTrainSpec:
    week: int
    X_tr: pd.DataFrame
    y_tr: np.ndarray
    X_va: pd.DataFrame
    y_va: np.ndarray
    params: dict
    categorical_feature: list[str] | None
    early_stopping_rounds: int


def _fit_one_week(spec: WeekTrainSpec) -> lgb.LGBMRegressor:
    p = dict(spec.params)
    p["random_state"] = p.get("random_state", 42) + spec.week
    p["n_jobs"] = 1  # one process per week
    m = lgb.LGBMRegressor(**p)
    fit_kw: dict = {}
    if spec.categorical_feature:
        fit_kw["categorical_feature"] = spec.categorical_feature
    if spec.X_va is not None and len(spec.X_va):
        fit_kw["eval_set"] = [(spec.X_va, spec.y_va)]
        fit_kw["eval_metric"] = "mae"
        fit_kw["callbacks"] = [
            lgb.early_stopping(spec.early_stopping_rounds, verbose=False)
        ]
    m.fit(spec.X_tr, spec.y_tr[:, spec.week], **fit_kw)
    return m


def _fit_one_week_final(spec: WeekTrainSpec) -> lgb.LGBMRegressor:
    p = dict(spec.params)
    p["random_state"] = p.get("random_state", 42) + spec.week
    p["n_jobs"] = 1
    m = lgb.LGBMRegressor(**p)
    fit_kw: dict = {}
    if spec.categorical_feature:
        fit_kw["categorical_feature"] = spec.categorical_feature
    m.fit(spec.X_tr, spec.y_tr[:, spec.week], **fit_kw)
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

    fit_fn = _fit_one_week_final if final_fit else _fit_one_week
    specs = [
        WeekTrainSpec(
            week=w,
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            params=params,
            categorical_feature=categorical_feature,
            early_stopping_rounds=early_stopping_rounds,
        )
        for w in range(5)
    ]
    return run_parallel_map(fit_fn, specs, n_workers=min(n_workers, 5))
