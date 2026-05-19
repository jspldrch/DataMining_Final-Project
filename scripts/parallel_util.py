"""Shared parallelism helpers for preprocessing and modeling."""
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_workers(cap: int = 8) -> int:
    """
    Worker count: env ``DM_WORKERS``, else ``CPU-1`` (capped).

    Set ``DM_WORKERS=1`` to disable multiprocessing.
    """
    env = os.environ.get("DM_WORKERS", "").strip()
    if env:
        return max(1, int(env))
    import multiprocessing as mp

    return max(1, min(cap, mp.cpu_count() - 1))


def run_parallel_map(
    fn: Callable[[T], R],
    tasks: list[T],
    *,
    n_workers: int | None = None,
) -> list[R]:
    """``pool.map`` — order preserved; fine for small task lists (e.g. 5 weeks)."""
    if not tasks:
        return []
    n_workers = n_workers or default_workers()
    if n_workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, tasks))


def run_parallel_consume(
    fn: Callable[[T], R],
    tasks: Iterator[T],
    consume: Callable[[R], None],
    *,
    n_workers: int | None = None,
    max_inflight: int | None = None,
) -> int:
    """
    Stream tasks through a process pool; call *consume* on each result as ready.
    Returns number of tasks completed.
    """
    n_workers = n_workers or default_workers()
    max_inflight = max_inflight or n_workers * 2
    n_done = 0

    if n_workers <= 1:
        for task in tasks:
            consume(fn(task))
            n_done += 1
        return n_done

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        inflight: dict = {}
        task_iter = iter(tasks)

        def submit_one() -> bool:
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            inflight[pool.submit(fn, task)] = True
            return True

        for _ in range(max_inflight):
            if not submit_one():
                break

        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut)
                consume(fut.result())
                n_done += 1
                submit_one()
    return n_done
