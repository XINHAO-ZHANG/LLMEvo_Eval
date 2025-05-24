# ================================================
#  evovle_core/executor.py — batch evaluation wrapper
# ================================================

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, List
import os

Genome = Any

def batch_eval(task_mod, genomes: List[Genome], n_worker: int | None = None) -> List[float]:
    """
    Evaluate *genomes* in parallel using task_mod.fitness().

    Parameters
    ----------
    n_worker : Optional[int]
        Number of threads. Defaults to os.cpu_count().
    """
    n_worker = n_worker or os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=n_worker) as ex:
        return list(ex.map(task_mod.fitness, genomes))