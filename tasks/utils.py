"""
Shared helpers for multiple tasks.
"""

from __future__ import annotations

import functools
from pathlib import Path
import math
import random
from typing import List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# TSP utilities
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=32)
def random_points(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 120, size=(n, 2))


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = np.abs(coords[:, None, :] - coords[None, :, :])
    return np.sum(diff, axis=-1)



# --------------------------------------------------------------------------- #
# Graph-coloring utilities
# --------------------------------------------------------------------------- #
def random_graph(n: int, p: float, seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    return edges
