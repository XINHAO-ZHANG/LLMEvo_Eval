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
import evaluate

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
# Text Evaluation Metrics (ROUGE, SARI)
# --------------------------------------------------------------------------- #

def compute_rouge(preds: list[str], refs: list[str]) -> dict:
    """
    Compute ROUGE-1/2/L scores using the HuggingFace evaluate library.
    preds: list of generated summaries
    refs:  list of reference summaries
    Returns: dict with rouge1/rouge2/rougeL scores
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=preds, references=refs)
    return results


def compute_sari(sources: list[str], preds: list[str], refs: list[list[str]]) -> dict:
    """
    Compute SARI score using the HuggingFace evaluate library.
    sources: list of original sentences
    preds:   list of simplified sentences
    refs:    list of reference simplifications (each element is a list of refs)
    Returns: dict with sari score
    """
    sari = evaluate.load("sari")
    results = sari.compute(sources=sources, predictions=preds, references=refs)
    return results
