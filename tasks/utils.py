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
# Graph-coloring utilities
# --------------------------------------------------------------------------- #
def random_graph(n: int, p: float, seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    return edges


# --------------------------------------------------------------------------- #
# Text Evaluation Metrics (ROUGE, SARI)
# --------------------------------------------------------------------------- #

def compute_rouge(preds: list[str], refs: list[str]) -> dict:
    """
    计算ROUGE-1/2/L分数，严格调用HuggingFace evaluate库。
    preds: 生成的摘要列表
    refs:  参考摘要列表
    返回: dict, 包含rouge1/rouge2/rougeL等分数
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=preds, references=refs)
    return results


def compute_sari(sources: list[str], preds: list[str], refs: list[list[str]]) -> dict:
    """
    sources: 原始句子列表
    preds:   生成的简化句子列表
    refs:    多参考简化句子列表（每个元素是list of refs）
    返回: dict, 包含sari等分数
    """
    
    sari = evaluate.load("sari")
    results = sari.compute(sources=sources, predictions=preds, references=refs)
    return results
