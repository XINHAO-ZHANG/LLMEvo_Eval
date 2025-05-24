"""
Graph-Coloring (500-node random graph) — minimise #conflicts.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ------------------------------ CONFIG ---------------------------------- #
# 支持两种模式：随机生成图 或 加载固定邻接矩阵
# 环境变量 GRAPH_MATRIX 指定路径，否则走随机模式
GRAPH_MATRIX = os.getenv("GRAPH_MATRIX", "")          # '' → use random graph
NODE_N       = int(os.getenv("GRAPH_N", 25))          # 节点数，随机模式下有效
EDGE_P       = float(os.getenv("GRAPH_P", 0.2))       # 随机模式下的 Erdos-Renyi 概率
_SEED        = int(os.getenv("GRAPH_SEED", 42))

# --------------------------- INSTANCE LOADING --------------------------- #
if GRAPH_MATRIX:
    _ADJ = np.loadtxt(Path(GRAPH_MATRIX), delimiter=",", dtype=int)
    if _ADJ.shape[0] != _ADJ.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    NODE_N = _ADJ.shape[0]
    _INSTANCE_ID = Path(GRAPH_MATRIX).stem
else:
    rng = np.random.RandomState(_SEED)
    # 生成 Erdos-Renyi 随机对称邻接矩阵（无自环）
    mat = rng.rand(NODE_N, NODE_N) < EDGE_P
    mat = np.triu(mat, 1)
    _ADJ = mat + mat.T
    _ADJ = _ADJ.astype(int)
    _INSTANCE_ID = f"graph_rand{NODE_N}_p{EDGE_P:.2f}_seed{_SEED}"
    os.makedirs("data/graph_coloring", exist_ok=True)
    np.savetxt(f"data/graph_coloring/{_INSTANCE_ID}.csv", _ADJ, delimiter=",", fmt="%d")

# -------------------------- FITNESS / REPAIR ---------------------------- #
def fitness(colors: List[int]) -> float:
    """
    评价函数：计算冲突边数（越少越好）。
    如果需要也可以加上颜色数惩罚，例如：+ 0.01 * max(colors)+1
    """
    arr = np.asarray(colors, dtype=int)
    if arr.shape[0] != NODE_N:
        raise ValueError(f"颜色列表长度应为 {NODE_N}，但收到 {len(colors)}")
    # 检查每条边是否冲突
    conflicts = (_ADJ * (arr[:, None] == arr[None, :])).sum() // 2
    return float(conflicts)

def repair(colors):
    """
    修复函数：保证每个节点都有一个合法颜色（0 到 NODE_N-1）。
    多余或非法值会被丢弃并随机补齐。
    """
    if isinstance(colors, str):
        try:
            colors = ast.literal_eval(colors)
        except Exception:
            raise ValueError(f"颜色字符串无法解析: {colors}")
    if not isinstance(colors, (list, tuple)):
        raise TypeError("colors must be list[int]")
    fixed = []
    seen_len = 0
    for c in colors:
        c_int = int(c)
        if 0 <= c_int < NODE_N:
            fixed.append(c_int)
        if len(fixed) >= NODE_N:
            break
    # 补齐：随机选择合法颜色
    import random
    while len(fixed) < NODE_N:
        fixed.append(random.randrange(0, NODE_N))
    return fixed

def generate_random_genome(rng=None) -> List[int]:
    """
    随机生成一个着色方案：每个节点随机指定一个颜色 [0, NODE_N-1]
    """
    if rng is None:
        rng = np.random.RandomState()
    return [int(c) for c in rng.randint(0, NODE_N, size=NODE_N)]

# ------------------------- CONTEXT FOR LLM ------------------------------ #
def get_ques() -> dict:
    """Return instance data to feed LLM."""
    return {"adjacency_matrix": _ADJ.tolist()}

def describe() -> str:
    return (
        "The graph coloring problem is a classic NP-Hard optimization task: "
        f"given an undirected graph with {NODE_N} nodes (0 to {NODE_N-1}), "
        "assign a color to each node such that no two adjacent nodes share the same color. "
        "The objective is to minimize the number of conflicting edges (i.e., edges whose endpoints have the same color). "
        "The adjacency matrix of the graph is provided below (row i, column j = 1 if there's an edge, else 0):"
    )

def get_genome_desc() -> str:
    return f"A valid genome for the graph coloring task is a list of {NODE_N} integers, each in the range [0, {NODE_N-1}], representing the color assigned to each node."

def get_mutation_ops() -> str:
    return (
        "1. change_color - randomly pick a node and assign it a new random color\n"
        "2. swap_colors  - pick two nodes and swap their colors\n"
        "3. recolor_group - pick a small group of nodes and recolor them randomly"
    )

def get_mutation_ops_list() -> str:
    return """[
        {"type": "change_color", "node": "<NodeIndex>", "new_color": "<ColorIndex>"},
        {"type": "swap_colors",  "i": "<Node1>",      "j": "<Node2>"},
        {"type": "recolor_group", "nodes": [<n1>,<n2>,...]}
    ]"""

def crossover_guideline() -> str:
    return (
        "To crossover two parent colorings, for each node index i, choose the color from parent A with probability 0.5, "
        "otherwise take it from parent B. After combining, ensure valid range and optionally repair conflicts."
    )

