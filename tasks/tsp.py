from __future__ import annotations

import os
from textwrap import dedent
import json
import random
import numpy as np
import re
import ast
from pathlib import Path

from tasks.utils import build_distance_matrix, random_points
from evolve_core.db import Genome


# ------------------------------ CONFIG ---------------------------------- #

# 获取项目根目录的绝对路径
PROJECT_ROOT = Path(__file__).parent.parent  # tasks目录的上一级
DATA_DIR = PROJECT_ROOT / "data" / "tsp"

CITY_NUM = 30  # 默认
DIST = None
SYS_PROMPT = """You are an optimization expert helping to solve a hard problem. You will be shown several candidate solutions with their scores. Your goal is to propose better solutions (lower score is better). """
def configure(cfg=None):
    global CITY_NUM, DIST
    if cfg and hasattr(cfg, "city_num"):
        CITY_NUM = int(cfg.city_num)
    print(f"TSP configured with CITY_NUM={CITY_NUM}")

    _INSTANCE_ID = f"tsp_dismat_{CITY_NUM}"
    DIST_PATH = DATA_DIR / f"{_INSTANCE_ID}.csv"

    if DIST_PATH.exists():
        DIST = np.loadtxt(DIST_PATH, delimiter=",")
        if DIST.shape[0] != CITY_NUM or DIST.shape[1] != CITY_NUM:
            print(f"Distance matrix shape mismatch, regenerating for CITY_NUM={CITY_NUM}")
            coords = random_points(CITY_NUM, 42)
            DIST = build_distance_matrix(coords)
            np.savetxt(DIST_PATH, DIST, delimiter=",", fmt="%d")
    else:
        coords = random_points(CITY_NUM, 42)
        DIST = build_distance_matrix(coords)
        DIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(DIST_PATH, DIST, delimiter=",", fmt="%d")

# ---------- genome helpers ----------
def seed_pool(n: int, rng: random.Random):
    genomes = [rng.sample(range(CITY_NUM), CITY_NUM) for _ in range(n)]
    return [Genome(genome=g, loss=eval(g), extra={}) for g in genomes]

def eval(path):
    arr = np.asarray(path, dtype=int)
    return float(DIST[arr[:-1], arr[1:]].sum() + DIST[arr[-1], arr[0]])


def repair(paths: list[list[int]]) -> list[list[int]]:
    fixed = []
    for p in paths:
        seen = set(); new = []
        for city in p:
            if city not in seen and 0 <= city < CITY_NUM:
                seen.add(city); new.append(city)
        missing = [c for c in range(CITY_NUM) if c not in seen]
        new.extend(missing)
        fixed.append(new[:CITY_NUM])
    return fixed

def diversity_key(g: Genome):
    return tuple(g.genome[:3]) # rough hash of first 3 cities

# ------------------------- CONTEXT FOR LLM ------------------------------ #
def parse_response(resp):
    content = resp.get("text", "")
    # 1. 先找json代码块
    m = re.search(r'```json(.*?)```', content, re.S)
    if m:
        try:
            data = json.loads(m.group(1))
            data["usage"] = resp.get("usage", {"total_tokens": 0})
            # genome为字符串时，尝试转为list
            if isinstance(data.get("genome"), str):
                try:
                    genome_eval = ast.literal_eval(data["genome"])
                    if isinstance(genome_eval, list):
                        data["genome"] = genome_eval
                except Exception:
                    pass
            return data
        except Exception:
            pass
    # 2. 直接找大括号
    m = re.search(r'({[^{}]*"genome"[^{}]*})', content, re.S)
    if m:
        try:
            data = json.loads(m.group(1))
            data["usage"] = resp.get("usage", {"total_tokens": 0})
            if isinstance(data.get("genome"), str):
                try:
                    genome_eval = ast.literal_eval(data["genome"])
                    if isinstance(genome_eval, list):
                        data["genome"] = genome_eval
                except Exception:
                    pass
            return data
        except Exception:
            try:
                data = ast.literal_eval(m.group(1))
                data["usage"] = resp.get("usage", {"total_tokens": 0})
                if isinstance(data.get("genome"), str):
                    try:
                        genome_eval = ast.literal_eval(data["genome"])
                        if isinstance(genome_eval, list):
                            data["genome"] = genome_eval
                    except Exception:
                        pass
                return data
            except Exception:
                pass
    # 3. 只找genome数组（字符串或列表）
    m = re.search(r'"genome"\s*:\s*(\[.*?\])', content, re.S)
    if m:
        arr_str = m.group(1).strip()
        try:
            genome = ast.literal_eval(arr_str)
            if isinstance(genome, list):
                return {"genome": genome, "usage": resp.get("usage", {"total_tokens": 0})}
        except Exception:
            pass
    # 4. 兜底：如果resp本身是json字符串
    try:
        data = json.loads(content)
        if isinstance(data.get("genome"), str):
            try:
                genome_eval = ast.literal_eval(data["genome"])
                if isinstance(genome_eval, list):
                    data["genome"] = genome_eval
            except Exception:
                pass
        data["usage"] = resp.get("usage", {"total_tokens": 0})
        return data
    except Exception:
        pass
    # 5. 兜底
    return {"genome": None, "text": content, "usage": resp.get("usage", {"total_tokens": 0})}

def get_zero_shot_prompt():
    sys = SYS_PROMPT
    ques_block = json.dumps({"distance_matrix": DIST.tolist()}, ensure_ascii=False, indent=2)
    user = dedent(
        f"""
        TASK DESC    : {describe()}
        QUESTION    : {ques_block}
        Please return the optimal solution as JSON without any extra explanation: {{ "genome": "<full-new>" }}.
        ATTENTION: The genome should be a list of {CITY_NUM} unique integers from 0 to {CITY_NUM-1}.
        """
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


def get_evolve_prompt(sampled_parents: list[list[int]]):
    parents, scores = zip(*[(g.genome, g.loss) for g in sampled_parents])
    sys = SYS_PROMPT
    parent_block = json.dumps(
        [{"genome":g, "score":s} for g,s in zip(parents,scores)],
        ensure_ascii=False, indent=2
    )
    ques_block = json.dumps({"distance_matrix": DIST.tolist()}, ensure_ascii=False, indent=2)
    user = dedent(
        f"""
        TASK DESC    : {describe()}
        QUESTION    : {ques_block}
        Here are {len(parents)} previous solutions and their scores:
        ```json
        {parent_block}
        ```
        Please return one BETTER child genome as JSON without any extra text: {{ "genome": "<full-new>" }}. 
        ATTENTION: The genome should be a list of {CITY_NUM} unique integers from 0 to {CITY_NUM-1}.
        """
    )
    
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]
    


def describe() -> str:
    return (
        "The traveling salesman problem (TSP) is a classic optimization problem that aims to find the shortest possible route that visits a set of cities, with each city being visited exactly once and the route returning to the original city."
        f"You must find the shortest path that visits all {CITY_NUM} cities. "
        "The distances between each pair of cities are provided below.The distance matrix is as follows (row i column j means distance from city i to city j):"
    )

# def get_genome_desc() -> str:
#     return f"A valid genome for the TSP task is a list of {CITY_NUM} unique integers from 0 to {CITY_NUM-1}."

# def crossover_guiline() -> str:
#     return "The crossover operator for the TSP is a simple one: two parent routes are combined to create a new child route. The child route is constructed by alternating the cities of the two parent routes, with a random number of cities from each parent included in the child. The resulting child route is then checked for validity and returned."

def create_fallback_genome():
    """创建一个简单的顺序路径作为fallback"""
    fallback_path = list(range(CITY_NUM))
    fallback_loss = eval(fallback_path)
    return fallback_path, fallback_loss
