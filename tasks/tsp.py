
from __future__ import annotations

import os
from textwrap import dedent
import json
import random
import numpy as np

from tasks.utils import build_distance_matrix, random_points



# ------------------------------ CONFIG ---------------------------------- #
CITY_NUM = 30
 # fixed map for fair comp
SYS_PROMPT = """You are an optimization expert helping to solve a hard problem. You will be shown several candidate solutions with their scores. Your goal is to propose better solutions (lower score is better). """

MATRIX_PATH = ""
if MATRIX_PATH:
    DIST = np.loadtxt(MATRIX_PATH, delimiter=",")
    CITY_NUM = DIST.shape[0]
    if DIST.shape[0] != DIST.shape[1]:
        raise ValueError("Distance matrix must be square")
    
else:
    coords = random_points(CITY_NUM, 42)
    DIST = build_distance_matrix(coords)
    _INSTANCE_ID = f"tsp_dismat_{CITY_NUM}"
    # save 
    os.makedirs("data/tsp", exist_ok=True)
    np.savetxt(f"data/tsp/{_INSTANCE_ID}.csv", DIST, delimiter="," ,fmt="%d")

# ---------- genome helpers ----------
def seed_pool(n: int, rng: random.Random):
    return [rng.sample(range(CITY_NUM), CITY_NUM) for _ in range(n)]

def fitness(path):
    arr = np.asarray(path, dtype=int)
    return float(DIST[arr[:-1], arr[1:]].sum() + DIST[arr[-1], arr[0]])

def repair(paths):
    # ensure each path is a permutation
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

def diversity_key(path):
    return tuple(path[:3])  # rough hash of first 3 cities

# ------------------------- CONTEXT FOR LLM ------------------------------ #
def parse_response(resp: str) -> str:
    content = resp["text"]
    try:
        data = json.loads(content)
    except Exception:
        # 尝试截取 ```json ... ``` 
        import re, textwrap
        m = re.search(
    r'(?:```json(.*?)```|<think>\s*</think>\s*({.*?}))',
    content,
    re.S)
        if m:
            try:
                data = json.loads(m.group(1) or m.group(2))
            except Exception:
                print(f"JSON parsing failed: {m.group(1) or m.group(2)}")
                data = {"text": content}
        else:
            print(f"JSON parsing failed: {content}")
            data = {"text": content}
    data["usage"] = resp["usage"] or {"total_tokens": 0}
    return data

def get_zero_shot_prompt():
    sys = SYS_PROMPT
    ques_block = json.dumps({"distance_matrix": DIST.tolist()}, ensure_ascii=False, indent=2)
    user = dedent(
        f"""
        TASK DESC    : {describe()}
        QUESTION    : {ques_block}
        Please return the optimal solution as JSON without any extra text: {{ "genome": "<full-new>" }}.
        ATTENTION: The genome should be a list of {CITY_NUM} unique integers from 0 to {CITY_NUM-1}.
        """
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


def get_evolve_prompt(sampled_parents: list[list[int]]):
    parents, scores = zip(*[(g.genome, g.fitness) for g in sampled_parents])
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
