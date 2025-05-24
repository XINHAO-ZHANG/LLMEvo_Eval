
from __future__ import annotations

import random
import numpy as np

from tasks.utils import build_distance_matrix, random_points



# ------------------------------ CONFIG ---------------------------------- #
CITY_NUM = 30
 # fixed map for fair comp

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
def get_ques() -> dict:
    """Return instance data to feed LLM (may be huge!)."""
    return {"distance_matrix": DIST.tolist()}

def describe() -> str:
    return (
        "The traveling salesman problem (TSP) is a classic optimization problem that aims to find the shortest possible route that visits a set of cities, with each city being visited exactly once and the route returning to the original city."
        f"You must find the shortest path that visits all {CITY_NUM} cities. "
        "The distances between each pair of cities are provided below.The distance matrix is as follows (row i column j means distance from city i to city j):"
    )

def get_genome_desc() -> str:
    return f"A valid genome for the TSP task is a list of {CITY_NUM} unique integers from 0 to {CITY_NUM-1}."

# def get_mutation_ops() -> str:
#     return (
#     "1. swap - to exchange the positions of two elements" 
#     "2. relocate - to take out an element and insert it into another position" 
#     "3. reverse - to reverse a subsequence"
#     )

# def get_mutation_ops_list() -> str:
#         return """[
#             {"type": "swap", "i": "<Index1>", "j": "<Index2>"},
#             {"type": "relocate", "idx_from": "<IndexOrig>", "idx_to": "<IndexDest>"},
#             {"type": "reverse", "start": "<IndexStart>", "end": "<IndexEnd>"},
#           ]"""


def crossover_guiline() -> str:
    return "The crossover operator for the TSP is a simple one: two parent routes are combined to create a new child route. The child route is constructed by alternating the cities of the two parent routes, with a random number of cities from each parent included in the child. The resulting child route is then checked for validity and returned."

