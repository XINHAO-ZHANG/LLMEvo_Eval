# ================================================
#  evovle_core/db.py — evolutionary database (Simple / MAP‑Elites ‑lite)
# ================================================
from __future__ import annotations

import json, random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable

Genome = Any

class SimplePoolDB:
    """
    A minimal random‑sampling population store.
    No explicit selection pressure: parents are drawn
    uniformly; children simply appended (FIFO truncation
    when capacity is reached).
    """
    def __init__(self,
                 task_mod,
                 capacity: int | None = None,
                 rng: random.Random | None = None) -> None:
        self.task = task_mod
        self.capacity = capacity
        self.rng = rng or random.Random()
        self.pool: List[Tuple[Genome, float]] = []
        self.best_score = float("inf")

    # ---------- lifecycle ----------
    def init(self, n_init: int, rng: random.Random | None = None) -> None:
        rng = rng or self.rng
        self.pool = [(g, self.task.fitness(g))
                     for g in self.task.seed_pool(n_init, rng)]
        if self.capacity:
            self.pool = self.pool[: self.capacity]
        

    # ---------- public API ----------
    def sample(self, k: int) -> Tuple[List[Genome], List[float]]:
        idx = self.rng.sample(range(len(self.pool)), k=min(k, len(self.pool)))
        parents, scores = zip(*[self.pool[i] for i in idx])
        return list(parents), list(scores)

    def add(self, genomes: List[Genome], scores: List[float]):
        for g, s in zip(genomes, scores):
            self.pool.append((g, s))
            self.best_score = min(self.best_score, s)
        # 如果超过容量，则按 fitness 升序（假设越小越好）截断
        if self.capacity and len(self.pool) > self.capacity:
            # pool 是 List[ (genome, fitness) ]
            self.pool.sort(key=lambda gs: gs[1])
            self.pool = self.pool[: self.capacity]
            self.best_score = self.pool[0][1]

    def get_best(self):
        return self.best_score

    # ---------- persistence ----------
    def to_json(self, path: Path):
        path.write_text(json.dumps(self.pool))

    def from_json(self, path: Path):
        self.pool = [tuple(p) for p in json.loads(path.read_text())]


class _Bucket:
    """Internal: top‑1 bucket used by MAP‑Elites."""
    def __init__(self):
        self.best: Tuple[float, Genome] | None = None

    def maybe_add(self, g: Genome, s: float):
        if self.best is None or s < self.best[0]:
            self.best = (s, g)


class MapElitesDB:
    """
    Very light MAP‑Elites table: each bucket keeps only
    the single best genome observed so far.
    """
    def __init__(self,
                 task_mod,
                 capacity: int = 2048,
                 key_fn: Callable[[Genome], Any] | None = None,
                 rng: random.Random | None = None) -> None:
        self.task = task_mod
        self.capacity = capacity
        self.key_fn = key_fn or (lambda g: hash(tuple(g)))
        self.buckets: Dict[Any, _Bucket] = {}
        self.rng = rng or random.Random()

    # ---------- lifecycle ----------
    def init(self, n_init: int, rng: random.Random | None = None):
        rng = rng or self.rng
        for g in self.task.seed_pool(n_init, rng):
            self.add([g], [self.task.fitness(g)])

    # ---------- public API ----------
    def sample(self, k: int) -> Tuple[List[Genome], List[float]]:
        pool = [b.best for b in self.buckets.values() if b.best]
        sel = self.rng.sample(pool, k=min(k, len(pool))) if pool else []
        if sel:
            genomes, scores = zip(*sel)
            return list(genomes), list(scores)
        return [], []

    def add(self, genomes: List[Genome], scores: List[float]):
        for g, s in zip(genomes, scores):
            k = self.key_fn(g)
            bucket = self.buckets.setdefault(k, _Bucket())
            bucket.maybe_add(g, s)
            if len(self.buckets) > self.capacity:
                self.buckets.pop(self.rng.choice(list(self.buckets)))

    # ---------- persistence ----------
    def to_json(self, path: Path):
        flat = {str(k): v.best for k, v in self.buckets.items() if v.best}
        path.write_text(json.dumps(flat))

    @classmethod
    def from_json(cls, task_mod, path: Path, **kwargs):
        db = cls(task_mod, **kwargs)
        raw = json.loads(path.read_text())
        for k, (s, g) in raw.items():
            db.buckets[k] = _Bucket(); db.buckets[k].best = (s, g)
        return db


# ------------------------------------------------------------------
# Factory for dynamic DB selection
# ------------------------------------------------------------------
DB_REGISTRY: Dict[str, type] = {
    "simple": SimplePoolDB,
    "map":    MapElitesDB,
}

def get_db(mode: str, task_mod, **kwargs):
    try:
        cls = DB_REGISTRY[mode.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown db mode '{mode}'. "
                         f"Available: {list(DB_REGISTRY)}") from e
    return cls(task_mod, **kwargs)
