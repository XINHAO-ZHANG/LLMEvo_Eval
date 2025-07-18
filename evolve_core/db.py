# ================================================
#  evovle_core/db.py — evolutionary database (Simple / MAP‑Elites ‑lite)
# ================================================
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Callable
import json, random
import numpy as np
from pathlib import Path

@dataclass
class Genome:
    genome: Any
    loss: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def _to_tuple(self, x):
        if isinstance(x, (list, tuple)):
            return tuple(self._to_tuple(i) for i in x)
        elif hasattr(x, 'tolist'):  # 兼容numpy array
            return tuple(self._to_tuple(i) for i in x.tolist())
        else:
            return x

    def __hash__(self):
        # 只用genome字段参与hash，loss和extra不参与
        return hash(self._to_tuple(self.genome))

    def __eq__(self, other):
        if not isinstance(other, Genome):
            return False
        return self._to_tuple(self.genome) == self._to_tuple(other.genome)

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
        self.pool: List[Genome] = []
        self.best_score = float("inf")
        self.genome_hashes = set()  # 用于检查唯一性

    # ---------- lifecycle ----------
    def init(self, n_init: int, rng: random.Random | None = None) -> None:
        rng = rng or self.rng
        self.pool = []
        self.genome_hashes = set()  # 重置哈希集
        
        # 使用seed_pool生成初始基因组并添加到池中
        for g in self.task.seed_pool(n_init, rng):
            if g in self.genome_hashes:
                continue
            self.pool.append(g)
            self.genome_hashes.add(g)
            
        if self.capacity and len(self.pool) > self.capacity:
            self.pool = self.pool[: self.capacity]
            # 重建哈希集
            self.genome_hashes = set(g for g in self.pool)
        
        # 更新最佳分数
        if self.pool:
            self.best_score = min(g.loss for g in self.pool)

    # ---------- public API ----------

    def _hash_genome(self, genome: Any) -> int:
        """生成基因组的哈希值，用于检查唯一性"""
        return hash(tuple(genome) if hasattr(genome, "__iter__") else genome)
        
    def sample(self, k: int, top_frac: float = 0.2) -> List[Genome]:
        # 先选出fitness最小的前N名，再从中随机采样k个
        n_top = max(1, int(len(self.pool) * top_frac))
        sorted_pool = sorted(self.pool, key=lambda g: g.loss)
        top_pool = sorted_pool[:n_top]
        # 计算权重（概率），权重与损失成反比
        weights = np.array([1.0 / (g.loss + 1e-8) for g in top_pool])
        # 归一化为概率
        probs = weights / weights.sum()
        idx = np.random.choice(len(top_pool), size=min(k, len(top_pool)), p=probs, replace=False)
        return [top_pool[i] for i in idx]

    def add(self, genomes: List[Genome]):
        for g in genomes:
            if g in self.genome_hashes:
                continue
            self.pool.append(g)
            self.genome_hashes.add(g)
            self.best_score = min(self.best_score, g.loss)
        # 如果超过容量，则按 fitness 升序（假设越小越好）截断
        if self.capacity and len(self.pool) > self.capacity:
            self.pool.sort(key=lambda g: g.loss)
            self.pool = self.pool[:self.capacity]
            self.genome_hashes = set(g for g in self.pool)
            self.best_score = self.pool[0].loss

    def get_best(self):
        return self.best_score

    # ---------- persistence ----------
    def to_json(self, path: Path):
        path.write_text(json.dumps([asdict(g) for g in self.pool]))

    def from_json(self, path: Path):
        # 读取为Genome对象
        raw = json.loads(path.read_text())
        self.pool = [Genome(**g) for g in raw]
        self.genome_hashes = set()
        for g in self.pool:
            self.genome_hashes.add(g)
        # 更新最佳分数
        if self.pool:
            self.best_score = min(g.loss for g in self.pool)


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
        self.key_fn = key_fn
        self.buckets: Dict[Any, _Bucket] = {}
        self.key_fn = key_fn
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
            if hasattr(self.task, "diversity_key"):
                k = self.task.diversity_key(g)
            else:
                k = hash(tuple(g))
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
