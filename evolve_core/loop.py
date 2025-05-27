# ================================================
#  ga_core/loop.py — α‑Evolve main loop
# ================================================

import time, random, json, uuid
from pathlib import Path
from typing import List, Callable, Any

from evolve_core.db import get_db, Genome
from evolve_core.executor import batch_eval
from llm_ops.api import call_llm
from llm_ops.prompt import build_evolve_prompt   # adjust if located elsewhere


LogCallback = Callable[[dict[str, Any]], None]

class RunStats:
    def __init__(self, task_name: str, model: str, seed: int):
        self.task = task_name; self.model = model; self.seed = seed
        self.calls = 0; self.tok = 0; self.best_curve: List[float] = []
        self.wall = 0.0
    def record(self, best_score: float):
        self.best_curve.append(best_score)
    def dump(self):
        return self.__dict__

def _snapshot_population(db):
    """Return current population as list[{genome, score}] (after truncation)."""
    if hasattr(db, "pool"):                 # SimplePoolDB
        return [{"genome": g, "score": s} for g, s in db.pool]
    else:                                     # MAP‑Elites
        flat = []
        for b in db.buckets.values():
            if b.best:
                s, g = b.best
                flat.append({"genome": g, "score": s})
        return flat


def run_evolve(task_mod,
               model_name: str,
               *,
               seed: int = 0,
               n_init: int = 40,
               n_parent: int = 4,
               n_child: int = 8,
               budget_calls: int = 100,
               init_pop_path: str | None = None,
               db_mode: str = "simple",
               db_kwargs: dict | None = None,
               out_dir: Path | None = None,
               log_callback: LogCallback | None = None) -> RunStats:

    rng = random.Random(seed)
    out_dir = out_dir or Path(f"runs/{model_name}_{task_mod.__name__}_{seed}_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) population DB ----------------------------------------------------------------
    db = get_db(db_mode, task_mod, **(db_kwargs or {}))
    if init_pop_path and Path(init_pop_path).exists():
       db.from_json(Path(init_pop_path))
       # 统计实际初始种群数量
       if hasattr(db, "pool"):
           n_init_actual = len(db.pool)
       elif hasattr(db, "buckets"):
           n_init_actual = sum(1 for b in db.buckets.values() if b.best)
       print(f"Loaded initial population from {init_pop_path}, size={n_init_actual}")
    else:
       db.init(n_init, rng)
       n_init_actual = n_init
    # 2) Stats -------------------------------------------------------------
    stats = RunStats(task_mod.__name__, model_name, seed); t0 = time.time()
    
    # -------- log generation 0 (seed only) --------
    pop0 = _snapshot_population(db)
    best0 = db.get_best()
    stats.record(best0)
    seed_log = {"uuid": f"{model_name}_{task_mod.__name__}_{seed}_np{n_parent}_nc{n_child}_b{budget_calls}","gen": 0, "best_so_far": best0, "children":[],"child_scores": [0], "population": pop0}
    if log_callback: log_callback(seed_log)
    (out_dir / "gen_log.jsonl").open("a").write(json.dumps(seed_log)+"")
    # 3) Main loop ---------------------------------------------------------
    gen_idx = 0
    while stats.calls < budget_calls:
        gen_idx += 1
        # 移除之前的单次采样
        # parents, scores = db.sample(n_parent)
        
        # 对于每个子代单独调用 LLM
        child_genomes, child_scores = [], []
        for _ in range(n_child):
            # 为每个子代单独采样父代
            parents, scores = db.sample(n_parent)
            prompt = build_evolve_prompt(task_mod, parents, scores)

            try:
                resp = call_llm(
                prompt,
                model=model_name,
                max_tokens=4096,
                seed=rng.randint(0, 2**30)
            )
                stats.calls += 1
            except Exception as e:
                print(f"Error: {e}")
                stats.calls += 1
                continue
            
            stats.tok += resp.get("usage", {}).get("total_tokens", 0)
            child_genome = resp.get("genome", [])
            child_genomes.append(child_genome)
        child_genomes = task_mod.repair(child_genomes)
        child_scores = batch_eval(task_mod, child_genomes)
        db.add(child_genomes, child_scores)

        # ---------- generation‑level metrics ----------
        if hasattr(db, "get_best"):
            best = db.get_best()
        else:
            best = min(s for _, s in db.pool) if hasattr(db, "pool") else \
                   min(b.best[0] for b in db.buckets.values() if b.best)
        stats.record(best)

        population = _snapshot_population(db)   # after truncation
        # 生成日志
        log = {
                "uuid": f"{model_name}_{task_mod.__name__}_{seed}_np{n_parent}_nc{n_child}_b{budget_calls}",
                "gen": gen_idx,
                "children": child_genomes,
                "child_scores": child_scores,
                "population": population,
                "best_so_far": best,
            }
        if log_callback:
            log_callback(log)
            (out_dir / "gen_log.jsonl").open("a").write(json.dumps(log) + "\n")

        # checkpoint
        if stats.calls % 20 == 0:
            db.to_json(out_dir / "pop.json")
            (out_dir / "stats.json").write_text(json.dumps(stats.dump(), indent=2))

    # 4) 结束后写 final
    stats.wall = time.time() - t0
    db.to_json(out_dir / "pop.json")
    (out_dir / "stats.json").write_text(json.dumps(stats.dump(), indent=2))
    return stats