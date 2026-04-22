# ================================================
#  ga_core/loop.py — α‑Evolve main loop
# ================================================

import time, random, json, uuid
from pathlib import Path
from typing import List, Callable, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from evolve.db import get_db, Genome
from llm.api import call_llm
from llm.prompts import build_evolve_prompt
from llm.zero_shot_eval import evaluate_zero_shot


LogCallback = Callable[[dict[str, Any]], None]

class RunStats:
    def __init__(self, task_name: str, model: str, seed: int):
        self.task = task_name
        self.model = model
        self.seed = seed
        self.calls = 0
        self.tok = 0
        self.best_curve: List[float] = []
        self.wall = 0.0
        self.zero_shot_best = None   # best zero-shot score across temperatures
        self.zero_shot_score = None  # mean zero-shot score across temperatures

    def record(self, best_score: float):
        self.best_curve.append(best_score)

    def dump(self):
        return self.__dict__


def _adapt_eval(task_mod, genome):
    """Call task_mod.eval with the correct signature, adapting to different task interfaces."""
    import inspect
    params = list(inspect.signature(task_mod.eval).parameters.keys())

    if len(params) == 1:
        return task_mod.eval(genome)
    elif 'task' in params:
        # promptopt: requires task parameter
        current_task = getattr(task_mod, 'CURRENT_TASK',
                               getattr(task_mod, 'DEFAULT_EVAL_TASK', 'sum'))
        return task_mod.eval(genome, task=current_task)
    elif 'split' in params:
        # symboreg: requires split parameter
        return task_mod.eval(genome, split="train")
    else:
        return task_mod.eval(genome)


def _generate_single_child(db, task_mod, model_name, n_parent, rng, stats, temperature: float = 0.7):
    """Generate a single offspring (LLM call + eval) for parallel execution."""
    try:
        sampled_parents = db.sample(n_parent)
        prompt = task_mod.get_evolve_prompt(sampled_parents)

        resp = call_llm(
            prompt,
            model=model_name,
            max_tokens=4096,
            temperature=temperature,
            seed=rng.randint(0, 2**30)
        )

        parsed_resp = task_mod.parse_response(resp)
        raw_genome = parsed_resp["genome"]

        eval_result = _adapt_eval(task_mod, raw_genome)

        if isinstance(eval_result, tuple):
            loss, extra_info = eval_result[0], (eval_result[1] if len(eval_result) > 1 else "")
            genome = Genome(genome=raw_genome, loss=loss, extra={"feedback": extra_info})
        else:
            genome = Genome(genome=raw_genome, loss=eval_result, extra={})

        token_usage = resp.get("usage", {}).get("total_tokens", 0)
        parent_lineage = [p.genome for p in sampled_parents]

        return {
            "success": True,
            "genome": genome,
            "token_usage": token_usage,
            "parent_lineage": parent_lineage
        }

    except Exception as e:
        print(f"Error generating child: {e}")
        return {
            "success": False,
            "error": str(e),
            "token_usage": 0,
            "parent_lineage": []
        }


def _snapshot_population(db):
    """Return current population as list[{genome, score}] (after truncation)."""
    if hasattr(db, "pool"):  # SimplePoolDB
        return [{"genome": g.genome, "score": g.loss} for g in db.pool]
    else:  # MAP‑Elites
        flat = []
        for b in db.buckets.values():
            if b.best:
                s, g = b.best
                flat.append({"genome": g, "score": s})
        return flat


def run_evolve(cfg,
               task_mod,
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
               log_callback: LogCallback | None = None,
               max_workers: int | None = None,
               temperature: float = 0.7) -> RunStats:

    rng = random.Random(seed)
    out_dir = out_dir or Path(f"runs/{model_name}_{task_mod.__name__}_{seed}_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) population DB ----------------------------------------------------------------
    db = get_db(db_mode, task_mod, **(db_kwargs or {}))
    if init_pop_path and Path(init_pop_path).exists():
        db.from_json(Path(init_pop_path))
        if hasattr(db, "pool"):
            n_init_actual = len(db.pool)
        elif hasattr(db, "buckets"):
            n_init_actual = sum(1 for b in db.buckets.values() if b.best)
        print(f"Loaded initial population from {init_pop_path}, size={n_init_actual}")
    else:
        db.init(n_init, rng)
        n_init_actual = n_init

    # 2) Zero-shot Evaluation --------------------------------------------------------
    enable_zero_shot = cfg.get("enable_zero_shot_eval", True)
    zero_shot_score = None
    zero_shot_best = None

    if enable_zero_shot:
        print(f"\n{'='*20} Zero-shot Evaluation {'='*20}")

        import inspect
        zero_shot_sig = inspect.signature(task_mod.get_zero_shot_prompt)
        zero_shot_params = list(zero_shot_sig.parameters.keys())

        def seeded_prompt_func(seed=None, temperature_index=None, call_index=None):
            if 'task' in zero_shot_params:
                # promptopt: has task parameter
                current_task = getattr(task_mod, 'CURRENT_TASK',
                                       getattr(task_mod, 'DEFAULT_EVAL_TASK', 'sum'))
                return task_mod.get_zero_shot_prompt(
                    task=current_task,
                    seed=seed,
                    temperature_index=temperature_index,
                    call_index=call_index
                )
            else:
                return task_mod.get_zero_shot_prompt(
                    seed=seed,
                    temperature_index=temperature_index,
                    call_index=call_index
                )

        zero_shot_results = evaluate_zero_shot(
            prompt_func=seeded_prompt_func,
            model=model_name,
            task=task_mod,
            trials_per_temp=2,
            temp_step=0.2,
            base_seed=seed
        )
        print(f"Zero-shot evaluation completed with {len(zero_shot_results)} temperatures.")

        means = [metrics['mean'] for metrics in zero_shot_results.values() if not np.isnan(metrics['mean'])]
        zero_shot_score = np.mean(means) if means else float('nan')
        zero_shot_best_temp = min(zero_shot_results.items(), key=lambda x: x[1]['mean'])[0]
        zero_shot_best = zero_shot_results[zero_shot_best_temp]['mean']

        print("\nZero-shot performance across temperatures:")
        print("-" * 50)
        print(f"{'Temp':>6} {'Mean':>10}")
        print("-" * 50)
        for temp, metrics in sorted(zero_shot_results.items()):
            print(f"{temp:6.1f} {metrics['mean']:10.2f}")
        print("-" * 50)
        print(f"\nOverall Zero-shot Score: {zero_shot_score:.2f}")
        print(f"{'='*20} Evaluation End {'='*20}\n")

        zero_shot_path = out_dir / "zero_shot_eval.json"
        zero_shot_path.write_text(
            json.dumps({
                "model": model_name,
                "task": task_mod.__name__,
                "seed": seed,
                "results": zero_shot_results,
                "zero_shot_score": float(zero_shot_score),
                "zero_shot_best": float(zero_shot_best),
                "timestamp": time.time()
            }, indent=2)
        )
    else:
        print(f"\n{'='*20} Skipping Zero-shot Evaluation {'='*20}\n")

    # 3) Stats -------------------------------------------------------------
    stats = RunStats(task_mod.__name__, model_name, seed)
    stats.zero_shot_score = float(zero_shot_score) if zero_shot_score is not None else None
    stats.zero_shot_best = float(zero_shot_best) if zero_shot_best is not None else None
    t0 = time.time()

    # -------- log generation 0 (seed only) --------
    pop0 = _snapshot_population(db)
    best0 = db.get_best()
    print(f"Initial population size: {len(pop0)}, Best score: {best0}")
    stats.record(best0)
    seed_log = {
        "uuid": f"{model_name}_{task_mod.__name__}_{seed}_np{n_parent}_nc{n_child}_b{budget_calls}",
        "gen": 0,
        "best_so_far": best0,
        "children": [],
        "child_scores": [0],
        "population": pop0
    }
    if log_callback: log_callback(seed_log)
    (out_dir / "gen_log.jsonl").open("a").write(json.dumps(seed_log) + "")

    # 4) Main loop ---------------------------------------------------------
    gen_idx = 0
    while stats.calls < budget_calls:
        gen_idx += 1

        print(f"Generation {gen_idx}: Starting parallel generation of {n_child} children...")

        child_genomes = []
        child_lineage = []

        # cap concurrency to avoid rate-limiting
        max_workers = min(n_child, 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for i in range(n_child):
                future = executor.submit(
                    _generate_single_child,
                    db, task_mod, model_name, n_parent, rng, stats, temperature
                )
                future_to_index[future] = i

            for future in as_completed(future_to_index):
                result = future.result()
                if result["success"]:
                    child_genomes.append(result["genome"])
                    child_lineage.append(result["parent_lineage"])
                    stats.tok += result["token_usage"]
                    stats.calls += 1
                else:
                    print(f"Child generation failed: {result.get('error', 'Unknown error')}")
                    stats.calls += 1

        print(f"Generated {len(child_genomes)} children successfully.")

        # optional repair step
        if hasattr(task_mod, "repair") and child_genomes:
            print("Applying repair function...")
            raw_genomes = [g.genome for g in child_genomes]
            repaired_genomes = task_mod.repair(raw_genomes)

            def _eval_repaired_genome(genome):
                eval_result = _adapt_eval(task_mod, genome)
                if isinstance(eval_result, tuple):
                    loss = eval_result[0]
                    extra_info = eval_result[1] if len(eval_result) > 1 else ""
                    return Genome(genome=genome, loss=loss, extra={"feedback": extra_info})
                else:
                    return Genome(genome=genome, loss=eval_result, extra={})

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                child_genomes = list(executor.map(_eval_repaired_genome, repaired_genomes))

        db.add(child_genomes)

        # ---------- generation‑level metrics ----------
        if hasattr(db, "get_best"):
            best = db.get_best()
        else:
            best = min(s for _, s in db.pool) if hasattr(db, "pool") else \
                   min(b.best[0] for b in db.buckets.values() if b.best)
        stats.record(best)

        population = _snapshot_population(db)
        log = {
            "uuid": f"{model_name}_{task_mod.__name__}_{seed}_np{n_parent}_nc{n_child}_b{budget_calls}",
            "gen": gen_idx,
            "children": [g.genome for g in child_genomes],
            "child_scores": [g.loss for g in child_genomes],
            "population": population,
            "best_so_far": best,
            "parent_lineage": child_lineage,
        }
        if log_callback:
            log_callback(log)
            (out_dir / "gen_log.jsonl").open("a").write(json.dumps(log) + "\n")

        # checkpoint
        if stats.calls % 20 == 0:
            db.to_json(out_dir / "pop.json")
            (out_dir / "stats.json").write_text(json.dumps(stats.dump(), indent=2))

    # 5) finalize
    stats.wall = time.time() - t0
    db.to_json(out_dir / "pop.json")
    (out_dir / "stats.json").write_text(json.dumps(stats.dump(), indent=2))
    return stats
