# ================================================
#  ga_core/loop.py — α‑Evolve main loop
# ================================================

import time, random, json, uuid
from pathlib import Path
from typing import List, Callable, Any
import numpy as np

from evolve_core.db import get_db, Genome
from evolve_core.executor import batch_eval
from llm_ops.api import call_llm
from llm_ops.prompt import build_evolve_prompt   # adjust if located elsewhere
from llm_ops.zero_shot_eval import evaluate_zero_shot


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
        # zero-shot相关
        self.zero_shot_best = None  # 最佳zero-shot分数
        self.zero_shot_score = None  # 整体平均zero-shot分数
        
    def record(self, best_score: float):
        self.best_curve.append(best_score)
        
    def dump(self):
        return self.__dict__

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

    # 2) Zero-shot Evaluation --------------------------------------------------------
    enable_zero_shot = cfg.get("enable_zero_shot_eval", True)
    zero_shot_score = None
    zero_shot_best = None
    
    if enable_zero_shot:
        print(f"\n{'='*20} Zero-shot Evaluation {'='*20}")
        
        # 构造评估用的prompt        # 在 run_evolve 函数中
        if enable_zero_shot:
            
            # 使用带种子的prompt函数
            def seeded_prompt_func(seed=None, temperature_index=None, call_index=None):
                return task_mod.get_zero_shot_prompt(seed=seed, 
                                                   temperature_index=temperature_index, 
                                                   call_index=call_index)
            
            # 评估zero-shot能力，使用实验种子确保可重现性
            zero_shot_results = evaluate_zero_shot(
                prompt_func=seeded_prompt_func,  # 修改：使用 prompt_func 而不是 prompt
                model=model_name,
                task=task_mod,
                trials_per_temp=2,
                temp_step=0.2,
                base_seed=seed
            )
        eval_prompt = task_mod.get_zero_shot_prompt()  
        
        # 评估zero-shot能力
        zero_shot_results = evaluate_zero_shot(
            prompt=eval_prompt,
            model=model_name,
            task=task_mod,
            trials_per_temp=2,  # 每个temperature测试2次
            temp_step=0.2,      # temperature从0到1，步长0.2
        )
        
        # 计算整体zero-shot能力分数（所有temperature下mean的平均值）
        means = [metrics['mean'] for metrics in zero_shot_results.values() if not np.isnan(metrics['mean'])]
        if means:
            zero_shot_score = np.mean(means)
        else:
            zero_shot_score = float('nan')
        zero_shot_best = min(zero_shot_results.items(), key=lambda x: x[1]['mean'])[0]
        
        print("\nZero-shot performance across temperatures:")
        print("-" * 50)
        print(f"{'Temp':>6} {'Mean':>10}")
        print("-" * 50)
        for temp, metrics in sorted(zero_shot_results.items()):
            print(f"{temp:6.1f} {metrics['mean']:10.2f}")
        print("-" * 50)
        print(f"\nOverall Zero-shot Score: {zero_shot_score:.2f}")
        print(f"{'='*20} Evaluation End {'='*20}\n")
        
        # 保存评估结果
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
    stats.record(best0)
    seed_log = {"uuid": f"{model_name}_{task_mod.__name__}_{seed}_np{n_parent}_nc{n_child}_b{budget_calls}","gen": 0, "best_so_far": best0, "children":[],"child_scores": [0], "population": pop0}
    if log_callback: log_callback(seed_log)
    (out_dir / "gen_log.jsonl").open("a").write(json.dumps(seed_log)+"")

    # 3) Main loop ---------------------------------------------------------
    gen_idx = 0
    while stats.calls < budget_calls:
        gen_idx += 1
        # 对于每个子代单独调用 LLM
        child_genomes = []
        child_lineage = []  # 新增：记录每个子代的父代
        for _ in range(n_child):
            sampled_parents = db.sample(n_parent)
            prompt = task_mod.get_evolve_prompt(sampled_parents)
            try:
                resp = call_llm(
                    prompt,
                    model=model_name,
                    max_tokens=4096,
                    seed=rng.randint(0, 2**30)
                )
                parsed_resp = task_mod.parse_response(resp)  # 返回字典 {"genome": ...}
                raw_genome = parsed_resp["genome"]
                # 计算损失值
                eval_result = task_mod.eval(raw_genome)
                # 处理eval可能返回元组的情况(loss, extra_info)
                if isinstance(eval_result, tuple):
                    loss = eval_result[0]
                    extra_info = eval_result[1] if len(eval_result) > 1 else ""
                    genome = Genome(genome=raw_genome, loss=loss, extra={"feedback": extra_info})
                else:
                    loss = eval_result
                    genome = Genome(genome=raw_genome, loss=loss, extra={})
                stats.calls += 1
            except Exception as e:
                print(f"Error: {e}")
                stats.calls += 1
                continue
            stats.tok += resp.get("usage", {}).get("total_tokens", 0)
            child_genomes.append(genome)
            # 记录父代
            child_lineage.append([p.genome for p in sampled_parents])
        # 可选修复
        if hasattr(task_mod, "repair"):
            raw_genomes = [g.genome for g in child_genomes]
            repaired_genomes = task_mod.repair(raw_genomes)
            # 重新创建Genome对象
            new_child_genomes = []
            for g in repaired_genomes:
                eval_result = task_mod.eval(g)
                if isinstance(eval_result, tuple):
                    loss = eval_result[0]
                    extra_info = eval_result[1] if len(eval_result) > 1 else ""
                    new_child_genomes.append(Genome(genome=g, loss=loss, extra={"feedback": extra_info}))
                else:
                    loss = eval_result
                    new_child_genomes.append(Genome(genome=g, loss=loss, extra={}))
            child_genomes = new_child_genomes
        
        db.add(child_genomes)

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

    # 4) 结束后写 final
    stats.wall = time.time() - t0
    db.to_json(out_dir / "pop.json")
    (out_dir / "stats.json").write_text(json.dumps(stats.dump(), indent=2))
    return stats