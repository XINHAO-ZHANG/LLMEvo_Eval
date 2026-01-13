# ================================================
#  ga_core/loop.py — α‑Evolve main loop
# ================================================

import time, random, json, uuid
from pathlib import Path
from typing import List, Callable, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from evolve.db import get_db, Genome
from evolve.executor import batch_eval
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
        # zero-shot相关
        self.zero_shot_best = None  # 最佳zero-shot分数
        self.zero_shot_score = None  # 整体平均zero-shot分数
        
    def record(self, best_score: float):
        self.best_curve.append(best_score)
        
    def dump(self):
        return self.__dict__


def _generate_single_child(db, task_mod, model_name, n_parent, rng, stats):
    """生成单个子代，包含LLM调用和评估，用于并行执行"""
    try:
        sampled_parents = db.sample(n_parent)
        prompt = task_mod.get_evolve_prompt(sampled_parents)
        
        resp = call_llm(
            prompt,
            model=model_name,
            max_tokens=4096,
            seed=rng.randint(0, 2**30)
        )
        
        parsed_resp = task_mod.parse_response(resp)  # 返回字典 {"genome": ...}
        raw_genome = parsed_resp["genome"]
        
        # 计算损失值
        # 检查eval函数的参数签名，适配不同任务的eval接口
        import inspect
        eval_sig = inspect.signature(task_mod.eval)
        eval_params = list(eval_sig.parameters.keys())
        
        if len(eval_params) == 1:
            # 只有一个参数的eval函数
            eval_result = task_mod.eval(raw_genome)
        elif 'task' in eval_params:
            # promptopt类型的任务，有task参数
            if hasattr(task_mod, 'CURRENT_TASK'):
                current_task = task_mod.CURRENT_TASK
            else:
                current_task = getattr(task_mod, 'DEFAULT_EVAL_TASK', 'sum')
            eval_result = task_mod.eval(raw_genome, task=current_task)
        elif 'split' in eval_params:
            # symboreg类型的任务，有split参数
            eval_result = task_mod.eval(raw_genome, split="train")
        else:
            # 其他多参数情况，使用默认调用
            eval_result = task_mod.eval(raw_genome)
        
        # 处理eval可能返回元组的情况(loss, extra_info)
        if isinstance(eval_result, tuple):
            loss = eval_result[0]
            extra_info = eval_result[1] if len(eval_result) > 1 else ""
            genome = Genome(genome=raw_genome, loss=loss, extra={"feedback": extra_info})
        else:
            loss = eval_result
            genome = Genome(genome=raw_genome, loss=loss, extra={})
        
        # 计算token使用量
        token_usage = resp.get("usage", {}).get("total_tokens", 0)
        
        # 记录父代信息
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
               max_workers: int | None = None) -> RunStats:

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
            
            # 使用带种子的prompt函数，适配不同任务的get_zero_shot_prompt签名
            def seeded_prompt_func(seed=None, temperature_index=None, call_index=None):
                import inspect
                zero_shot_sig = inspect.signature(task_mod.get_zero_shot_prompt)
                zero_shot_params = list(zero_shot_sig.parameters.keys())
                
                # 根据参数个数调用不同的get_zero_shot_prompt
                if 'task' in zero_shot_params:
                    # promptopt类型的任务，有task参数
                    if hasattr(task_mod, 'CURRENT_TASK'):
                        current_task = task_mod.CURRENT_TASK
                    else:
                        current_task = getattr(task_mod, 'DEFAULT_EVAL_TASK', 'sum')
                    return task_mod.get_zero_shot_prompt(
                        task=current_task,
                        seed=seed, 
                        temperature_index=temperature_index, 
                        call_index=call_index
                    )
                else:
                    # 其他任务，标准签名
                    return task_mod.get_zero_shot_prompt(
                        seed=seed, 
                        temperature_index=temperature_index, 
                        call_index=call_index
                    )
            
            # 评估zero-shot能力，使用实验种子确保可重现性
            zero_shot_results = evaluate_zero_shot(
                prompt_func=seeded_prompt_func,  # 修改：使用 prompt_func 而不是 prompt
                model=model_name,
                task=task_mod,
                trials_per_temp=2,
                temp_step=0.2,
                base_seed=seed
            )
            print(f"Zero-shot evaluation completed with {len(zero_shot_results)} temperatures.")
        
        # 计算整体zero-shot能力分数（所有temperature下mean的平均值）
        means = [metrics['mean'] for metrics in zero_shot_results.values() if not np.isnan(metrics['mean'])]
        if means:
            zero_shot_score = np.mean(means)
        else:
            zero_shot_score = float('nan')
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
    print(f"Initial population size: {len(pop0)}, Best score: {best0}")
    stats.record(best0)
    seed_log = {"uuid": f"{model_name}_{task_mod.__name__}_{seed}_np{n_parent}_nc{n_child}_b{budget_calls}","gen": 0, "best_so_far": best0, "children":[],"child_scores": [0], "population": pop0}
    if log_callback: log_callback(seed_log)
    (out_dir / "gen_log.jsonl").open("a").write(json.dumps(seed_log)+"")

    # 3) Main loop ---------------------------------------------------------
    gen_idx = 0
    while stats.calls < budget_calls:
        gen_idx += 1
        
        # ====== 并行子代生成 ======
        print(f"Generation {gen_idx}: Starting parallel generation of {n_child} children...")
        
        child_genomes = []
        child_lineage = []
        
        # 使用线程池并行生成子代
        max_workers = min(n_child, 4)  # 限制并发数避免API限流
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有子代生成任务
            future_to_index = {}
            for i in range(n_child):
                future = executor.submit(_generate_single_child, db, task_mod, model_name, n_parent, rng, stats)
                future_to_index[future] = i
            
            # 收集结果
            for future in as_completed(future_to_index):
                result = future.result()
                
                if result["success"]:
                    child_genomes.append(result["genome"])
                    child_lineage.append(result["parent_lineage"])
                    stats.tok += result["token_usage"]
                    stats.calls += 1
                else:
                    # 处理失败的情况
                    print(f"Child generation failed: {result.get('error', 'Unknown error')}")
                    stats.calls += 1
        
        print(f"Generated {len(child_genomes)} children successfully.")
        
        # 可选修复（如果支持的话，也可以并行化）
        if hasattr(task_mod, "repair") and child_genomes:
            print("Applying repair function...")
            raw_genomes = [g.genome for g in child_genomes]
            repaired_genomes = task_mod.repair(raw_genomes)
            
            # 并行重新评估修复后的基因组
            def _eval_repaired_genome(genome):
                # 检查eval函数的参数签名，适配不同任务的eval接口
                import inspect
                eval_sig = inspect.signature(task_mod.eval)
                eval_params = list(eval_sig.parameters.keys())

                if len(eval_params) == 1:
                    # 只有一个参数的eval函数
                    eval_result = task_mod.eval(genome)
                elif 'task' in eval_params:
                    # promptopt类型的任务，有task参数
                    if hasattr(task_mod, 'CURRENT_TASK'):
                        current_task = task_mod.CURRENT_TASK
                    else:
                        current_task = getattr(task_mod, 'DEFAULT_EVAL_TASK', 'sum')
                    eval_result = task_mod.eval(genome, task=current_task)
                elif 'split' in eval_params:
                    # symboreg类型的任务，有split参数
                    eval_result = task_mod.eval(genome, split="train")
                else:
                    # 其他多参数情况，使用默认调用
                    eval_result = task_mod.eval(genome)
                
                if isinstance(eval_result, tuple):
                    loss = eval_result[0]
                    extra_info = eval_result[1] if len(eval_result) > 1 else ""
                    return Genome(genome=genome, loss=loss, extra={"feedback": extra_info})
                else:
                    loss = eval_result
                    return Genome(genome=genome, loss=loss, extra={})
            
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