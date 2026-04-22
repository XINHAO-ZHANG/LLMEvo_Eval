#!/usr/bin/env python
"""CLI to launch an EvoEval run or sweep.

   Examples
   --------
   python scripts/run_exp.py --task tsp --model gpt-4o --db_mode map --budget 300
"""
from __future__ import annotations

import time
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from evolve import run_evolve
from scripts.wandb_logger import make_wandb_callback
import importlib
import sys

TASK_PKG = "tasks"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

########################################################################
# Helpers
########################################################################

def load_task(task_name: str):
    try:
        return importlib.import_module(f"{TASK_PKG}.{task_name}")
    except ModuleNotFoundError as e:
        print(e)
        sys.exit(f"❌  Unknown task: {task_name}")


def _task_label(cfg: DictConfig) -> str:
    """Build a short task label including task-specific params (e.g. tsp30, symboreg_osc1)."""
    task = cfg.get("task", "")
    if task == "tsp":
        return f"tsp{cfg.get('city_num', 30)}"
    if task == "symboreg_oscillator1":
        return "symboreg_osc1"
    if task == "symboreg_oscillator2":
        return "symboreg_osc2"
    if task == "bin_packing":
        dtype = cfg.get("dataset_type", "or3")
        return f"bin_packing_{dtype}"
    if task == "promptopt":
        eval_task = cfg.get("eval_task", "sum")
        return f"promptopt_{eval_task}"
    return task or "run"


def _temp_str(t: float) -> str:
    """Format temperature for directory name (e.g. 0.1 -> 0p1)."""
    s = str(t).replace(".", "p")
    return s.replace("-", "m")

@hydra.main(config_path="../config", config_name="exp_grid")
def main(cfg: DictConfig):
    task_cfg = cfg.tasks.get(cfg.task, {})
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, task_cfg)
    print("\n========== config  ==========")
    print(OmegaConf.to_yaml(cfg))
    print("============================\n")

    task_mod = load_task(cfg.task)
    
    if hasattr(task_mod, 'configure'):
        task_mod.configure(cfg)
    
    task_label = _task_label(cfg)
    temp = cfg.get("temperature", 0.7)
    temp_str = _temp_str(temp)
    seed = cfg.seed
    ts = int(time.time())
    out_dir = PROJECT_ROOT / "outputs" / f"{task_label}_temp{temp_str}_{seed}_{ts}"

    stats = run_evolve(
        cfg=cfg,
        task_mod=task_mod,
        model_name=cfg.model,
        seed=seed,
        n_init=cfg.n_init,
        init_pop_path=cfg.get("init_pop_path", None),
        n_parent=cfg.parent_slots,
        n_child=cfg.child_slots,
        budget_calls=cfg.budget,
        db_mode=cfg.db_mode,
        db_kwargs={"capacity": cfg.capacity},
        max_workers=cfg.get("max_workers", None),
        log_callback=make_wandb_callback(project=f"EvolveEval-{cfg.task}-new", cfg=OmegaConf.to_container(cfg)),
        temperature=temp,
        out_dir=out_dir,
    )
    print("✅ Stats:", stats.dump())
    print("✅ Done. Best score:", stats.best_curve[-1])

if __name__ == "__main__":
    main()
