#!/usr/bin/env python
"""CLI to launch an EvoEval run or sweep.

   Examples
   --------
   python scripts/run_exp.py --task tsp --model gpt-4o --db_mode map --budget 300
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from evolve_core import run_evolve
from scripts.wandb_logger import make_wandb_callback
import importlib
import sys

TASK_PKG = "tasks"

########################################################################
# Helpers
########################################################################

def load_task(task_name: str):
    try:
        return importlib.import_module(f"{TASK_PKG}.{task_name}")
    except ModuleNotFoundError as e:
        print(e)
        sys.exit(f"❌  Unknown task: {task_name}")

@hydra.main(config_path="../config", config_name="exp_grid")
def main(cfg: DictConfig):
    # 选择任务专属参数并merge
    task_cfg = cfg.tasks.get(cfg.task, {})
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, task_cfg)
    print("\n========== config  ==========")
    print(OmegaConf.to_yaml(cfg))
    print("============================\n")

    task_mod = load_task(cfg.task)
    safe_model_name = cfg.model.replace("/", "_")

    stats = run_evolve(
        cfg=cfg,
        task_mod=task_mod,
        model_name=cfg.model,
        seed=cfg.seed,
        n_init=cfg.n_init,
        init_pop_path=cfg.get("init_pop_path", None),
        n_parent=cfg.parent_slots,
        n_child=cfg.child_slots,
        budget_calls=cfg.budget,
        db_mode=cfg.db_mode,
        db_kwargs={"capacity": cfg.capacity},
        log_callback=make_wandb_callback(project=f"EvolveEval-{cfg.task}-new", cfg=OmegaConf.to_container(cfg)),
    )
    print("✅ Stats:", stats.dump())
    print("✅ Done. Best score:", stats.best_curve[-1])

if __name__ == "__main__":
    main()
