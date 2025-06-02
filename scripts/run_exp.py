#!/usr/bin/env python
"""CLI to launch an EvoEval run or sweep.

   Examples
   --------
   python scripts/run_exp.py --task tsp --model gpt-4o --db_mode map --budget 300
"""
from __future__ import annotations

import argparse, importlib, yaml, json, sys
from pathlib import Path
from evolve_core import run_evolve
from scripts.wandb_logger import make_wandb_callback

TASK_PKG = "tasks"

########################################################################
# Helpers
########################################################################

def load_task(task_name: str):
    try:
        return importlib.import_module(f"{TASK_PKG}.{task_name}")
    except ModuleNotFoundError as e:
        sys.exit(f"❌  Unknown task: {task_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="tsp | gcolor | promptopt | codegen")
    ap.add_argument("--model", required=True)
    ap.add_argument("--db_mode", default="simple", choices=["simple", "map"])
    ap.add_argument("--parent_slots", type=int, default=4)
    ap.add_argument("--child_slots", type=int, default=8)
    ap.add_argument("--n_init", type=int, default=40)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_pop_path", type=str, default=None, help="Path to pop.json for resuming or initializing population")
    ap.add_argument("--capacity", type=int, default=40)
    ap.add_argument("--cfg", help="YAML file that overrides/augments CLI args")
    args = ap.parse_args()

    # Optional YAML override (useful for sweeps) -------------------------
    if args.cfg:
        cfg = yaml.safe_load(Path(args.cfg).read_text())
        for k, v in cfg.items():
            setattr(args, k, v)

    task_mod = load_task(args.task)

    safe_model_name = args.model.replace("/", "_")

    stats = run_evolve(
        task_mod=task_mod,
        model_name=args.model,
        seed=args.seed,
        n_init=args.n_init,
        init_pop_path=args.init_pop_path,
        n_parent=args.parent_slots,
        n_child=args.child_slots,
        budget_calls=args.budget,
        db_mode=args.db_mode,
        db_kwargs={"capacity": args.capacity},
        log_callback=make_wandb_callback(project=f"EvolveEval-{args.task}-new", cfg=vars(args)),
    )
    print("✅ Stats:",stats.dump())
    print("✅ Done. Best score:", stats.best_curve[-1])

if __name__ == "__main__":
    main()
