#!/usr/bin/env python
"""
Main Experiment Runner

This script provides a unified entry point for running LLMEvo experiments
with the new modular structure.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys
import os
import logging
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolve import run_evolve
from tasks import get_task
from llm import get_llm_provider
from utils import setup_logger


def setup_experiment(cfg: DictConfig) -> tuple:
    """Setup experiment components
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Tuple of (task_module, output_dir)
    """
    # Setup output directory
    output_dir = Path(cfg.get("output_dir", "outputs")) 
    output_dir = output_dir / cfg.task / cfg.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(
        "llmevo", 
        output_dir / "experiment.log",
        getattr(logging, cfg.get("log_level", "INFO"))
    )
    
    # Load task
    task_module = get_task(cfg.task)
    
    # Load task data if specified - use original task loading logic
    if hasattr(task_module, "load_data"):
        # Use default data path or configured path
        data_path = f"data/{cfg.task}"
        if Path(data_path).exists():
            task_module.load_data(data_path)
    
    logger.info(f"Experiment setup complete:")
    logger.info(f"  Task: {cfg.task}")
    logger.info(f"  Model: {cfg.model}")
    logger.info(f"  Output: {output_dir}")
    
    return task_module, output_dir


def create_wandb_callback(cfg: DictConfig):
    """Create W&B callback if tracking is enabled"""
    if not cfg.get("enable_wandb", False):
        return None
    
    try:
        import wandb
        
        wandb.init(
            project=cfg.get("wandb_project", "llmevo"),
            entity=cfg.get("wandb_entity"),
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.task}_{cfg.model}_{cfg.seed}"
        )
        
        def wandb_callback(log_data):
            wandb.log({
                "generation": log_data["gen"],
                "best_fitness": log_data["best_so_far"],
                "population_size": len(log_data["population"])
            })
        
        return wandb_callback
    except ImportError:
        logging.warning("wandb not installed, skipping tracking")
        return None


@hydra.main(config_path="../config", config_name="exp_grid", version_base="1.1")
def main(cfg: DictConfig) -> float:
    """Main experiment function
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Best fitness achieved
    """
    print("="*60)
    print("🚀 LLMEvo Experiment Runner")
    print("="*60)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Setup experiment
    task_module, output_dir = setup_experiment(cfg)
    
    # Setup callbacks
    wandb_callback = create_wandb_callback(cfg)
    
    # Extract parameters from simple config structure
    evolve_params = {
        "seed": cfg.get("seed", 0),
        "n_init": cfg.get("n_init", 40),
        "n_parent": cfg.get("parent_slots", 4),
        "n_child": cfg.get("child_slots", 8),
        "budget_calls": cfg.get("budget", 200),
        "init_pop_path": cfg.get("init_pop_path"),
        "db_mode": cfg.get("db_mode", "simple"),
        "out_dir": output_dir,
        "log_callback": wandb_callback,
        "max_workers": cfg.get("max_workers", 4)
    }
    
    # Apply task-specific overrides if they exist
    task_cfg = cfg.get("tasks", {}).get(cfg.task, {})
    for key, value in task_cfg.items():
        if key in ["n_init", "parent_slots", "child_slots", "budget"]:
            param_name = {"parent_slots": "n_parent", "child_slots": "n_child", "budget": "budget_calls"}.get(key, key)
            evolve_params[param_name] = value
    
    print(f"\n🔧 Evolution Parameters:")
    for key, value in evolve_params.items():
        if key != "log_callback":
            print(f"  {key}: {value}")
    
    # Run evolution
    print(f"\n🧬 Starting evolution...")
    try:
        stats = run_evolve(
            cfg=cfg,
            task_mod=task_module,
            model_name=cfg.model,
            **evolve_params
        )
        
        print(f"\n✅ Evolution completed!")
        print(f"  Best fitness: {min(stats.best_curve) if stats.best_curve else 'N/A'}")
        print(f"  Generations: {len(stats.best_curve)}")
        print(f"  Total calls: {stats.calls}")
        print(f"  Runtime: {stats.wall:.2f}s")
        
        if wandb_callback:
            import wandb
            wandb.finish()
        
        return min(stats.best_curve) if stats.best_curve else float('inf')
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        if wandb_callback:
            import wandb
            wandb.finish()
        raise


if __name__ == "__main__":
    main()