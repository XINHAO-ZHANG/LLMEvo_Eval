#!/usr/bin/env python
"""
Batch Experiment Runner

Run multiple experiments with different configurations.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import itertools
from datetime import datetime


def generate_experiment_configs(base_config: Dict[str, Any], 
                              param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters
    
    Args:
        base_config: Base configuration
        param_grid: Parameter grid for sweep
        
    Returns:
        List of configuration dictionaries
    """
    # Get all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    configs = []
    for combination in itertools.product(*param_values):
        config = base_config.copy()
        for name, value in zip(param_names, combination):
            # Handle nested parameters with dot notation
            keys = name.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        configs.append(config)
    
    return configs


def run_single_experiment(config: Dict[str, Any], 
                         experiment_dir: Path) -> Dict[str, Any]:
    """Run a single experiment
    
    Args:
        config: Experiment configuration
        experiment_dir: Directory for this experiment
        
    Returns:
        Results dictionary
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_file = experiment_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Build command
    cmd = [
        "python", "experiments/run_experiment.py",
        "--config-path", str(config_file.parent),
        "--config-name", config_file.stem
    ]
    
    # Add overrides for main parameters
    overrides = []
    for key, value in config.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                overrides.append(f"{key}.{subkey}={subvalue}")
        else:
            overrides.append(f"{key}={value}")
    
    cmd.extend(overrides)
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            return {
                "status": "success",
                "runtime": runtime,
                "stdout": result.stdout,
                "config": config
            }
        else:
            return {
                "status": "failed",
                "runtime": runtime,
                "error": result.stderr,
                "stdout": result.stdout,
                "config": config
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "runtime": time.time() - start_time,
            "config": config
        }
    except Exception as e:
        return {
            "status": "error", 
            "runtime": time.time() - start_time,
            "error": str(e),
            "config": config
        }


def main():
    """Main batch runner function"""
    
    # Example configuration
    base_config = {
        "task": {"name": "tsp"},
        "model": {"name": "gpt-4o"},
        "seed": 0,
        "budget": 100
    }
    
    # Parameter grid for sweep
    param_grid = {
        "task.name": ["tsp", "bin_packing"],
        "model.name": ["gpt-4o", "claude-3-5-sonnet"],
        "seed": [0, 1, 2],
        "parent_slots": [4, 6],
        "child_slots": [8, 12]
    }
    
    # Generate configurations
    configs = generate_experiment_configs(base_config, param_grid)
    
    print(f"📊 Generated {len(configs)} experiment configurations")
    
    # Setup batch directory
    batch_dir = Path("outputs") / f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    results = []
    for i, config in enumerate(configs):
        print(f"\n{'='*50}")
        print(f"🔄 Experiment {i+1}/{len(configs)}")
        print(f"{'='*50}")
        
        exp_dir = batch_dir / f"exp_{i:03d}"
        result = run_single_experiment(config, exp_dir)
        results.append(result)
        
        # Save intermediate results
        results_file = batch_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Experiment {i+1} completed: {result['status']}")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    print(f"\n{'='*50}")
    print(f"📈 Batch Experiment Summary")
    print(f"{'='*50}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()