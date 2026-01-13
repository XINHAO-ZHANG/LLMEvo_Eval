# Algorithm Analysis Notebook Template

Analyze and visualize evolutionary algorithm performance.

## Setup

```python
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path().parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plots
plt.style.use('default')
sns.set_palette("husl")
```

## Load Experimental Results

```python
# Load results from experiment runs
results_dir = Path("../outputs")

def load_experiment_results(exp_path):
    """Load results from an experiment directory"""
    stats_file = exp_path / "stats.json"
    gen_log_file = exp_path / "gen_log.jsonl" 
    
    if not stats_file.exists():
        return None
    
    # Load stats
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Load generation logs
    gen_logs = []
    if gen_log_file.exists():
        with open(gen_log_file) as f:
            for line in f:
                gen_logs.append(json.loads(line.strip()))
    
    return {"stats": stats, "gen_logs": gen_logs}

# Example: Load specific experiment
# exp_path = results_dir / "tsp" / "gpt-4o" / "your_run"
# results = load_experiment_results(exp_path)
```

## Convergence Analysis

```python
def plot_convergence(results_list, labels=None):
    """Plot convergence curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, results in enumerate(results_list):
        if results is None:
            continue
        
        best_curve = results["stats"]["best_curve"]
        label = labels[i] if labels else f"Run {i+1}"
        ax.plot(best_curve, label=label, linewidth=2)
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Convergence Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax

# Example usage:
# results_list = [load_experiment_results(path) for path in experiment_paths]
# plot_convergence(results_list, labels=["Run 1", "Run 2", "Run 3"])
```

## Performance Metrics

```python
def compute_performance_metrics(results_list):
    """Compute performance metrics across runs"""
    metrics = []
    
    for results in results_list:
        if results is None:
            continue
            
        stats = results["stats"]
        best_curve = stats["best_curve"]
        
        if best_curve:
            metrics.append({
                "final_best": min(best_curve),
                "initial_best": best_curve[0],
                "improvement": best_curve[0] - min(best_curve),
                "convergence_gen": len(best_curve),
                "total_calls": stats["calls"],
                "runtime": stats["wall"]
            })
    
    return pd.DataFrame(metrics)

# Example:
# metrics_df = compute_performance_metrics(results_list)
# print(metrics_df.describe())
```