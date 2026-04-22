# LLMEvo: LLM-driven Evolutionary Optimization Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2604.19440-b31b1b.svg)](https://arxiv.org/abs/2604.19440)

An experimental framework for studying LLM-guided evolutionary search across optimization tasks. Supports systematic trajectory analysis as described in our paper.

## Features

- **Multi-LLM support**: OpenAI, Azure, OpenRouter, HuggingFace, Cerebras, Ollama (see `config/exp_grid.yaml` for full model list)
- **Multiple tasks**: TSP, Bin Packing, Prompt Optimization, Symbolic Regression (two oscillators)
- **Parallel offspring generation**: configurable `max_workers`
- **Zero-shot evaluation**: temperature sweep (0.0–1.0) for baseline comparison
- **Hydra config management**: centralized parameters with CLI overrides
- **W&B logging** (optional)
- **Refinement rate analysis**: `calculate_refinement_rate.py` for trajectory inspection

## Installation

Requires Python 3.12+. Uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/XINHAO-ZHANG/LLMEvo_Eval.git
cd LLMEvo_Eval
uv sync
```

Set up API keys (only the providers you use):

```bash
export OPENAI_API_KEY="your-openai-key"
export OPENROUTER_API_KEY="your-openrouter-key"   # for deepseek, llama, mistral, etc.
```

## Quick Start

```bash
# TSP with GPT-4o-mini, 300 LLM calls
uv run python scripts/run_exp.py task=tsp model=openai/gpt-4o-mini budget=300 seed=42

# Bin packing with DeepSeek via OpenRouter
uv run python scripts/run_exp.py task=bin_packing model=deepseek/deepseek-chat-v3-0324:free budget=200

# Symbolic regression (oscillator 1) with temperature sweep
uv run python scripts/run_exp.py task=symboreg_oscillator1 temperature=0.5 seed=21

# Prompt optimisation (summarization)
uv run python scripts/run_exp.py task=promptopt tasks.promptopt.eval_task=sum budget=150
```

Any parameter in `config/exp_grid.yaml` can be overridden from the command line.

Outputs are saved to `outputs/{task_label}_temp{temp}_{seed}_{timestamp}/`.

## Analyse Results

```bash
# Compute local refinement rate across all runs
python calculate_refinement_rate.py

# Single experiment
python calculate_refinement_rate.py --single outputs/tsp30_temp0p7_21_*/
```

## Project Structure

```
config/          # Hydra config (exp_grid.yaml)
data/            # Datasets (tsp, bin_packing, promptopt, symboreg)
tasks/           # Task definitions (tsp, bin_packing, promptopt, symboreg_oscillator{1,2})
evolve/          # Evolutionary algorithm core (loop.py, db.py)
llm/             # LLM interface (api.py, prompts.py, zero_shot_eval.py, providers/)
experiments/     # Hydra experiment entry point (run_experiment.py)
scripts/         # Main CLI (run_exp.py), W&B logger (wandb_logger.py)
utils/           # Shared utilities
calculate_refinement_rate.py  # Trajectory analysis script
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `task` | Task name (`tsp`, `bin_packing`, `promptopt`, `symboreg_oscillator1/2`) | `tsp` |
| `model` | LLM model string | `openai/gpt-4o-mini` |
| `n_init` | Initial population size | `40` |
| `parent_slots` | Parents sampled per generation | `3` |
| `child_slots` | Children generated per generation | `10` |
| `budget` | Total LLM call budget | `300` |
| `temperature` | LLM sampling temperature | `0.7` |
| `db_mode` | Population database (`simple`/`map`) | `simple` |
| `seed` | Random seed | `21` |
| `enable_zero_shot_eval` | Run temperature-sweep baseline | `false` |
| `max_workers` | Parallel offspring workers (null=auto) | `null` |

See `config/exp_grid.yaml` for the full parameter list and per-task overrides.

## Citation

If you find our work useful, please cite:

```bibtex
@misc{zhang2026makesllmgoodoptimizer,
      title={What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search},
      author={Xinhao Zhang and Xi Chen and François Portet and Maxime Peyrard},
      year={2026},
      eprint={2604.19440},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.19440},
}
```

