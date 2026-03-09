# LLMEvo: LLM-driven Evolutionary Optimization Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An experimental framework for solving optimization problems with LLMs and evolutionary algorithms. Inspired by Google DeepMind's FunSearch.

## Features

- **Multi-LLM support**: OpenAI, Vertex AI, OpenRouter, etc. (see `config/exp_grid.yaml`)
- **Multiple tasks**: TSP, Bin Packing, Prompt Optimization, Symbolic Regression
- **Parallel offspring generation**: configurable `max_workers`
- **Hydra config management**: centralized parameters with CLI overrides
- **W&B logging** (optional)

## Installation

Requires Python 3.12+. Uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/your-username/LLMEvo_Eval.git
cd LLMEvo_Eval
uv sync
```

Set up API key:

```bash
export OPENAI_API_KEY="your-openai-key"
```

## Quick Start

```bash
# Run a single experiment
uv run python experiments/run_experiment.py task=tsp model=openai/gpt-4o-mini budget=100 seed=42

# Batch experiments
uv run python experiments/batch_runner.py
```

Any parameter in `config/exp_grid.yaml` can be overridden from the command line.

## Project Structure

```
config/          # Hydra config (exp_grid.yaml)
data/            # Datasets (tsp, bin_packing, promptopt)
tasks/           # Task definitions (base.py, tsp.py, bin_packing.py, promptopt.py, ...)
evolve/          # Evolutionary algorithm core (loop.py, db.py, executor.py)
llm/             # LLM interface (api.py, prompts.py, providers/)
experiments/     # Experiment entry points (run_experiment.py, batch_runner.py)
scripts/         # Utility scripts
utils/           # Helper functions
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `task` | Task name | `tsp` |
| `model` | LLM model | `openai/gpt-4o-mini` |
| `n_init` | Initial population size | `40` |
| `parent_slots` | Parents sampled per generation | `3` |
| `child_slots` | Children generated per generation | `10` |
| `budget` | Total LLM call budget | `300` |
| `db_mode` | Database mode (`simple`/`map`) | `simple` |
| `seed` | Random seed | `21` |

See `config/exp_grid.yaml` for the full parameter list.

## License

MIT
