# EvoEval: LLM-driven Evolutionary Optimization Evaluation Framework

## Overview

EvoEval is a general-purpose optimization platform that combines Large Language Models (LLMs) with evolutionary algorithms, inspired by Google DeepMind's AlphaEvolve. By leveraging LLMs to generate, mutate, and evolve candidate solutions ("genomes"), and evaluating them with task-specific fitness functions, EvoEval enables automated, intelligent search and optimization. The framework supports multiple tasks (e.g., TSP, Graph Coloring, Prompt Optimization, Code Generation) and a variety of LLM providers (OpenAI, Azure, Ollama/local, etc.).

---

## Directory Structure

```
.
├── tasks/           # Task definitions (TSP, Graph Coloring, PromptOpt, CodeGen, etc.)
├── evolve_core/     # Evolutionary algorithm core (main loop, database, parallel evaluation)
├── llm_ops/         # LLM API wrappers and prompt builders
├── scripts/         # Experiment runner scripts and logging tools
├── config/          # Configuration files (Hydra style, centralized)
├── data/            # Task data (distance matrices, graph structures, etc.)
```

---

## Configuration & Parameter Management (Hydra + exp_grid.yaml)

- **Centralized configuration**: All experiment parameters are managed in a single `config/exp_grid.yaml`, supporting both global and task-specific structured parameters.
- **Hydra integration**: The main entry script `scripts/run_exp.py` uses Hydra to automatically load configuration, so you don't need to pass arguments manually.
- **Task-structured config example**:

```yaml
# config/exp_grid.yaml

task: tsp
model: gpt-4o
seed: 0
n_init: 40
parent_slots: 4
child_slots: 8
budget: 200
capacity: 40
init_pop_path: null
db_mode: simple

tasks:
  tsp:
    n_init: 60
    parent_slots: 6
    child_slots: 12
  kernelopt:
    n_init: 30
    parent_slots: 3
    child_slots: 6
  gcolor:
    n_init: 50
    parent_slots: 5
    child_slots: 10
```

- At runtime, just specify `task`, and the corresponding task-specific parameters will be merged automatically.
- You can override any parameter from the command line, e.g.:
  `python scripts/run_exp.py task=kernelopt model=gpt-3.5-turbo`

---

## Main Features

- **Evolutionary main loop**: Each generation samples parents, calls the LLM to generate children, repairs and evaluates fitness, updates the population database, and logs progress.
- **Multi-task support**: Each task module implements `seed_pool`, `fitness`, `repair`, etc.
- **Multi-LLM support**: Unified API for OpenAI, Azure, Ollama/local, and more.
- **Database modes**: Supports both simple pool and MAP-Elites population management.
- **Logging & visualization**: Integrated with wandb for real-time tracking and diversity visualization.
- **Hydra config management**: Centralized experiment parameters, supports sweeps, and auto-generated experiment directories.

---

## Task Descriptions

### 1. TSP (Traveling Salesman Problem)
- **Goal**: Find the shortest route visiting all cities exactly once.
- **Genome**: A permutation of city indices.
- **Fitness**: Total path length (lower is better).

### 2. Graph Coloring
- **Goal**: Assign colors to nodes of an undirected graph to minimize the number of conflicting edges.
- **Genome**: List of color indices for each node.
- **Fitness**: Number of conflicting edges (lower is better).

### 3. Prompt Optimization
- **Goal**: Optimize a prompt to maximize LLM accuracy on a fixed QA set.
- **Genome**: Prompt string.
- **Fitness**: 1 - accuracy (lower is better).

### 4. Code Generation
- **Goal**: Generate a Python function that passes all unit tests.
- **Genome**: Python code string.
- **Fitness**: Number of failed tests (lower is better).

---

## Key Parameters (see exp_grid.yaml)

| Parameter      | Description                                   | Example                  |
|--------------- |-----------------------------------------------|--------------------------|
| task           | Task name (tsp/gcolor/promptopt/codegen)      | task: tsp                |
| model          | LLM name (e.g. openai/gpt-4o)                 | model: openai/gpt-4o     |
| n_init         | Initial population size                       | n_init: 40               |
| parent_slots   | Number of parents sampled per generation      | parent_slots: 20         |
| child_slots    | Number of children generated per generation   | child_slots: 8           |
| budget         | Total LLM call budget                         | budget: 80               |
| db_mode        | Database mode (simple/map)                    | db_mode: simple          |
| seed           | Random seed                                   | seed: 21                 |
| capacity       | Max population size                           | capacity: 40             |
| ...            | See exp_grid.yaml for more parameters         |                          |

---

## Example Usage

```bash
python scripts/run_exp.py task=tsp model=openai/gpt-4o n_init=40 parent_slots=20 child_slots=8 budget=80 db_mode=simple seed=21
```

- Supported models and tasks are listed in `config/exp_grid.yaml`.
- Logs and results are saved in the `runs/` directory and can be visualized in real time via wandb.

---

## Installation

- Python 3.8+
- Main dependencies: `openai`, `requests`, `wandb`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `dotenv`, `hydra-core`, `omegaconf`
- Example installation:

```bash
pip install openai requests wandb numpy pandas scikit-learn pyyaml dotenv hydra-core omegaconf
```

---

## Advanced Usage

- **Custom tasks**: Add a new module under `tasks/` and implement `seed_pool`, `fitness`, `repair`, etc.
- **Custom prompts**: Modify `llm_ops/prompt.py` to adapt LLM interaction for your task.
- **Batch experiments**: List sweep parameters directly in `exp_grid.yaml` or use shell scripts for batch runs.

---

## Results & Logging

- Each run creates a folder under `runs/` containing per-generation population, fitness curves, and the best solution.
- If wandb is configured, progress and diversity metrics are uploaded and visualized automatically.

---

## FAQ

- **API Key Setup**: Set `OPENAI_API_KEY` and other environment variables as needed (see comments in `llm_ops/api.py`).
- **Local Model Support**: For Ollama/local models, ensure the service is running (see `llm_ops/api.py`).
- **Task Data**: Some tasks (e.g., TSP, Graph Coloring) auto-generate data files in the `data/` directory.

---

For further customization or questions, please refer to the source code or contact the authors.
