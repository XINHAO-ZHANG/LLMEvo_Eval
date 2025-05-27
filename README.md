# EvoEval: LLM-driven Evolutionary Optimization Evaluation Framework

## Overview

This project implements a general-purpose optimization platform that combines Large Language Models (LLMs) with evolutionary algorithms, inspired by the recent work of AlphaEvovle by GoogleDeepmind. By leveraging LLMs to generate, mutate, and evolve candidate solutions ("genomes") and evaluating them with task-specific fitness functions, the framework enables automated intelligent search and optimization. It supports multiple tasks (e.g., TSP, Graph Coloring, Prompt Optimization, Code Generation) and various LLM providers (OpenAI, Azure, Ollama/local, etc.).

---

## Directory Structure

```
.
├── tasks/           # Task definitions (TSP, Graph Coloring, PromptOpt, CodeGen, etc.)
├── evolve_core/     # Evolutionary algorithm core (main loop, database, parallel evaluation)
├── llm_ops/         # LLM API wrappers and prompt builders
├── scripts/         # Experiment runner scripts and logging tools
├── config/          # Configuration files
├── data/            # Task data (distance matrices, graph structures, etc.)
```

---

## Main Features & Principles

- **Evolutionary Main Loop**: Each generation samples parents from the population, calls the LLM to generate children, repairs and evaluates their fitness, updates the population database, and logs progress.
- **Multi-task Support**: Each task module implements `seed_pool` (initial solution generation), `fitness` (evaluation), `repair` (fix invalid solutions), etc.
- **Multi-LLM Support**: Supports OpenAI, Azure, Ollama/local models via a unified API.
- **Database Modes**: Supports both simple pool (`simple`) and MAP-Elites (`map`) population management strategies.
- **Logging & Visualization**: Integrated with wandb for real-time tracking of optimization progress and population diversity.

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

## Key Parameters

| Parameter        | Description                                         | Example                  |
|------------------|-----------------------------------------------------|--------------------------|
| --task           | Task name (tsp/gcolor/promptopt/codegen)            | --task tsp               |
| --model          | LLM name (e.g. openai/gpt-4o, gpt-3.5-turbo)        | --model openai/gpt-4o    |
| --n_init         | Initial population size                             | --n_init 40              |
| --parent_slots   | Number of parents sampled per generation            | --parent_slots 20        |
| --child_slots    | Number of children generated per generation         | --child_slots 8          |
| --budget         | Total LLM call budget                               | --budget 80              |
| --db_mode        | Database mode (simple/map)                          | --db_mode simple         |
| --seed           | Random seed                                         | --seed 21                |
| --capacity       | Maximum number of solutions in the population       | --capacity 40            |
| --cfg            | YAML config file path (optional, overrides CLI)     | --cfg config/xxx.yaml    |

More parameters can be found in `config/exp_grid.yaml`.

---

## Example Usage

```bash
python -m scripts.run_exp \
  --task tsp \
  --model openai/gpt-4o \
  --n_init 40 \
  --parent_slots 20 \
  --child_slots 8 \
  --budget 80 \
  --db_mode simple \
  --seed 21
```

- Supported models and tasks are listed in `config/exp_grid.yaml`.
- Logs and results are saved in the `runs/` directory, and can be visualized in real time via wandb.

---

## Installation

- Python 3.8+
- Main dependencies: `openai`, `requests`, `wandb`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `dotenv`
- Example installation:

```bash
pip install openai requests wandb numpy pandas scikit-learn pyyaml dotenv
```

---

## Advanced Usage

- **Custom Tasks**: Add a new module under `tasks/` and implement `seed_pool`, `fitness`, `repair`, etc.
- **Custom Prompts**: Modify `llm_ops/prompt.py` to adapt LLM interaction for your task.
- **Batch Experiments**: Use shell scripts and the `--cfg` parameter for sweeps.

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
