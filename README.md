# LLMEvo: LLM-driven Evolutionary Optimization Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLMEvo is a comprehensive framework for solving optimization problems using Large Language Models (LLMs) and evolutionary algorithms. Inspired by Google DeepMind's work, it combines the reasoning capabilities of LLMs with the search power of evolutionary algorithms to tackle complex optimization challenges.

## 🚀 Features

- **Multi-LLM Support**: OpenAI GPT, Anthropic Claude, local Ollama models
- **Diverse Tasks**: TSP, Bin Packing, Prompt Optimization, Symbolic Regression
- **Modular Design**: Clean separation of concerns with pluggable components
- **Parallel Execution**: Efficient parallel evaluation of candidates
- **Experiment Tracking**: Built-in logging and optional W&B integration
- **Configuration Management**: Hierarchical configuration with Hydra
- **Zero-shot Evaluation**: Baseline performance assessment

## 📦 Installation

### Clone and Setup

```bash
git clone https://github.com/your-username/LLMEvo_Eval.git
cd LLMEvo_Eval
pip install -r requirements.txt
```

### Set up API Keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
```

## 🏗️ Project Structure

```
LLMEvo_Eval/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── config/                           # Configuration management
│   └── exp_grid.yaml                # Experiment configuration
│
├── data/                            # Datasets and benchmarks
│   ├── tsp/                         # TSP instances
│   ├── bin_packing/                 # Bin packing problems
│   ├── symboreg/                    # Symbolic regression data
│   └── promptopt/                   # Prompt optimization tasks
│
├── tasks/                           # Task definitions
│   ├── base.py                      # Abstract base class
│   ├── tsp.py                       # TSP implementation
│   └── ...                          # Other task modules
│
├── evolve/                          # Evolutionary algorithm core
│   ├── algorithm.py                 # Main evolution loop
│   ├── population.py                # Population management
│   └── database.py                  # Results database
│
├── llm/                            # LLM interface layer
│   ├── api.py                       # Unified API interface
│   ├── prompts.py                   # Prompt engineering
│   └── providers/                   # Different LLM providers
│
├── evaluation/                     # Evaluation system
│   ├── evaluator.py                # Base evaluator
│   └── parallel.py                 # Parallel evaluation
│
├── utils/                          # Utility functions
│   ├── logging.py                  # Logging utilities
│   └── visualization.py            # Plotting functions
│
├── experiments/                    # Experiment management
│   ├── run_experiment.py          # Main experiment runner
│   ├── batch_runner.py            # Batch experiment execution
│   └── analysis/                  # Result analysis scripts
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Data analysis
│   ├── 02_algorithm_analysis.ipynb # Algorithm visualization
│   └── templates/                  # Notebook templates
│
├── tests/                          # Unit tests
│   ├── test_tasks/                 # Task tests
│   ├── test_evolve/               # Algorithm tests
│   └── test_llm/                  # LLM interface tests
│
├── scripts/                       # Utility scripts
│   └── clean_logs.py              # Log cleanup
│
└── docs/                          # Documentation
    ├── api/                       # API documentation
    ├── tutorials/                 # Usage tutorials
    └── examples/                  # Code examples
```

## 🎯 Quick Start

### 1. Set up API Keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
```

### 2. Run a Simple Experiment

```bash
# Run TSP with GPT-4
python experiments/run_experiment.py task=tsp model=gpt-4o

# Run with custom parameters
python experiments/run_experiment.py task=tsp model=gpt-4o seed=42 budget=100
```

### 3. Batch Experiments

```bash
python experiments/batch_runner.py
```

### 4. Programmatic Usage

```python
import sys
from pathlib import Path
sys.path.append('.')  # Add project root to path

from evolve import run_evolve
from tasks import get_task

# Load a task
task_module = get_task("tsp")

# Configure and run evolution
results = run_evolve(
    cfg=config,
    task_mod=task_module,
    model_name="gpt-4o",
    seed=42,
    budget_calls=100
)

print(f"Best fitness: {min(results.best_curve)}")
```

## 📊 Configuration

LLMEvo uses a simple, centralized configuration file (`config/exp_grid.yaml`) that supports both global and task-specific parameters:

```yaml
# Global default parameters
task: tsp
model: openai/gpt-4o
seed: 0
budget: 300
n_init: 40
parent_slots: 4
child_slots: 8
db_mode: simple
enable_zero_shot_eval: true

# Task-specific overrides
tasks:
  tsp:
    n_init: 60
    parent_slots: 6
    child_slots: 12
    max_workers: 4
  
  promptopt:
    n_init: 10
    parent_slots: 2
    child_slots: 5
    budget: 150
    eval_task: sum
    max_workers: 2
```

You can override any parameter from command line:

```bash
python experiments/run_experiment.py task=tsp model=openai/gpt-4o budget=100 seed=42
```

## 🔧 Adding New Tasks

1. **Create task module** in `tasks/`:

```python
from tasks.base import BaseTask

class MyTask(BaseTask):
    def init(self, n_population, rng):
        # Generate initial population
        pass
    
    def eval(self, genome):
        # Evaluate genome fitness
        pass
    
    def get_evolve_prompt(self, parents):
        # Generate evolution prompt
        pass
    
    def get_zero_shot_prompt(self):
        # Generate zero-shot prompt
        pass
```

2. **Add configuration** in `config/exp_grid.yaml`

3. **Register task** in `tasks/__init__.py`

## 📈 Experiment Analysis

Use the provided Jupyter notebooks for analysis:

- **Data Exploration**: `notebooks/01_data_exploration.ipynb`
- **Algorithm Analysis**: `notebooks/02_algorithm_analysis.ipynb`
- **Results Visualization**: `notebooks/03_results_visualization.ipynb`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Google DeepMind's evolutionary optimization research
- Built with Hydra for configuration management
- Uses OpenAI and Anthropic APIs for LLM integration

## 📚 Citation

If you use LLMEvo in your research, please cite:

```bibtex
@software{llmevo2024,
  title={LLMEvo: LLM-driven Evolutionary Optimization Framework},
  author={Zhang, Xin},
  year={2024},
  url={https://github.com/your-username/LLMEvo_Eval}
}
```
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
