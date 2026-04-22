import numpy as np
import pandas as pd
import textwrap
import random
import json
import re
import time
import pickle
from evolve.db import Genome

# ------------------------------ CONFIG ---------------------------------- #
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bin_packing"

DATASET_TYPE = 'or3'
ALL_INSTANCES = [f'u500_{i:02d}' for i in range(20)]  # OR3 has 20 instances

def configure(cfg=None):
    global DATASET_TYPE, ALL_INSTANCES

    if cfg:
        if hasattr(cfg, 'dataset_type'):
            DATASET_TYPE = cfg.dataset_type
            if DATASET_TYPE == 'weibull':
                ALL_INSTANCES = [f'test_{i}' for i in range(5)]
            elif DATASET_TYPE == 'or3':
                ALL_INSTANCES = [f'u500_{i:02d}' for i in range(20)]

    print(f"Bin Packing configured:")
    print(f"  DATASET_TYPE: {DATASET_TYPE}")
    print(f"  ALL_INSTANCES: {len(ALL_INSTANCES)} instances")

# ------------------------------ DATA LOADERS ---------------------------- #
def load_weibull_dataset():
    dataset_path = DATA_DIR / "weibull_dataset.pkl"
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

def load_or3_dataset():
    dataset_path = DATA_DIR / "OR3_dataset.pkl"
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

def get_dataset(dataset_type=None):
    if dataset_type is None:
        dataset_type = DATASET_TYPE

    if dataset_type == 'weibull':
        return load_weibull_dataset()
    elif dataset_type == 'or3':
        return load_or3_dataset()
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'weibull' or 'or3'")

SYS_PROMPT = "You are an expert in online bin packing algorithms. Your goal is to design priority functions that minimize the number of bins needed."

CORE_SNIPPETS = [
    "return -(bins - item)",                          # best-fit
    "return bins - item",                             # worst-fit
    "return np.ones_like(bins)",                      # first-fit
    textwrap.dedent("""
    rem = bins - item
    tight = (rem >= 0) & (rem < 0.2*bins.max())
    return np.where(tight, -rem, -1e6)
    """),                                             # only tight fits
    "return (bins**2)/(item + 1e-6)",
    "return (bins - item)/(bins + 1e-6)",
    "return safe_div((bins-item)**2, bins)",
    "return bins * (bins - item)",
    "return (bins - item) * bins**0.5",
]

# ------------------------------ CORE BIN PACKING ALGORITHM -------------- #
def get_valid_bin_indices(item, bins):
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]

def online_binpack(items, bins, priority_func):
    """Performs online binpacking of items into bins using priority_func."""
    packing = [[] for _ in bins]

    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        if len(valid_bin_indices) == 0:
            continue

        try:
            priorities = priority_func(item, bins[valid_bin_indices])
        except Exception:
            priorities = np.ones(len(valid_bin_indices))

        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)

    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


def evaluate_all_instances(priority_func, dataset_type='or3'):
    """Evaluate heuristic on ALL instances in the dataset (FunSearch-style)."""
    start_time = time.time()
    num_bins = []
    dataset = get_dataset(dataset_type)

    for i, instance_name in enumerate(ALL_INSTANCES):
        if time.time() - start_time > 100:
            print(f"Timeout after {i} instances, elapsed: {time.time() - start_time:.2f} seconds")
            return 5000.0

        try:
            capacity = dataset[instance_name]['capacity']
            items = dataset[instance_name]['items']
            bins = np.array([capacity for _ in range(dataset[instance_name]['num_items'])])
            _, bins_packed = online_binpack(items, bins, priority_func)
            bins_used = (bins_packed != capacity).sum()
            num_bins.append(bins_used)
        except Exception as e:
            print(f"Warning: Failed to evaluate instance {instance_name}: {e}")
            num_bins.append(5000)

    return np.mean(num_bins)

# ------------------------------ FITNESS FUNCTION ------------------------ #
def eval(code_str):
    """Evaluate a priority function on ALL instances of the dataset."""
    code_str = repair(code_str)

    local_env = {"np": np, "safe_div": lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y!=0)}
    try:
        exec(code_str, local_env)
        priority_func = local_env["priority"]
    except Exception as e:
        print(f"Error executing code: {e}")
        return 5000.0

    dataset_type = DATASET_TYPE

    start_eval = time.time()
    try:
        avg_bins = evaluate_all_instances(priority_func, dataset_type)
        eval_time = time.time() - start_eval

        if eval_time > 540:
            print(f"Warning: Evaluation took too long ({eval_time:.2f} seconds)")
            return 5000.0
        return float(avg_bins)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 5000.0
    finally:
        end_eval = time.time()
        print(f"Evaluation time: {end_eval - start_eval:.2f} seconds")


def repair(code_input):
    """Repair code string(s): wrap bare return statements in a priority() function."""

    def repair_single(code_str: str) -> str:
        if code_str is None or code_str.strip() == "":
            return """def priority(item, bins):
    return -(bins - item)"""

        code_str = code_str.strip()

        if "def priority(" not in code_str:
            if code_str.startswith("return "):
                code_str = f"""def priority(item, bins):
    {code_str}"""
            else:
                code_str = f"""def priority(item, bins):
    # Generated priority function
    {code_str}
    return -(bins - item)"""

        import re
        code_str = re.sub(r'\bd\b(?!\w)', 'bins', code_str)

        return code_str

    if isinstance(code_input, list):
        return [repair_single(code) for code in code_input]
    else:
        return repair_single(code_input)

def create_fallback_genome():
    fallback_code = """def priority(item, bins):
    return -(bins - item)"""
    fallback_loss = eval(fallback_code)
    return fallback_code, fallback_loss

# ------------------------------ EVOLUTION INTERFACE --------------------- #
def random_priority_func(rng: random.Random) -> str:
    snippet = rng.choice(CORE_SNIPPETS)
    header = "def priority(item, bins):\n    import numpy as np\n"
    body = "    " + snippet.replace("\n", "\n    ")
    return header + body

def seed_pool(n: int, rng: random.Random):
    genomes = []
    for _ in range(n):
        code = random_priority_func(rng)
        loss = eval(code)
        genomes.append(Genome(genome=code, loss=loss, extra={}))
    return genomes

def diversity_key(g: Genome):
    return hash(g.genome[:50])

# ------------------------------ LLM INTERFACE --------------------------- #
def get_zero_shot_prompt(seed=None, temperature_index=None, call_index=None):
    """Generate zero-shot prompt for bin packing priority function."""
    sys = SYS_PROMPT

    if seed is not None:
        if temperature_index is not None and call_index is not None:
            specific_seed = seed + temperature_index * 100 + call_index * 10
        else:
            specific_seed = seed
        np.random.seed(specific_seed)
        print(f"Zero-shot sampling with seed: {specific_seed} (temp_idx={temperature_index}, call_idx={call_index})")

    user = textwrap.dedent(f"""
        TASK DESC: {describe()}

        Your priority function will be called as:
        priority(item, bins)

        Where:
        - item: float, the size of current item to pack
        - bins: numpy array, remaining capacities of bins that can fit the item

        Return: numpy array of same length as bins, with priority scores

        IMPORTANT: Please return ONLY a Python function as a code block. Do not include any explanations or additional text.

        Example:
        ```python
        def priority(item, bins):
            # Best fit: prefer bins with least remaining space after placing item
            return -(bins - item)
        ```

        Your response should contain ONLY the code block, nothing else. Please provide a function better than the example above!!
        """)

    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]

def get_evolve_prompt(sampled_parents):
    """Generate evolution prompt based on parent priority functions."""
    sys = "You are an expert in online bin packing algorithms. Your goal is to design a new, better priority function that minimizes the number of bins needed."

    parent_block = json.dumps(
        [{"code": g.genome, "avg_bins_used": g.loss} for g in sampled_parents],
        ensure_ascii=False, indent=2
    )

    user = textwrap.dedent(f"""
        TASK DESC: {describe()}

        EVALUATION: Performance is measured by averaging across ALL {len(ALL_INSTANCES)} instances in the {DATASET_TYPE.upper()} dataset.

        Here are previous priority functions and their performance (average bins used across all instances):
        ```json
        {parent_block}
        ```

        Please return a new, better Python function as a code block without any explanation.
        The function should implement a priority heuristic for online bin packing.

        Your function signature should be:
        def priority(item, bins):
            # Your heuristic here
            return priority_scores

        Where:
        - item: float, size of current item to pack
        - bins: numpy array, remaining capacities of bins that can fit the item
        - return: numpy array of same size as bins with priority scores of each bin

        Try to achieve fewer average bins than the parent functions shown above!

        Example:
        ```python
        def priority(item, bins):
            # better heuristic example
        ```
        """)

    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]

def parse_response(resp):
    """Parse LLM response to extract priority function code."""
    content = resp.get("text", "")

    m = re.search(r'```python(.*?)```', content, re.S)
    if m:
        code = m.group(1).strip()
        return {"genome": code, "usage": resp.get("usage", {"total_tokens": 0})}

    m = re.search(r'```python(.*)', content, re.S)
    if m:
        code = m.group(1).strip()
        return {"genome": code, "usage": resp.get("usage", {"total_tokens": 0})}

    m = re.search(r'def priority\([^)]*\):(.*?)(?=\n\n|\n[^\s]|$)', content, re.S)
    if m:
        code = f"def priority{m.group(0)[len('def priority'):]}"
        return {"genome": code, "usage": resp.get("usage", {"total_tokens": 0})}

    print(f"Warning: Could not parse response, using fallback. Content: {content[:200]}...")
    fallback_code = """def priority(item, bins):
    return -(bins - item)"""
    return {"genome": fallback_code, "usage": resp.get("usage", {"total_tokens": 0})}

def describe() -> str:
    return (
        f"Online bin packing optimization task using ALL {len(ALL_INSTANCES)} instances from the {DATASET_TYPE.upper()} dataset. "
        "The goal is to minimize the average number of bins used across all instances. "
        "This comprehensive evaluation ensures good generalization across diverse item patterns."
    )
