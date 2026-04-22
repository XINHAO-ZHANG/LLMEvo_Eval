import numpy as np
import pandas as pd
import textwrap
import random
import json
import re
from evolve.db import Genome

# ------------------------------ CONFIG ---------------------------------- #
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "symboreg" / "oscillator2"
ALLOWED_FUNCS = ["np.sin", "np.cos", "np.exp", "np.log", "np.abs", "np.sqrt", "np.tanh"]
SYS_PROMPT = "You are a scientific equation discovery expert. Your goal is to propose symbolic expressions for acceleration in a damped nonlinear oscillator system with driving force, given time, position, and velocity data. Lower MSE is better."

def configure(cfg=None):
    global ALLOWED_FUNCS
    if cfg and hasattr(cfg, 'allowed_funcs'):
        ALLOWED_FUNCS = cfg.allowed_funcs
    print(f"Symboreg Oscillator2 configured with {len(ALLOWED_FUNCS)} allowed functions")

# ------------------------------ DATA & FITNESS -------------------------- #
def load_data(split="train"):
    df = pd.read_csv(f"{DATA_DIR}/{split}.csv")
    inputs = df[["t", "x", "v"]].values
    outputs = df["a"].values
    return inputs, outputs

def eval(code_str, split="train") -> float:
    inputs, outputs = load_data(split)
    code_str = repair(code_str)
    try:
        local_env = {}
        exec(code_str, {"np": np}, local_env)
        equation = local_env.get("equation")
        if equation is None:
            return 1e6
        try:
            y_pred = equation(inputs[:,0], inputs[:,1], inputs[:,2])
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return 1e6
            return float(np.mean((y_pred - outputs) ** 2))
        except Exception:
            return 1e6
    except Exception:
        return 1e6

def eval_tout(genome):
    return {split: eval(genome, split) for split in ["train", "test_id", "test_ood"]}

def repair(code_str: str) -> str:
    if code_str is None:
        return ""
    return code_str

def create_fallback_genome():
    fallback_code = "def equation(t, x, v):\\n    return 0"
    fallback_loss = 1e6
    return fallback_code, fallback_loss

# ------------------------------ EVOLUTION INTERFACE --------------------- #
def random_expr(rng: random.Random, max_terms=5):
    """Generate a random syntactically valid expression."""
    ops = ["+", "-", "*", "/"]
    funcs = ["", "np.sin", "np.cos", "np.exp", "np.log", "np.abs"]
    vars = ["t", "x", "v"]

    n_terms = rng.randint(2, max_terms)
    expr_terms = []

    for _ in range(n_terms):
        func = rng.choice(funcs)
        var = rng.choice(vars)
        op = rng.choice(ops)
        coef = rng.uniform(-2, 2)
        base = f"{coef:.3f}*{var}"

        if op == "/":
            # add absolute value and small constant to avoid division by zero
            denom = f"np.abs({base})+1e-6"
            numer = f"{rng.uniform(-2,2):.3f}*{rng.choice(vars)}"
            term = f"/ ({denom})"
            if func and func != "np.log":
                numer = f"{func}({numer})"
            term = f"{numer} {term}"
        else:
            if func == "np.log":
                # ensure log argument is positive
                base = f"np.abs({base})+1e-6"
            if func:
                term = f"{op} {func}({base})"
            else:
                term = f"{op} {base}"

        expr_terms.append(term)

    expr = " ".join(expr_terms).lstrip("+*/- ")
    return f"def equation(t, x, v):\n    return {expr}"

def seed_pool(n: int, rng: random.Random):
    genomes = []
    for _ in range(n):
        code = random_expr(rng)
        loss = eval(code, split="train")
        genomes.append(Genome(genome=code, loss=loss, extra={}))
    return genomes

def diversity_key(g: Genome):
    return hash(g.genome[:60])

# ------------------------------ LLM INTERFACE --------------------------- #
def get_zero_shot_prompt(seed=None, temperature_index=None, call_index=None):
    """Generate zero-shot prompt with fine-grained seed control."""
    sys = SYS_PROMPT
    inputs, outputs = load_data("train")

    if seed is not None:
        if temperature_index is not None and call_index is not None:
            specific_seed = seed + temperature_index * 100 + call_index * 10
        else:
            specific_seed = seed
        np.random.seed(specific_seed)
        print(f"Zero-shot sampling with seed: {specific_seed} (temp_idx={temperature_index}, call_idx={call_index})")

    idx = np.random.choice(len(inputs), size=min(150, len(inputs)), replace=False)
    sample = []
    for i in idx:
        sample.append({
            "t": float(inputs[i][0]),
            "x": float(inputs[i][1]),
            "v": float(inputs[i][2]),
            "a": float(outputs[i])
        })

    ques_block = json.dumps(sample, ensure_ascii=False, indent=2)

    user = textwrap.dedent(f"""
        TASK DESC    : {describe()}
        QUESTION    : Here are some data points from a damped nonlinear oscillator with driving force:
        ```json
        {ques_block}
        ```
        Please return a Python function that takes time (t), position (x), and velocity (v) as input and returns the predicted acceleration (a).

        For example:
        ```python
        def equation(t, x, v):
            return 1.2*x + 0.8*v + 0.5*np.sin(t)
        ```

        Return only the Python function without any explanation.
        """)

    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]

def get_evolve_prompt(sampled_parents):
    sys = "You are a scientific equation discovery expert. Your goal is to propose a new, better mathematical expression for acceleration in a damped nonlinear oscillator system that fits the data well (lower MSE is better)."
    parent_block = json.dumps(
        [{"code": g.genome, "mse_score": g.loss} for g in sampled_parents],
        ensure_ascii=False, indent=2
    )

    user = textwrap.dedent(f"""
        TASK DESC    : {describe()}
        Here are previous candidate expressions and their MSE scores:
        ```json
        {parent_block}
        ```
        Please return a new, better Python function that takes time (t), position (x), and velocity (v) as input and returns the predicted acceleration (a).

        For example:
        ```python
        def equation(t, x, v):
            return 1.2*x + 0.8*v + 0.5*np.sin(t) + 0.1*x**2
        ```

        Return only the Python function without any explanation.
        """)

    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]

def parse_response(resp):
    content = resp.get("text", "")

    m = re.search(r'```python(.*?)```', content, re.S)
    if m:
        code = m.group(1).strip()
        return {"genome": code, "usage": resp.get("usage", {"total_tokens": 0})}

    m = re.search(r'def equation\(.*?\):(.*?)(?=\n\S|\Z)', content, re.S)
    if m:
        func_def = f"def equation{m.group(0)[12:]}"
        return {"genome": func_def, "usage": resp.get("usage", {"total_tokens": 0})}

    return {"genome": content, "usage": resp.get("usage", {"total_tokens": 0})}

def describe() -> str:
    return (
        "This is a symbolic regression task for a damped nonlinear oscillator system with driving force. "
        "Given time (t), position (x), velocity (v), and acceleration (a) data, your goal is to find a mathematical expression a = f(t, x, v) that fits the data as accurately as possible. "
        "The function should be a valid Python function that takes t, x, v as input and may use mathematical operations and numpy functions."
    )
