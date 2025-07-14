import numpy as np
import pandas as pd
import textwrap
import random
import json
import re
from evolve_core.db import Genome

# ------------------------------ CONFIG ---------------------------------- #
import os
from pathlib import Path

# 获取项目根目录的绝对路径
PROJECT_ROOT = Path(__file__).parent.parent  # tasks目录的上一级
DATA_DIR = PROJECT_ROOT / "data" / "symboreg" / "oscillator1"
MAX_NPARAMS = 3  # 以三项为例，和骨架保持一致
ALLOWED_FUNCS = ["np.sin", "np.cos", "np.exp", "np.log", "np.abs", "np.sqrt"]
EQUATION_TEMPLATE = """
def equation(x, v, params):
    
    return params[0]*x + params[1]*v + params[2]
"""
SYS_PROMPT = "You are a scientific equation discovery expert. Your goal is to propose symbolic expressions that fit the data well (lower MSE is better)."

# -------------------------- 数据加载与适应度 ----------------------------- #
def load_data(split="train"):
    df = pd.read_csv(f"{DATA_DIR}/{split}.csv")
    X = df[["x", "v"]].values
    y = df["a"].values
    return X, y

def eval(code_str, split="train") -> float:
    X, y_true = load_data(split)
    code_str = repair(code_str)
    try:
        local_env = {}
        exec(code_str, {"np": np}, local_env)
        equation = local_env["equation"]
        y_pred = equation(X[:,0], X[:,1])
        mse = np.mean((y_pred - y_true) ** 2)
        return mse
    except Exception:
        pass

def eval_tout(genome):
    return {split: eval(genome, split) for split in ["train", "test_id", "test_ood"]}

def repair(code_str: str) -> str:
    """修复代码字符串，如果无法修复则返回原字符串"""
    if code_str is None:
        return ""
    
    # 这里可以添加具体的修复逻辑
    # 目前只是简单返回原字符串
    return code_str

def create_fallback_genome():
    """当LLM生成失败时，创建一个fallback基因组"""
    fallback_code = "def equation(x, v):\n    return 0"
    fallback_loss = 1e6  # 很高的损失值，确保不会被选择
    return fallback_code, fallback_loss

# -------------------------- 进化相关接口 ----------------------------- #
def random_expr(rng: random.Random, max_terms=5):
    ops = ["+", "-", "*", "/"]
    funcs = ["", "np.sin", "np.cos", "np.exp", "np.log", "np.abs"]
    vars = ["x", "v"]
    n_terms = rng.randint(2, max_terms)
    expr_terms = []
    for _ in range(n_terms):
        func = rng.choice(funcs)
        var = rng.choice(vars)
        op = rng.choice(ops)
        coef = rng.uniform(-2, 2)
        base = f"{coef:.3f}*{var}"
        if op == "/":
            # 分母加绝对值和小常数
            denom = f"np.abs({base})+1e-6"
            numer = f"{rng.uniform(-2,2):.3f}*{rng.choice(vars)}"
            term = f"/ ({denom})"
            # 让分子和分母都合理
            if func and func != "np.log":
                numer = f"{func}({numer})"
            term = f"{op} {numer} {term}"
        else:
            # 其他操作同上
            if func == "np.log":
                base = f"np.abs({base})+1e-6"
            if func:
                term = f"{op} {func}({base})"
            else:
                term = f"{op} {base}"
        expr_terms.append(term)
    expr = " ".join(expr_terms).lstrip("+*/- ")
    return f"def equation(x, v):\n    return {expr}"

def random_params(rng: random.Random, n_params=MAX_NPARAMS):
    return [rng.uniform(-2, 2) for _ in range(n_params)]

def seed_pool(n: int, rng: random.Random):
    genomes = []
    for _ in range(n):
        code = random_expr(rng)
        loss = eval(code, split="train")
        genomes.append(Genome(genome=code, loss=loss, extra={}))
    return genomes

def diversity_key(g: Genome):
    return hash(g.genome[:40])

# -------------------------- LLM交互相关 ----------------------------- #
def get_zero_shot_prompt(seed=None, temperature_index=None, call_index=None):
    """
    生成zero-shot提示，支持精细的种子控制
    
    Args:
        seed: 基础种子，如果为None则不控制随机性
        temperature_index: 温度索引 (0-4 对应 5种温度)
        call_index: 调用索引 (0-1 对应每种温度的2次调用)
    """
    sys = SYS_PROMPT
    X, y = load_data("train")
    
    # 精细的种子控制策略
    if seed is not None:
        if temperature_index is not None and call_index is not None:
            # 为每种温度的每次调用生成独特但确定性的种子
            specific_seed = seed + temperature_index * 100 + call_index * 10
        else:
            # 使用基础种子
            specific_seed = seed
        
        np.random.seed(specific_seed)
        print(f"Zero-shot sampling with seed: {specific_seed} (temp_idx={temperature_index}, call_idx={call_index})")
    
    idx = np.random.choice(len(X), size=100, replace=False)
    sample = [{"x": float(X[i][0]), "v": float(X[i][1]), "a": float(y[i])} for i in idx]
    ques_block = json.dumps(sample, ensure_ascii=False, indent=2)
    
    user = textwrap.dedent(f"""
        TASK DESC    : {describe()}
        QUESTION    : Here are some data points:
        ```json
        {ques_block}
        ```
        Please return a Python function as a code block that fits the data without any explanation.
        The function should take x and v as input and return the predicted acceleration a.
        For example:
        ```python
        def equation(x, v):
            return 1.2*x + 0.8*v + np.sin(x)
        ```
        """)
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]

def get_evolve_prompt(sampled_parents):
    sys = "You are a scientific equation discovery expert. Your goal is to propose a new, better mathematical expression that fits the data well (lower MSE is better)."
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
        Please return a new, better Python function as a code block without any explanation.
        The function should take x and v as input and return the predicted acceleration a.
        For example:
        ```python
        def equation(x, v):
            return 1.2*x + 0.8*v + np.sin(x)
        ```
        """)
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]

def parse_response(resp):
    content = resp.get("text", "")
    m = re.search(r'```python(.*?)```', content, re.S)
    if m:
        code = m.group(1).strip()
        return {"genome": code, "usage": resp.get("usage", {"total_tokens": 0})}
    return {"genome": content, "usage": resp.get("usage", {"total_tokens": 0})}

def describe() -> str:
    return (
        "This is a symbolic regression task for a damped nonlinear oscillator. "
        "Given (x, v) and acceleration a, your goal is to find a mathematical expression a = f(x, v) that fits the data as accurately as possible. "
        "The function should be a valid Python function of x and v, and may use mathematical operations and numpy functions."
    )
