import numpy as np
import pandas as pd
import textwrap
import random
import json
import re
import time
import pickle
from evolve_core.db import Genome

# ------------------------------ CONFIG ---------------------------------- #
import os
from pathlib import Path


# 获取项目根目录的绝对路径
PROJECT_ROOT = Path(__file__).parent.parent  # tasks目录的上一级
DATA_DIR = PROJECT_ROOT / "data" / "bin_packing"

# 全局配置变量
# DATASET_TYPE = 'weibull'  # 默认数据集
# TEST_NAME = 'test_0'      # 默认测试实例
DATASET_TYPE = 'or3'
# 不再区分训练测试集，使用全部实例
ALL_INSTANCES = [f'u500_{i:02d}' for i in range(20)]  # OR3有20个实例

def configure(cfg=None):
    """配置bin packing任务"""
    global DATASET_TYPE, ALL_INSTANCES
    
    if cfg:
        if hasattr(cfg, 'dataset_type'):
            DATASET_TYPE = cfg.dataset_type
            # 根据数据集类型更新实例列表
            if DATASET_TYPE == 'weibull':
                ALL_INSTANCES = [f'test_{i}' for i in range(5)]  # 假设weibull有10个实例
            elif DATASET_TYPE == 'or3':
                ALL_INSTANCES = [f'u500_{i:02d}' for i in range(20)]
    
    print(f"Bin Packing configured:")
    print(f"  DATASET_TYPE: {DATASET_TYPE}")
    print(f"  ALL_INSTANCES: {len(ALL_INSTANCES)} instances")

# -------------------------- 数据加载函数 ----------------------------- #
def load_weibull_dataset():
    """从pickle文件加载Weibull数据集"""
    dataset_path = DATA_DIR / "weibull_dataset.pkl"
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

def load_or3_dataset():
    """从pickle文件加载OR3数据集"""
    dataset_path = DATA_DIR / "OR3_dataset.pkl"
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

def get_dataset(dataset_type=None):
    """获取指定测试集的数据
    
    Args:
        dataset_type: 数据集类型，如果为None则使用全局DATASET_TYPE
    """
    if dataset_type is None:
        dataset_type = DATASET_TYPE
    
    if dataset_type == 'weibull':
        datasets = load_weibull_dataset()
        return datasets

    elif dataset_type == 'or3':
        datasets = load_or3_dataset()
        return datasets

    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'weibull' or 'or3'")

SYS_PROMPT = "You are an expert in online bin packing algorithms. Your goal is to design priority functions that minimize the number of bins needed."

CORE_SNIPPETS = [
    # ——单特征—————————————————————————————
    "return -(bins - item)",                          # best-fit
    "return bins - item",                             # worst-fit
    "return np.ones_like(bins)",                      # first-fit
    # ——阈值控制—————————————————————————————
    textwrap.dedent("""
    rem = bins - item
    tight = (rem >= 0) & (rem < 0.2*bins.max())
    return np.where(tight, -rem, -1e6)               
    """),                                            # only tight fits
    # ——非线性/混合———————————————————————————
    "return (bins**2)/(item + 1e-6)",
    "return (bins - item)/(bins + 1e-6)",
    "return safe_div((bins-item)**2, bins)",          # high penalty far away
    # ——简单混合策略————————————————————————————
    "return bins * (bins - item)",                    # 容量和剩余空间的乘积
    "return (bins - item) * bins**0.5",              # 平方根权重
]

# -------------------------- 核心bin packing算法 ----------------------------- #
def get_valid_bin_indices(item, bins):
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]

def online_binpack(items, bins, priority_func):
    """Performs online binpacking of items into bins using priority_func."""
    # Track which items are added to each bin
    packing = [[] for _ in bins]
    
    # Add items to bins
    for item in items:
        # Extract bins that have sufficient space to fit item
        valid_bin_indices = get_valid_bin_indices(item, bins)
        if len(valid_bin_indices) == 0:
            continue  # Skip if no valid bins
            
        # Score each bin based on heuristic
        try:
            priorities = priority_func(item, bins[valid_bin_indices])
        except Exception as e:
            priorities = np.ones(len(valid_bin_indices))  # Default: equal priority
        
        # Add item to bin with highest priority
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    
    # Remove unused bins from packing
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


def evaluate_all_instances(priority_func, dataset_type='or3'):
    """
    Evaluate heuristic function on ALL instances of the dataset.
    仿照FunSearch的评估方法
    """
    start_time = time.time()  # 添加开始时间记录
    # List storing number of bins used for each instance.
    num_bins = []
    dataset = get_dataset(dataset_type)
    # Perform online binpacking for each instance.
    for i, instance_name in enumerate(ALL_INSTANCES):
        # 检查是否超时
        if time.time() - start_time > 100:
            print(f"Timeout after {i} instances, elapsed: {time.time() - start_time:.2f} seconds")
            return 5000.0  # 直接返回惩罚值
            
        try:
            capacity = dataset[instance_name]['capacity']
            items = dataset[instance_name]['items']

            # Create num_items bins so there will always be space for all items
            bins = np.array([capacity for _ in range(dataset[instance_name]['num_items'])])

            # Pack items into bins
            _, bins_packed = online_binpack(items, bins, priority_func)
            
            # Count number of used bins
            bins_used = (bins_packed != capacity).sum()
            num_bins.append(bins_used)
            
        except Exception as e:
            print(f"Warning: Failed to evaluate instance {instance_name}: {e}")
            # 给一个惩罚值
            num_bins.append(5000)
    
    return np.mean(num_bins)

# -------------------------- 适应度函数 ----------------------------- #
def eval(code_str):
    """
    Evaluate a priority function on ALL instances of the dataset
    """
    code_str = repair(code_str)
    
    # Execute the code to get the priority function
    local_env = {"np": np, "safe_div": lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y!=0)}
    try:
        exec(code_str, local_env)
        priority_func = local_env["priority"]
    except Exception as e:
        print(f"Error executing code: {e}")
        return 5000.0  # 返回高惩罚值
    
    dataset_type = DATASET_TYPE
    
    # 始终使用全实例评估
    start_eval = time.time()
    try:
        avg_bins = evaluate_all_instances(priority_func, dataset_type)
        eval_time = time.time() - start_eval

        if eval_time > 540:
            print(f"Warning: Evaluation took too long ({eval_time:.2f} seconds)")
            return 5000.0  # 返回高惩罚值
        return float(avg_bins)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 5000.0  # 返回高惩罚值
    finally:
        end_eval = time.time()
        print(f"Evaluation time: {end_eval - start_eval:.2f} seconds")


def repair(code_input):
    """修复代码字符串，支持单个字符串或字符串列表"""
    
    def repair_single(code_str: str) -> str:
        """修复单个代码字符串"""
        if code_str is None or code_str.strip() == "":
            return """def priority(item, bins):
    return -(bins - item)"""
        
        # 清理代码字符串
        code_str = code_str.strip()
        
        # 如果没有函数定义，尝试包装在priority函数中
        if "def priority(" not in code_str:
            # 检查是否只是一个返回语句
            if code_str.startswith("return "):
                code_str = f"""def priority(item, bins):
    {code_str}"""
            else:
                # 尝试包装整个代码
                code_str = f"""def priority(item, bins):
    # Generated priority function
    {code_str}
    return -(bins - item)"""
        
        # 简单的语法修复
        import re
        # 移除常见的语法错误
        code_str = re.sub(r'\bd\b(?!\w)', 'bins', code_str)  # 替换单独的 'd' 为 'bins'
        
        return code_str
    
    # 处理列表输入
    if isinstance(code_input, list):
        return [repair_single(code) for code in code_input]
    else:
        # 处理单个字符串输入
        return repair_single(code_input)

def create_fallback_genome():
    """当LLM生成失败时，创建一个fallback基因组"""
    fallback_code = """def priority(item, bins):
    return -(bins - item)"""  # Best fit heuristic
    fallback_loss = eval(fallback_code)
    return fallback_code, fallback_loss

# -------------------------- 进化相关接口 ----------------------------- #
def random_priority_func(rng: random.Random) -> str:
    snippet = rng.choice(CORE_SNIPPETS)
    header = "def priority(item, bins):\n    import numpy as np\n"
    body = "    " + snippet.replace("\n", "\n    ")
    return header + body

def seed_pool(n: int, rng: random.Random):
    """Generate initial pool of priority functions"""
    genomes = []
    for _ in range(n):
        code = random_priority_func(rng)
        loss = eval(code)
        genomes.append(Genome(genome=code, loss=loss, extra={}))
    return genomes

def diversity_key(g: Genome):
    """Generate diversity key for genome"""
    return hash(g.genome[:50])

# -------------------------- LLM交互相关 ----------------------------- #
def get_zero_shot_prompt(seed=None, temperature_index=None, call_index=None):
    """Generate zero-shot prompt for bin packing priority function"""
    sys = SYS_PROMPT
    
    # Use seed for reproducible sampling if provided
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
    """Generate evolution prompt based on parent priority functions"""
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
    """Parse LLM response to extract priority function code"""
    content = resp.get("text", "")
    
    # 首先尝试找完整的python代码块
    m = re.search(r'```python(.*?)```', content, re.S)
    if m:
        code = m.group(1).strip()
        return {"genome": code, "usage": resp.get("usage", {"total_tokens": 0})}
    
    # 如果没有找到完整代码块，尝试找不完整的python代码块
    m = re.search(r'```python(.*)', content, re.S)
    if m:
        code = m.group(1).strip()
        return {"genome": code, "usage": resp.get("usage", {"total_tokens": 0})}
    
    # 尝试找def priority函数定义
    m = re.search(r'def priority\([^)]*\):(.*?)(?=\n\n|\n[^\s]|$)', content, re.S)
    if m:
        code = f"def priority{m.group(0)[len('def priority'):]}"
        return {"genome": code, "usage": resp.get("usage", {"total_tokens": 0})}
    
    # 如果都没找到，返回一个简单的best-fit作为fallback
    print(f"Warning: Could not parse response, using fallback. Content: {content[:200]}...")
    fallback_code = """def priority(item, bins):
    return -(bins - item)"""
    return {"genome": fallback_code, "usage": resp.get("usage", {"total_tokens": 0})}

def describe() -> str:
    """生成任务描述"""
    return (
        f"Online bin packing optimization task using ALL {len(ALL_INSTANCES)} instances from the {DATASET_TYPE.upper()} dataset. "
        "The goal is to minimize the average number of bins used across all instances. "
        "This comprehensive evaluation ensures good generalization across diverse item patterns."
    )

# 移除不再需要的全局变量和函数
# 删除 TEST_NAME, TRAIN_INSTANCES, TEST_INSTANCES, USE_MULTI_INSTANCE 等
# 删除 compute_l1_lower_bound() 的无参数调用，因为现在都需要指定实例
