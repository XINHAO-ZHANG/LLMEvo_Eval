"""
Prompt Optimisation task
------------------------
Genome = str (prompt).  Fitness = 1 - accuracy on a fixed eval set.
"""

from __future__ import annotations

import random
from typing import List, Any, Dict
from tqdm import tqdm
from evolve_core.db import Genome
from llm_ops.api import call_llm
from tasks.utils import compute_rouge, compute_sari

# ========== 配置与数据集路径（可通过外部参数传入） ==========
DEFAULT_INIT_MODELS = ["openai/gpt-3.5-turbo", "openai/gpt-4o"]
DEFAULT_N_INIT_PER_MODEL = 3
DEFAULT_EVAL_TASK = "sum"  # "sum" for summarization, "sim" for simplification
DEFAULT_EVAL_SET = "test"  # "test" or "valid"

# 全局变量，存储当前任务类型
CURRENT_TASK = DEFAULT_EVAL_TASK

def configure(cfg=None):
    """配置当前任务类型"""
    global CURRENT_TASK
    if cfg and hasattr(cfg, 'task_type'):
        CURRENT_TASK = cfg.task_type
    print(f"PromptOpt configured with task: {CURRENT_TASK}")

# ========== 数据集加载 ==========
def load_eval_data(task=DEFAULT_EVAL_TASK, split=DEFAULT_EVAL_SET):
    # 获取项目根目录
    import os
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    if task == "sum":
        # SAMSum - 如果请求的split不存在，使用test
        available_splits = ["test", "valid"]
        if split not in available_splits:
            print(f"Warning: Split '{split}' not found for sum task, using 'test' instead")
            split = "test"
            
        src_path = project_root / "data" / "promptopt" / "sum" / "sam" / f"{split}.src"
        tgt_path = project_root / "data" / "promptopt" / "sum" / "sam" / f"{split}.tgt"
        
        # 检查文件是否存在
        if not src_path.exists() or not tgt_path.exists():
            print(f"Warning: Files for split '{split}' not found, using 'test' instead")
            split = "test"
            src_path = project_root / "data" / "promptopt" / "sum" / "sam" / f"{split}.src"
            tgt_path = project_root / "data" / "promptopt" / "sum" / "sam" / f"{split}.tgt"
        
        with open(src_path) as fsrc, open(tgt_path) as ftgt:
            srcs = [l.strip() for l in fsrc]
            tgts = [l.strip() for l in ftgt]
        return srcs, tgts
        
    elif task == "sim":
        # ASSET - 如果请求的split不存在，使用test
        available_splits = ["test", "dev"]
        if split not in available_splits:
            print(f"Warning: Split '{split}' not found for sim task, using 'test' instead")
            split = "test"
        
        # 将valid映射到dev
        if split == "valid":
            split = "dev"
            
        src_path = project_root / "data" / "promptopt" / "sim" / "asset" / split / f"asset.{split}.src"
        
        # 检查文件是否存在
        if not src_path.exists():
            print(f"Warning: Files for split '{split}' not found, using 'test' instead")
            split = "test"
            src_path = project_root / "data" / "promptopt" / "sim" / "asset" / split / f"asset.{split}.src"
        
        refs = []
        for i in range(10):
            ref_path = project_root / "data" / "promptopt" / "sim" / "asset" / split / f"asset.{split}.simp.{i}"
            with open(ref_path) as f:
                refs.append([l.strip() for l in f])
        srcs = [l.strip() for l in open(src_path)]
        # refs: list[list[str]] -> 转置为 list of list of refs
        refs = list(map(list, zip(*refs)))
        return srcs, refs
    else:
        raise ValueError(f"Unknown eval task: {task}")

# ========== 进化接口 ==========
def seed_pool(n: int, rng: random.Random, init_models=None, n_per_model=None, prompt_task=None, eval_set=None) -> List[Genome]:
    """
    从对应任务的prompts.txt文件中读取初始prompt种群。
    如果文件中没有评估分数，则进行评估并保存结果。
    """
    prompt_task = prompt_task or DEFAULT_EVAL_TASK
    eval_set = eval_set or DEFAULT_EVAL_SET
    
    # 获取项目根目录
    import os
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    # 根据任务类型加载对应的prompts.txt
    if prompt_task == "sum":
        prompts_path = project_root / "data" / "promptopt" / "sum" / "sam" / "prompts.txt"
    elif prompt_task == "sim":
        prompts_path = project_root / "data" / "promptopt" / "sim" / "asset" / "prompts.txt"
    else:
        raise ValueError(f"Unknown prompt task: {prompt_task}")
    
    # 读取prompts.txt文件
    try:
        with open(prompts_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")
    
    prompts_with_scores = []
    prompts_to_evaluate = []
    
    # 解析每行：如果有tab分隔的分数就使用，否则需要评估
    for line in lines:
        if '\t' in line:
            parts = line.split('\t')
            prompt = parts[0]
            try:
                score = float(parts[1])
                prompts_with_scores.append((prompt, score))
            except ValueError:
                # 如果分数解析失败，加入待评估列表
                prompts_to_evaluate.append(prompt)
        else:
            # 没有分数，需要评估
            prompts_to_evaluate.append(line)
    
    # 评估没有分数的prompt
    if prompts_to_evaluate:
        print(f"Evaluating {len(prompts_to_evaluate)} prompts for {prompt_task} task...")
        for prompt in tqdm(prompts_to_evaluate, desc="Evaluating prompts"):
            score = eval(prompt, prompt_task, eval_set)
            prompts_with_scores.append((prompt, score))
        
        # 保存更新后的文件
        with open(prompts_path, 'w', encoding='utf-8') as f:
            for prompt, score in prompts_with_scores:
                f.write(f"{prompt}\t{score:.6f}\n")
        print(f"Updated {prompts_path} with evaluation scores")
    
    # 去重（基于prompt文本）
    seen = set()
    unique_prompts = []
    for prompt, score in prompts_with_scores:
        if prompt and prompt not in seen:
            unique_prompts.append((prompt, score))
            seen.add(prompt)
    
    # 随机打乱并取前n个
    rng.shuffle(unique_prompts)
    selected_prompts = unique_prompts[:n]
    
    # 如果prompts不够n个，就重复使用
    if len(selected_prompts) < n:
        selected_prompts = (selected_prompts * ((n // len(selected_prompts)) + 1))[:n]
    
    # 创建Genome对象
    genomes = []
    for i, (prompt, score) in enumerate(selected_prompts):
        genomes.append(Genome(genome=prompt, loss=score, extra={"source": "prompts.txt", "index": i}))
    
    return genomes

def eval(prompt: str, task=DEFAULT_EVAL_TASK, split=DEFAULT_EVAL_SET) -> float:
    """
    评估prompt在指定任务和数据集上的表现，返回loss（越小越好）。
    """
    srcs, tgts_or_refs = load_eval_data(task, split)
    
    # 随机采样25条进行评测
    import random
    if len(srcs) > 25:
        indices = random.sample(range(len(srcs)), 25)
        eval_srcs = [srcs[i] for i in indices]
        if task == "sum":
            eval_tgts = [tgts_or_refs[i] for i in indices]
        else:  # sim
            eval_tgts = [tgts_or_refs[i] for i in indices]
    else:
        eval_srcs = srcs
        eval_tgts = tgts_or_refs
    
    preds = []
    for src in tqdm(eval_srcs, desc=f"Evaluating {task} prompts using gpt-4o-mini"):
        resp = call_llm(prompt + "\n" + src, model="openai/gpt-4o-mini", max_tokens=1280)
        out = parse_response(resp)["genome"]
        toks =parse_response(resp).get("usage", {}).get("total_tokens", 0)
        print(f"Prompt: {prompt[:50]}... | Tokens: {toks} | Response: {out[:50]}...")
        preds.append(out)
    if task == "sum":
        scores = compute_rouge(preds, eval_tgts)
        return 1.0 - scores["rougeL"]  # 以rougeL为主
    elif task == "sim":
        scores = compute_sari(eval_srcs, preds, eval_tgts)
        return 1.0 - scores["sari"] / 100.0
    else:
        return 1.0

def repair(prompts: List[str]) -> List[str]:
    # 去重、去空
    seen = set(); out = []
    for p in prompts:
        p = p.strip()
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

def parse_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """解析LLM响应，提取prompt文本"""
    if "text" in resp:
        text = resp["text"].strip()
        
        # 移除开头和结尾的引号
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # 移除开头的空格
        text = text.strip()
        
        # 如果还有引号包围，再去掉一层
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        return {"genome": text, "usage": resp.get("usage", {"total_tokens": 0})}
    
    return {"genome": str(resp), "usage": {"total_tokens": 0}}

def get_zero_shot_prompt(task=None):
    """
    根据具体任务生成zero-shot prompt
    """
    task = task or CURRENT_TASK
    sys = "You are a prompt optimization expert. Your goal is to design effective prompts for language models."
    
    if task == "sum":
        user = """Please design a prompt for text summarization task. 
        
The task is to summarize dialogues from the SAMSum dataset. The input will be conversations between people, and the output should be a concise summary capturing the key points.

Requirements:
- The prompt should guide the model to produce concise, accurate summaries
- Focus on the main events, decisions, and outcomes in the conversation
- Avoid unnecessary details while preserving important information
- The summary should be coherent and well-structured

Please return only the prompt text that will be used to instruct the language model."""
    
    elif task == "sim":
        user = """Please design a prompt for text simplification task.
        
The task is to simplify complex English sentences to make them more understandable for non-native speakers or people with reading difficulties. The input will be complex sentences from the ASSET dataset.

Requirements:
- The prompt should guide the model to simplify vocabulary and sentence structure
- Preserve the original meaning while making it more accessible
- Use simpler words, shorter sentences, and clearer structure
- Maintain grammatical correctness and natural flow
- The output should be significantly easier to read than the input

Please return only the prompt text that will be used to instruct the language model."""
    
    else:
        user = "Please design a prompt for the downstream task."
    
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ]

def get_evolve_prompt(sampled_parents: List[Genome], task=None):
    """
    生成进化prompt，根据任务类型提供具体指导
    """
    task = task or CURRENT_TASK
    if task == "sum":
        sys = """You are a prompt optimization expert specializing in text summarization. 
        Given several prompts and their performance scores, create a better prompt for dialogue summarization."""
        task_desc = "The task is to summarize dialogues from conversations. Lower score means better performance."
    elif task == "sim":
        sys = """You are a prompt optimization expert specializing in text simplification. 
        Given several prompts and their performance scores, create a better prompt for sentence simplification."""
        task_desc = "The task is to simplify complex sentences for better readability. Lower score means better performance."
    else:
        sys = "You are a prompt optimization expert. Given several prompts and their scores, propose a better prompt."
        task_desc = "Lower score means better performance."
    
    parent_block = "\n".join([f"Prompt: {g.genome}\nScore: {g.loss:.4f}" for g in sampled_parents])
    
    user = f"""{task_desc}

Here are previous prompts and their performance scores:
{parent_block}

Analyze the patterns in successful prompts and create a new, improved prompt by:
1. Identifying what makes the better-performing prompts effective
2. Combining successful elements from multiple prompts
3. Adding new techniques or phrasings that might improve performance
4. Ensuring the prompt is clear, specific, and actionable

Please return ONLY the new prompt text (not in a list, just the plain text that will be used to instruct the language model)."""
    
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ]

def describe() -> str:
    return "Prompt optimization for text summarization or simplification. Genome is a prompt string. Fitness is 1 - metric (ROUGE/SARI) on a fixed eval set."


