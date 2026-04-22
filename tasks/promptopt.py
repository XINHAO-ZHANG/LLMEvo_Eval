"""
Prompt Optimisation task
------------------------
Genome = str (prompt).  Fitness = 1 - accuracy on a fixed eval set.
"""

from __future__ import annotations

import os
from pathlib import Path
import hashlib
import random
from typing import List, Any, Dict
from tqdm import tqdm
from evolve.db import Genome
from llm.api import call_llm
from tasks.utils import compute_rouge, compute_sari

# ---------- config & dataset paths ----------
DEFAULT_INIT_MODELS = ["openai/gpt-3.5-turbo", "openai/gpt-4o"]
DEFAULT_N_INIT_PER_MODEL = 3
DEFAULT_EVAL_TASK = "sum"  # "sum" for summarization, "sim" for simplification
DEFAULT_EVAL_SET = "test"
PROJECT_ROOT = Path(__file__).parent.parent

# current task type (set by configure())
CURRENT_TASK = DEFAULT_EVAL_TASK

# evaluation cache: prompt_hash -> (loss, metadata)
_EVAL_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}

def _prompt_hash(prompt: str, task: str, split: str) -> str:
    content = f"{prompt}|{task}|{split}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def configure(cfg=None):
    global CURRENT_TASK
    print(f"[DEBUG] configure() called with cfg type: {type(cfg)}")

    if cfg is not None:
        if hasattr(cfg, 'tasks') and hasattr(cfg.tasks, 'promptopt'):
            if hasattr(cfg.tasks.promptopt, 'eval_task'):
                CURRENT_TASK = cfg.tasks.promptopt.eval_task
                print(f"PromptOpt configured with task: {CURRENT_TASK} (from cfg.tasks.promptopt.eval_task)")
                return

        if hasattr(cfg, 'eval_task'):
            CURRENT_TASK = cfg.eval_task
            print(f"PromptOpt configured with task: {CURRENT_TASK} (from cfg.eval_task)")
            return

        print(f"[DEBUG] Could not find eval_task in config, using default")

    print(f"PromptOpt configured with default task: {CURRENT_TASK}")

# ---------- dataset loading ----------
def load_eval_data(task=CURRENT_TASK, split=DEFAULT_EVAL_SET):
    if task == "sum":
        available_splits = ["test", "valid"]
        if split not in available_splits:
            print(f"Warning: Split '{split}' not found for sum task, using 'test' instead")
            split = "test"

        src_path = PROJECT_ROOT / "data" / "promptopt" / "sum" / "sam" / f"{split}.src"
        tgt_path = PROJECT_ROOT / "data" / "promptopt" / "sum" / "sam" / f"{split}.tgt"

        if not src_path.exists() or not tgt_path.exists():
            print(f"Warning: Files for split '{split}' not found, using 'test' instead")
            split = "test"
            src_path = PROJECT_ROOT / "data" / "promptopt" / "sum" / "sam" / f"{split}.src"
            tgt_path = PROJECT_ROOT / "data" / "promptopt" / "sum" / "sam" / f"{split}.tgt"

        with open(src_path) as fsrc, open(tgt_path) as ftgt:
            srcs = [l.strip() for l in fsrc]
            tgts = [l.strip() for l in ftgt]
        return srcs, tgts

    elif task == "sim":
        available_splits = ["test", "dev"]
        if split not in available_splits:
            print(f"Warning: Split '{split}' not found for sim task, using 'test' instead")
            split = "test"

        if split == "valid":
            split = "dev"

        src_path = PROJECT_ROOT / "data" / "promptopt" / "sim" / "asset" / split / f"asset.{split}.src"
        tgt_path = PROJECT_ROOT / "data" / "promptopt" / "sim" / "asset" / split / f"asset.{split}.tgt"

        if not src_path.exists() or not tgt_path.exists():
            print(f"Warning: Files for split '{split}' not found, using 'test' instead")
            split = "test"
            src_path = PROJECT_ROOT / "data" / "promptopt" / "sim" / "asset" / split / f"asset.{split}.src"
            tgt_path = PROJECT_ROOT / "data" / "promptopt" / "sim" / "asset" / split / f"asset.{split}.tgt"

        with open(src_path) as f:
            srcs = [l.strip() for l in f]

        with open(tgt_path) as f:
            refs = []
            for line in f:
                ref_list = [ref.strip() for ref in line.strip().split('\t')]
                refs.append(ref_list)

        return srcs, refs
    else:
        raise ValueError(f"Unknown eval task: {task}")

# ---------- evolution interface ----------
def seed_pool(n: int, rng: random.Random, init_models=None, n_per_model=None, prompt_task=None, eval_set=None) -> List[Genome]:
    """
    Load initial prompt population from the task's prompts.txt file.
    Evaluates and caches scores for any prompts that lack them.
    """
    prompt_task = prompt_task or DEFAULT_EVAL_TASK
    eval_set = eval_set or DEFAULT_EVAL_SET

    if prompt_task == "sim":
        cache_keys_to_remove = [k for k in _EVAL_CACHE if "sim" in k]
        for key in cache_keys_to_remove:
            del _EVAL_CACHE[key]
        if cache_keys_to_remove:
            print(f"Cleared {len(cache_keys_to_remove)} cached sim evaluations")

    if prompt_task == "sum":
        prompts_path = PROJECT_ROOT / "data" / "promptopt" / "sum" / "sam" / "prompts.txt"
    elif prompt_task == "sim":
        prompts_path = PROJECT_ROOT / "data" / "promptopt" / "sim" / "asset" / "prompts.txt"
    else:
        raise ValueError(f"Unknown prompt task: {prompt_task}")

    try:
        with open(prompts_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

    prompts_with_scores = []
    prompts_to_evaluate = []

    for line in lines:
        if '\t' in line:
            parts = line.split('\t')
            prompt = parts[0]
            try:
                score = float(parts[1])
                prompts_with_scores.append((prompt, score))
            except ValueError:
                prompts_to_evaluate.append(prompt)
        else:
            prompts_to_evaluate.append(line)

    if prompts_to_evaluate:
        print(f"Evaluating {len(prompts_to_evaluate)} prompts for {prompt_task} task...")
        for prompt in tqdm(prompts_to_evaluate, desc="Evaluating prompts"):
            score = eval(prompt, prompt_task, eval_set)
            prompts_with_scores.append((prompt, score))

        with open(prompts_path, 'w', encoding='utf-8') as f:
            for prompt, score in prompts_with_scores:
                f.write(f"{prompt}\t{score:.6f}\n")
        print(f"Updated {prompts_path} with evaluation scores")

    seen = set()
    unique_prompts = []
    for prompt, score in prompts_with_scores:
        if prompt and prompt not in seen:
            unique_prompts.append((prompt, score))
            seen.add(prompt)

    rng.shuffle(unique_prompts)
    selected_prompts = unique_prompts[:n]

    if len(selected_prompts) < n:
        selected_prompts = (selected_prompts * ((n // len(selected_prompts)) + 1))[:n]

    genomes = []
    for i, (prompt, score) in enumerate(selected_prompts):
        genomes.append(Genome(genome=prompt, loss=score, extra={"source": "prompts.txt", "index": i}))

    return genomes

def eval(prompt: str, task=CURRENT_TASK, split=DEFAULT_EVAL_SET) -> float:
    """
    Evaluate a prompt on the given task and split. Returns loss (lower = better).
    Results are cached to avoid redundant LLM calls.
    """
    cache_key = _prompt_hash(prompt, task, split)
    if cache_key in _EVAL_CACHE:
        cached_loss, metadata = _EVAL_CACHE[cache_key]
        print(f"Cache hit for prompt: {prompt[:50]}... | Cached loss: {cached_loss:.6f}")
        return cached_loss

    print(f"Cache miss - evaluating prompt: {prompt[:50]}...")

    srcs, tgts_or_refs = load_eval_data(task, split)

    import random
    if len(srcs) > 25:
        indices = random.sample(range(len(srcs)), 25)
        eval_srcs = [srcs[i] for i in indices]
        eval_tgts = [tgts_or_refs[i] for i in indices]
    else:
        eval_srcs = srcs
        eval_tgts = tgts_or_refs

    preds = []
    total_tokens = 0
    for src in tqdm(eval_srcs, desc=f"Evaluating {task} prompts using gpt-4o-mini"):
        resp = call_llm(prompt + "\n" + src, model="openai/gpt-4o-mini", max_tokens=1280)
        out = parse_response(resp)["genome"]
        toks = parse_response(resp).get("usage", {}).get("total_tokens", 0)
        total_tokens += toks
        print(f"Prompt: {prompt[:50]}... | Tokens: {toks} | Response: {out[:50]}...")
        preds.append(out)

    if task == "sum":
        scores = compute_rouge(preds, eval_tgts)
        loss = 1.0 - scores["rougeL"]
        metric_name = "rougeL"
        metric_value = scores["rougeL"]
        print(f"ROUGE-1: {scores['rouge1']:.4f}, ROUGE-2: {scores['rouge2']:.4f}, ROUGE-L: {scores['rougeL']:.4f}")
    elif task == "sim":
        scores = compute_sari(eval_srcs, preds, eval_tgts)
        loss = 1.0 - scores["sari"] / 100.0
        metric_name = "sari"
        metric_value = scores["sari"]
        print(f"SARI: {scores['sari']:.4f}")
    else:
        print(f"Unknown task: {task}. Cannot evaluate.")
        loss = 1.0
        metric_name = "unknown"
        metric_value = 0.0

    metadata = {
        "task": task,
        "split": split,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "total_tokens": total_tokens,
        "num_samples": len(eval_srcs)
    }
    _EVAL_CACHE[cache_key] = (loss, metadata)

    print(f"Evaluation complete | Loss: {loss:.6f} | {metric_name}: {metric_value:.4f} | Tokens: {total_tokens}")
    return loss

def repair(prompts: List[str]) -> List[str]:
    seen = set(); out = []
    for p in prompts:
        p = p.strip()
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

def parse_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Extract prompt text from LLM response, stripping surrounding quotes."""
    if "text" in resp:
        text = resp["text"].strip()

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        text = text.strip()

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        return {"genome": text, "usage": resp.get("usage", {"total_tokens": 0})}

    return {"genome": str(resp), "usage": {"total_tokens": 0}}

def get_zero_shot_prompt(task=None, seed=None, temperature_index=None, call_index=None):
    """Generate a zero-shot prompt for the given task type."""
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
    """Generate evolution prompt with task-specific guidance."""
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

def clear_eval_cache():
    global _EVAL_CACHE
    cache_size = len(_EVAL_CACHE)
    _EVAL_CACHE.clear()
    print(f"Cleared evaluation cache ({cache_size} entries)")

def get_cache_stats():
    total_entries = len(_EVAL_CACHE)
    if total_entries == 0:
        return {"total_entries": 0, "tasks": {}}

    stats = {"total_entries": total_entries, "tasks": {}}
    for cache_key, (loss, metadata) in _EVAL_CACHE.items():
        task = metadata.get("task", "unknown")
        if task not in stats["tasks"]:
            stats["tasks"][task] = {"count": 0, "total_tokens": 0}
        stats["tasks"][task]["count"] += 1
        stats["tasks"][task]["total_tokens"] += metadata.get("total_tokens", 0)

    return stats
