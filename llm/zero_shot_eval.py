"""
llm_ops.zero_shot_eval
======================

Evaluate a model's zero-shot capability across a temperature sweep (0.0–1.0,
step 0.2). Supports different task interfaces via signature inspection.
"""

from typing import Dict, List, Callable, Union, Any, Protocol, Optional
import numpy as np
from .api import call_llm

class Task(Protocol):
    """Minimal task interface required for zero-shot evaluation."""
    def parse_response(self, resp: Dict[str, Any]) -> Any:
        ...
    def eval(self, genome: Any, split: str = "train") -> float:
        ...

def evaluate_zero_shot(
    prompt_func: Callable[[int, int, int], str],
    model: str,
    task: Task,
    trials_per_temp: int = 2,
    temp_step: float = 0.2,
    base_seed: int = 42,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate zero-shot performance across a temperature sweep.

    Args:
        prompt_func: callable(seed, temperature_index, call_index) → prompt
        model: model name string
        task: task module with parse_response and eval
        trials_per_temp: number of trials per temperature
        temp_step: temperature step size
        base_seed: base random seed for reproducibility

    Returns:
        {temperature: {'mean', 'std', 'count', 'scores'}}
    """
    temperatures = np.arange(0.0, 1.0 + temp_step, temp_step)
    results = {}

    for temp_idx, temp in enumerate(temperatures):
        temp_results = []

        for call_idx in range(trials_per_temp):
            try:
                if hasattr(prompt_func, '__code__') and 'seed' in prompt_func.__code__.co_varnames:
                    prompt = prompt_func(seed=base_seed, temperature_index=temp_idx, call_index=call_idx)
                else:
                    prompt = prompt_func()

                response = call_llm(prompt, model=model, temperature=temp)
                parsed = task.parse_response(response)
                print(f"Parsed response: {parsed}")
                genome = parsed.get("genome", "")

                if genome:
                    import inspect
                    eval_params = list(inspect.signature(task.eval).parameters.keys())

                    if len(eval_params) == 1:
                        score = task.eval(genome)
                    elif 'split' in eval_params:
                        score = task.eval(genome, split='train')
                    else:
                        # promptopt: has task parameter
                        current_task = getattr(task, 'CURRENT_TASK',
                                               getattr(task, 'DEFAULT_EVAL_TASK', 'sum'))
                        score = task.eval(genome, task=current_task)

                    if not np.isnan(score) and not np.isinf(score):
                        temp_results.append(score)
                        print(f"  Temp {temp:.1f}, Call {call_idx+1}: {score:.4f}")
                    else:
                        print(f"  Temp {temp:.1f}, Call {call_idx+1}: Invalid score")
                else:
                    print(f"  Temp {temp:.1f}, Call {call_idx+1}: Failed to parse")

            except Exception as e:
                print(f"  Temp {temp:.1f}, Call {call_idx+1}: Error - {e}")

        if temp_results:
            results[temp] = {
                'mean': np.mean(temp_results),
                'std': np.std(temp_results),
                'count': len(temp_results),
                'scores': temp_results
            }
        else:
            results[temp] = {
                'mean': np.nan,
                'std': np.nan,
                'count': 0,
                'scores': []
            }

    return results

def evaluate_batch_zero_shot(
    prompts: List[Union[str, List[Dict[str, str]]]],
    model: str,
    task: Task,
    trials_per_temp: int = 5,
    temp_step: float = 0.2,
    max_tokens: int = 4800,
    custom_parser: Optional[Callable[[Dict[str, Any]], Any]] = None,
    custom_fitness: Optional[Callable[[Any], float]] = None,
) -> Dict[float, Dict[str, float]]:
    """Evaluate multiple prompts and merge results across temperatures."""
    all_results = []
    for prompt in prompts:
        result = evaluate_zero_shot(
            prompt=prompt,
            model=model,
            task=task,
            trials_per_temp=trials_per_temp,
            temp_step=temp_step,
            max_tokens=max_tokens,
            custom_parser=custom_parser,
            custom_fitness=custom_fitness
        )
        all_results.append(result)

    merged = {}
    temperatures = np.arange(0.0, 1.01, temp_step)

    for temp in temperatures:
        temp_scores = [r[temp]['mean'] for r in all_results]
        merged[float(temp)] = {
            'mean': float(np.mean(temp_scores)),
            'std': float(np.std(temp_scores)),
            'max': float(np.max(temp_scores)),
            'min': float(np.min(temp_scores))
        }

    return merged
