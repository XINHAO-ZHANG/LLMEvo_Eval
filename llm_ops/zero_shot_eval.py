"""
llm_ops.zero_shot_eval
=====================

用于评估模型的 zero-shot 能力的模块。
通过不同 temperature 值（0.0-1.0，间隔0.2）评估模型表现。
支持不同任务的特定评估逻辑。
"""

from typing import Dict, List, Callable, Union, Any, Protocol, Optional
import numpy as np
from .api import call_llm

class Task(Protocol):
    """任务接口协议"""
    def parse_response(self, resp: Dict[str, Any]) -> Any:
        """解析LLM响应"""
        ...
    def eval(self, genome: Any, split: str = "train") -> float:
        """评估基因组的性能"""
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
    评估zero-shot性能
    
    参数:
        prompt_func: 生成提示的函数，应该接受seed, temperature_index, call_index参数
        model: 模型名称
        task: 任务模块
        trials_per_temp: 每个温度的试验次数
        temp_step: 温度步长
        base_seed: 基础随机种子
    
    返回:
        Dict[float, Dict[str, float]]: {
            temperature: {
                'mean': float,  # 平均分
                'std': float,   # 标准差
                'max': float,   # 最高分
                'min': float    # 最低分
            }
        }
    """
    temperatures = np.arange(0.0, 1.0 + temp_step, temp_step)
    results = {}
    
    for temp_idx, temp in enumerate(temperatures):
        temp_results = []
        
        for call_idx in range(trials_per_temp):
            try:
                # 传递种子信息给prompt函数
                if hasattr(prompt_func, '__code__') and 'seed' in prompt_func.__code__.co_varnames:
                    prompt = prompt_func(seed=base_seed, temperature_index=temp_idx, call_index=call_idx)
                else:
                    prompt = prompt_func()
                
                response = call_llm(prompt, model=model, temperature=temp)
                parsed = task.parse_response(response)
                print(f"Parsed response: {parsed}")  # 调试输出
                genome = parsed.get("genome", "")
                
                if genome:
                    # 检查eval函数的参数签名，适配不同任务的eval接口
                    import inspect
                    eval_sig = inspect.signature(task.eval)
                    eval_params = list(eval_sig.parameters.keys())
                    
                    if len(eval_params) == 1:
                        # 只有一个参数的eval函数 (如其他任务)
                        score = task.eval(genome)
                    elif 'split' in eval_params:
                        # 有split参数的eval函数 (如superglue)
                        score = task.eval(genome, split='train')
                    else:
                        # 多参数的eval函数 (如promptopt)
                        # 使用CURRENT_TASK和默认split
                        if hasattr(task, 'CURRENT_TASK'):
                            current_task = task.CURRENT_TASK
                        else:
                            current_task = getattr(task, 'DEFAULT_EVAL_TASK', 'sum')
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
    """批量评估多个prompt"""
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
    
    # 合并所有结果
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