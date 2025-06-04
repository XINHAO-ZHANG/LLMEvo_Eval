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
    
    def fitness(self, parsed_output: Any) -> float:
        """计算适应度分数"""
        ...

def evaluate_zero_shot(
    prompt: Union[str, List[Dict[str, str]]],
    model: str,
    task: Task,
    trials_per_temp: int = 2,
    temp_step: float = 0.2,
    max_tokens: int = 4096,
    custom_parser: Optional[Callable[[Dict[str, Any]], Any]] = None,
    custom_fitness: Optional[Callable[[Any], float]] = None,
) -> Dict[float, Dict[str, float]]:
    """
    用不同temperature评估模型的zero-shot能力
    
    参数:
        prompt: 提示词或消息列表
        model: 模型名称
        task: 任务对象，必须实现parse_response和fitness方法
        trials_per_temp: 每个temperature测试次数
        temp_step: temperature间隔（默认0.2）
        max_tokens: 最大token数
        custom_parser: 可选的自定义解析函数，覆盖task的parse_response
        custom_fitness: 可选的自定义适应度函数，覆盖task的fitness
    
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
    results = {}
    
    # 使用自定义函数或任务默认函数
    parse_fn = custom_parser or task.parse_response
    fitness_fn = custom_fitness or task.fitness
    
    # 生成temperature列表：0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    temperatures = np.arange(0.0, 1.01, temp_step)
    
    for temp in temperatures:
        scores = []
        for _ in range(trials_per_temp):
            try:
                # 调用LLM
                output = call_llm(
                    prompt_or_msgs=prompt,
                    model=model,
                    temperature=float(temp),
                    max_tokens=max_tokens,
                )
                
                # 解析输出
                parsed = parse_fn(output)
                print(parsed)
                
                # 对于TSP等任务，parsed是一个dict，需要取出genome
                if isinstance(parsed, dict) and "genome" in parsed:
                    genome = parsed["genome"]

                
                # 计算分数
                score = fitness_fn(genome)
                scores.append(score)
            except Exception as e:
                print(f"Error at temperature {temp}: {e}")
                continue  # 跳过错误的结果，而不是添加0分
        
        results[float(temp)] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'max': float(np.max(scores)),
            'min': float(np.min(scores))
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