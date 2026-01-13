"""
Task Definition Module

This module contains implementations of various optimization tasks including:
- TSP (Traveling Salesman Problem)
- Bin Packing
- Prompt Optimization
- Symbolic Regression
"""

from .base import BaseTask

# Import task implementations
from . import tsp
from . import bin_packing
from . import promptopt
from . import symboreg_oscillator1
from . import symboreg_oscillator2

# Task registry
TASKS = {
    "tsp": tsp,
    "bin_packing": bin_packing,
    "promptopt": promptopt, 
    "symboreg_oscillator1": symboreg_oscillator1,
    "symboreg_oscillator2": symboreg_oscillator2,
}

def get_task(task_name: str):
    """Get a task module by name
    
    Args:
        task_name: Name of the task
        
    Returns:
        Task module
        
    Raises:
        ValueError: If task name is not found
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
    
    return TASKS[task_name]

def list_tasks():
    """List all available tasks
    
    Returns:
        List of task names
    """
    return list(TASKS.keys())

__all__ = ["BaseTask", "get_task", "list_tasks", "TASKS"]