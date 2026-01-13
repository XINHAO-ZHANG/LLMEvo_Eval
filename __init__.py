"""
LLMEvo: LLM-driven Evolutionary Optimization Framework

A general-purpose optimization platform that combines Large Language Models (LLMs)
with evolutionary algorithms for automated, intelligent search and optimization.
"""

__version__ = "0.1.0"
__author__ = "XINHAO-ZHANG"
__email__ = "zhangxinhao672@gmail.com"

# Core imports
from evolve import run_evolve, RunStats
from tasks import get_task, list_tasks
from llm import call_llm

__all__ = [
    "run_evolve", 
    "RunStats", 
    "get_task", 
    "list_tasks", 
    "call_llm",
    "__version__"
]