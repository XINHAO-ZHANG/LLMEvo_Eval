"""
LLM Interface Module

This module provides a unified interface for different LLM providers including:
- OpenAI GPT models
- Anthropic Claude models  
- Local Ollama models
- Custom providers
"""

from .api import call_llm, get_llm_provider
from .prompts import build_evolve_prompt
from .zero_shot_eval import evaluate_zero_shot

__all__ = ["call_llm", "get_llm_provider", "build_evolve_prompt", "evaluate_zero_shot"]
