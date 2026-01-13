"""
Provider module for different LLM backends
"""

from .base import BaseLLMProvider
from .openai import OpenAIProvider

# Register available providers
PROVIDERS = {
    "openai": OpenAIProvider,
}

def get_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """Get a provider instance by name"""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDERS.keys())}")
    
    return PROVIDERS[provider_name](**kwargs)

__all__ = ["BaseLLMProvider", "OpenAIProvider", "get_provider", "PROVIDERS"]