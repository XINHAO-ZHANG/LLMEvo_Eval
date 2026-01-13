"""
Base LLM Provider Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
    
    @abstractmethod
    def generate(self, 
                 prompt: str, 
                 model: str,
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 **kwargs) -> Dict[str, Any]:
        """Generate text using the LLM
        
        Returns:
            Dict containing:
            - text: Generated text
            - usage: Token usage statistics
            - model: Model name used
            - success: Boolean indicating success
            - error: Error message if failed
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(provider={self.provider_name})"