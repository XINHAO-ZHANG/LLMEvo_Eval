"""
OpenAI Provider Implementation
"""

import openai
from typing import Dict, Any, Optional
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT models provider"""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        super().__init__("openai")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base
        )
    
    def generate(self, 
                 prompt: str, 
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 **kwargs) -> Dict[str, Any]:
        """Generate text using OpenAI API"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return {
                "text": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "success": True
            }
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "success": False
            }