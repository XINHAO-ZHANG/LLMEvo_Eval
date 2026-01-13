"""
Base Task Interface

This module defines the abstract base class that all optimization tasks should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


class BaseTask(ABC):
    """Abstract base class for optimization tasks"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._data = None
    
    @abstractmethod
    def init(self, n_population: int, rng: Any) -> List[str]:
        """Initialize population with n_population individuals
        
        Args:
            n_population: Number of individuals to generate
            rng: Random number generator
            
        Returns:
            List of genome strings
        """
        pass
    
    @abstractmethod
    def eval(self, genome: str, **kwargs) -> Union[float, Tuple[float, str]]:
        """Evaluate a single genome
        
        Args:
            genome: Genome string to evaluate
            **kwargs: Additional task-specific parameters
            
        Returns:
            Float fitness score, or tuple of (score, feedback_string)
        """
        pass
    
    @abstractmethod
    def get_evolve_prompt(self, parents: List[str], **kwargs) -> str:
        """Generate evolution prompt for LLM
        
        Args:
            parents: List of parent genomes
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def get_zero_shot_prompt(self, **kwargs) -> str:
        """Generate zero-shot evaluation prompt
        
        Args:
            **kwargs: Task-specific parameters
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def repair(self, genomes: List[str]) -> List[str]:
        """Optional repair function for invalid genomes
        
        Args:
            genomes: List of potentially invalid genomes
            
        Returns:
            List of repaired genomes
        """
        # Default implementation: no repair
        return genomes
    
    def load_data(self, data_path: str, **kwargs):
        """Load task-specific data
        
        Args:
            data_path: Path to data directory
            **kwargs: Additional loading parameters
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get task information and metadata
        
        Returns:
            Dictionary with task information
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
            "has_repair": hasattr(self, "repair") and callable(getattr(self, "repair")),
            "data_loaded": self._data is not None
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"