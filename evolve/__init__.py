"""
Evolutionary Algorithm Core Module

This module contains the core evolutionary algorithm components including:
- Main evolutionary loop
- Population database management
- Parallel execution utilities
"""

from evolve.loop import run_evolve, RunStats
from evolve.db import get_db, Genome
from evolve.executor import batch_eval

__all__ = ["run_evolve", "RunStats", "get_db", "Genome", "batch_eval"]