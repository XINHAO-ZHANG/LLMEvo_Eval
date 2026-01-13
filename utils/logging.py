"""
Logging utilities
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any


def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_generation(log_file: Path, generation_data: Dict[str, Any]):
    """Log generation data to JSONL file
    
    Args:
        log_file: Path to log file
        generation_data: Generation data to log
    """
    with open(log_file, 'a') as f:
        f.write(json.dumps(generation_data) + '\n')