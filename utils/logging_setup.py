"""
Logging Setup Module

This module provides a centralized logging configuration for the Klipstream Analysis project.
"""

import logging
import os
from pathlib import Path
from .config import LOGS_DIR

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with file and console handlers
    
    Args:
        name (str): Logger name
        log_file (str, optional): Log file path. If None, uses name.log
        level (int, optional): Logging level. Defaults to INFO.
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file is None:
        log_file = LOGS_DIR / f"{name}.log"
    else:
        log_file = LOGS_DIR / log_file
        
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
