"""
Logging Setup Module

This module provides a centralized logging configuration for the Klipstream Analysis project.
"""

import logging
from .config import LOGS_DIR

def setup_logger(name, log_file=None, level=logging.CRITICAL):
    """
    Set up extremely minimal logging - only critical errors and main progress

    Args:
        name (str): Logger name
        log_file (str, optional): Log file path. If None, uses name.log
        level (int, optional): Logging level. Defaults to CRITICAL for minimal output.

    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

    # Create logger
    logger = logging.getLogger(name)

    # Prevent propagation to avoid duplicate messages
    logger.propagate = False

    # Only main, processor, and convex_client show progress, everything else is silent
    if name in ['main', 'processor', 'convex_client']:
        logger.setLevel(logging.INFO)
        console_level = logging.INFO
    else:
        logger.setLevel(logging.CRITICAL)  # Essentially silent
        console_level = logging.CRITICAL

    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Create simple formatter for console (no timestamps, no logger names)
    console_formatter = logging.Formatter('%(message)s')

    # Create console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Only create file handler for main components
    if name in ['main', 'processor', 'convex_client']:
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_file is None:
            log_file = LOGS_DIR / f"{name}.log"
        else:
            log_file = LOGS_DIR / log_file

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
