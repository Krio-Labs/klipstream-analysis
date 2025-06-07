"""
Cleanup Module

This module handles cleaning up directories before starting the raw pipeline.
"""

import os
import shutil
from pathlib import Path

from utils.config import (
    OUTPUT_DIR,
    DOWNLOADS_DIR,
    DATA_DIR,
    LOGS_DIR,
    create_directories
)
from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("cleanup", "cleanup.log")

def cleanup_directories():
    """
    Remove directories from previous runs and recreate them.
    
    This function:
    1. Removes the downloads, data, and output directories
    2. Recreates the directory structure
    """
    directories_to_clean = [
        OUTPUT_DIR,
        DOWNLOADS_DIR,
        DATA_DIR
    ]

    # Remove directories
    for directory in directories_to_clean:
        if directory.exists():
            try:
                shutil.rmtree(directory)
            except Exception as e:
                logger.error(f"Error removing directory {directory}: {str(e)}")

    # Recreate directory structure
    create_directories()
    
    return True

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    cleanup_directories()
