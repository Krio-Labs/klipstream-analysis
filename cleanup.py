#!/usr/bin/env python3
"""
Cleanup script to remove directories and files from previous runs
"""

import os
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_directories():
    """Remove directories from previous runs"""
    directories_to_clean = ['downloads', 'data', 'outputs']
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                logger.info(f"Successfully removed directory: {directory}")
            except Exception as e:
                logger.error(f"Error removing directory {directory}: {str(e)}")
        else:
            logger.info(f"Directory does not exist, skipping: {directory}")
    
    # Recreate empty directories
    for directory in directories_to_clean:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created empty directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting cleanup process...")
    cleanup_directories()
    logger.info("Cleanup completed.")
