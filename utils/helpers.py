"""
Helper Utilities Module

This module provides common utility functions used across the Klipstream Analysis project.
"""

import re
import os
import platform
import subprocess
from pathlib import Path
import shutil

def extract_video_id(url):
    """
    Extract the video ID from a Twitch VOD URL
    
    Args:
        url (str): Twitch VOD URL
        
    Returns:
        str: Video ID
    """
    # Extract the video ID from the URL
    match = re.search(r'videos/(\d+)', url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid Twitch VOD URL: {url}")

def is_valid_vod_url(url):
    """
    Check if a URL is a valid Twitch VOD URL
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    pattern = r'^https?://(?:www\.)?twitch\.tv/videos/\d+/?(?:\?.*)?$'
    return bool(re.match(pattern, url))

def set_executable_permissions(file_path):
    """
    Set executable permissions for a file on Unix-like systems
    
    Args:
        file_path (str): Path to the file
    """
    if platform.system() != "Windows" and os.path.exists(file_path):
        os.chmod(file_path, 0o755)
        
def run_command(command, check=True, shell=False, env=None):
    """
    Run a shell command and return the result
    
    Args:
        command (list): Command to run
        check (bool, optional): Whether to check for errors. Defaults to True.
        shell (bool, optional): Whether to run in shell. Defaults to False.
        env (dict, optional): Environment variables. Defaults to None.
        
    Returns:
        subprocess.CompletedProcess: Result of the command
    """
    # Set shell=True for Windows to properly execute .exe files
    if platform.system() == "Windows":
        shell = True
        
    # Run the command
    return subprocess.run(
        command,
        check=check,
        shell=shell,
        env=env or os.environ.copy()
    )

def copy_file(src, dst):
    """
    Copy a file from source to destination
    
    Args:
        src (str or Path): Source file path
        dst (str or Path): Destination file path
        
    Returns:
        Path: Path to the copied file
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    # Create destination directory if it doesn't exist
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Copy the file
    return Path(shutil.copy2(src_path, dst_path))
