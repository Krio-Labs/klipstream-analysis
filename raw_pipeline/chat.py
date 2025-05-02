"""
Chat Module

This module handles downloading and processing Twitch chat data.
"""

import logging
import re
import subprocess
import json
import csv
import platform
from pathlib import Path
import os
import shutil

from utils.config import (
    RAW_CHAT_DIR,
    DATA_DIR,
    BINARY_PATHS
)
from utils.logging_setup import setup_logger
from utils.helpers import (
    extract_video_id,
    is_valid_vod_url,
    run_command
)

# Set up logger
logger = setup_logger("chat", "twitch_chat_downloader.log")

def download_chat(video_id, output_dir=None):
    """
    Downloads chat using TwitchDownloaderCLI and converts it to CSV format.
    
    Args:
        video_id (str): Twitch VOD ID
        output_dir (Path, optional): Directory to save output files. If not provided, uses RAW_CHAT_DIR.
        
    Returns:
        Path: Path to the saved CSV file
    """
    # Determine output directory
    if output_dir is None:
        output_dir = RAW_CHAT_DIR
    else:
        output_dir = Path(output_dir)
    
    # Create file paths
    json_path = output_dir / f"{video_id}_chat.json"
    csv_path = output_dir / f"{video_id}_chat.csv"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading chat for VOD {video_id} to {output_dir}")
    
    # Get the appropriate CLI command for the current OS
    cli_command = BINARY_PATHS["twitch_downloader"]
    
    # Download chat in JSON format
    logger.info(f"Downloading chat for VOD: {video_id}")
    download_cmd = [
        cli_command,
        "chatdownload",
        "--id",
        video_id,
        "--output",
        str(json_path)
    ]
    
    # Set DOTNET_BUNDLE_EXTRACT_BASE_DIR environment variable
    env = os.environ.copy()
    if platform.system().lower() == 'darwin':  # macOS specific fix
        temp_dir = Path.home() / '.twitch_downloader_temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        env['DOTNET_BUNDLE_EXTRACT_BASE_DIR'] = str(temp_dir)
    
    try:
        # Add shell=True for Windows to properly execute the .exe
        is_windows = platform.system().lower() == 'windows'
        subprocess.run(
            download_cmd,
            check=True,
            shell=is_windows,
            env=env  # Add environment variables
        )
        logger.info(f"Chat downloaded successfully to {json_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download chat: {e}")
        logger.error("Make sure TwitchDownloaderCLI is installed and in the correct location:")
        logger.error("Windows: Place TwitchDownloaderCLI.exe in the same directory or add to PATH")
        logger.error("Linux/MacOS: Place TwitchDownloaderCLI in the same directory and ensure it's executable (chmod +x)")
        raise
    except FileNotFoundError:
        logger.error("TwitchDownloaderCLI not found!")
        logger.error("Please download it from: https://github.com/lay295/TwitchDownloader/releases")
        logger.error("Windows: Place TwitchDownloaderCLI.exe in the same directory or add to PATH")
        logger.error("Linux/MacOS: Place TwitchDownloaderCLI in the same directory and ensure it's executable (chmod +x)")
        raise
    
    # Convert JSON to CSV
    convert_json_to_csv(json_path, csv_path)
    logger.info(f"Chat converted to CSV: {csv_path}")
    
    # Optionally remove JSON file to save space
    json_path.unlink()
    
    # Also save a copy to the data directory for backward compatibility
    data_csv_path = DATA_DIR / f"{video_id}_chat.csv"
    shutil.copy2(csv_path, data_csv_path)
    logger.info(f"Chat data also copied to: {data_csv_path} for backward compatibility")
    
    return csv_path

def convert_json_to_csv(json_path, csv_path):
    """
    Converts the downloaded JSON chat file to CSV format.
    
    Args:
        json_path (Path): Path to input JSON file
        csv_path (Path): Path to output CSV file
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'username', 'message'])
        
        for comment in chat_data['comments']:
            time_in_seconds = comment['content_offset_seconds']
            username = comment['commenter']['display_name']
            message = comment['message']['body']
            writer.writerow([time_in_seconds, username, message])

def download_chat_data(video_id):
    """
    Download chat data for a video ID
    
    Args:
        video_id (str): Twitch VOD ID
        
    Returns:
        dict: Dictionary with chat_file path
    """
    try:
        logger.info(f"Downloading chat for {video_id}...")
        
        # Download chat
        chat_file = download_chat(video_id, RAW_CHAT_DIR)
        
        logger.info(f"Chat data saved to: {chat_file}")
        
        return {
            "chat_file": chat_file
        }
    except Exception as e:
        logger.error(f"Error downloading chat: {str(e)}")
        raise
