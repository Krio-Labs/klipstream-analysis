"""
Configuration Module

This module contains all configuration settings and constants for the Klipstream Analysis project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
OUTPUT_DIR = Path("Output")
RAW_DIR = OUTPUT_DIR / "Raw"
DOWNLOADS_DIR = Path("downloads")
TEMP_DIR = DOWNLOADS_DIR / "temp"
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")

# Raw file directories
RAW_VIDEOS_DIR = RAW_DIR / "Videos"
RAW_AUDIO_DIR = RAW_DIR / "Audio"
RAW_TRANSCRIPTS_DIR = RAW_DIR / "Transcripts"
RAW_WAVEFORMS_DIR = RAW_DIR / "Waveforms"
RAW_CHAT_DIR = RAW_DIR / "Chat"

# Analysis output directories
ANALYSIS_DIR = OUTPUT_DIR / "Analysis"
ANALYSIS_AUDIO_DIR = ANALYSIS_DIR / "Audio"
ANALYSIS_CHAT_DIR = ANALYSIS_DIR / "Chat"

# GCS bucket names
VODS_BUCKET = "klipstream-vods-raw"
TRANSCRIPTS_BUCKET = "klipstream-transcripts"
CHATLOGS_BUCKET = "klipstream-chatlogs"

# API Keys
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Binary paths based on platform
def get_binary_paths():
    """Get the appropriate binary paths based on the platform"""
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        return {
            "twitch_downloader": "./TwitchDownloaderCLI_mac",
            "ffmpeg": "./ffmpeg_mac"
        }
    elif system == "Windows":
        return {
            "twitch_downloader": "./TwitchDownloaderCLI.exe",
            "ffmpeg": "./ffmpeg.exe"
        }
    else:  # Linux (Cloud Functions uses Linux)
        # Check if we're in a container environment (like Cloud Functions)
        if os.path.exists("/app/bin/TwitchDownloaderCLI"):
            return {
                "twitch_downloader": "/app/bin/TwitchDownloaderCLI",
                "ffmpeg": "/usr/bin/ffmpeg"
            }
        else:
            return {
                "twitch_downloader": "./TwitchDownloaderCLI",
                "ffmpeg": "./ffmpeg"
            }

# Get binary paths
BINARY_PATHS = get_binary_paths()

# Create necessary directories
def create_directories():
    """Create all necessary directories"""
    directories = [
        OUTPUT_DIR,
        RAW_DIR,
        RAW_VIDEOS_DIR,
        RAW_AUDIO_DIR,
        RAW_TRANSCRIPTS_DIR,
        RAW_WAVEFORMS_DIR,
        RAW_CHAT_DIR,
        DOWNLOADS_DIR,
        TEMP_DIR,
        DATA_DIR,
        LOGS_DIR,
        ANALYSIS_DIR,
        ANALYSIS_AUDIO_DIR,
        ANALYSIS_CHAT_DIR
    ]

    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
