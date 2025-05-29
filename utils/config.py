"""
Configuration Module

This module contains all configuration settings and constants for the Klipstream Analysis project.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load environment variables from .env.yaml file
try:
    if os.path.exists('.env.yaml'):
        with open('.env.yaml', 'r') as f:
            yaml_env = yaml.safe_load(f)
            for key, value in yaml_env.items():
                os.environ[key] = str(value)
except Exception as e:
    print(f"Error loading .env.yaml: {str(e)}")

# Check if running in Cloud environment (Cloud Run or Cloud Functions)
IS_CLOUD_ENV = os.environ.get('K_SERVICE') is not None

# Get base directory from environment variable or use default
BASE_DIR = Path(os.environ.get('BASE_DIR', "/tmp" if IS_CLOUD_ENV else "."))
OUTPUT_DIR = BASE_DIR / "output"
RAW_DIR = OUTPUT_DIR / "Raw"
DOWNLOADS_DIR = BASE_DIR / "downloads"
TEMP_DIR = DOWNLOADS_DIR / "temp"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Whether to use GCS for file storage
USE_GCS = os.environ.get('USE_GCS', 'false').lower() == 'true'

# GCS project ID
GCS_PROJECT = os.environ.get('GCS_PROJECT', 'klipstream')

# Log the configuration being used
print(f"Using base directory: {BASE_DIR} (Cloud Environment: {IS_CLOUD_ENV})")
print(f"Using GCS for file storage: {USE_GCS}")
print(f"GCS Project: {GCS_PROJECT}")

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
ANALYSIS_BUCKET = "klipstream-analysis"

# API Keys
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# GCP Service Account
GCP_SERVICE_ACCOUNT_PATH = os.getenv("GCP_SERVICE_ACCOUNT_PATH")

# Binary paths based on platform
def get_binary_paths():
    """Get the appropriate binary paths based on the platform"""
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        return {
            "twitch_downloader": "./raw_pipeline/bin/TwitchDownloaderCLI_mac",
            "ffmpeg": "./raw_pipeline/bin/ffmpeg_mac"
        }
    elif system == "Windows":
        return {
            "twitch_downloader": "./raw_pipeline/bin/TwitchDownloaderCLI.exe",
            "ffmpeg": "./raw_pipeline/bin/ffmpeg.exe"
        }
    else:  # Linux (Cloud Functions uses Linux)
        # Check if we're in a container environment (like Cloud Functions)
        if os.path.exists("/app/raw_pipeline/bin/TwitchDownloaderCLI"):
            return {
                "twitch_downloader": "/app/raw_pipeline/bin/TwitchDownloaderCLI",
                "ffmpeg": "/usr/bin/ffmpeg"  # Use system ffmpeg to avoid architecture issues
            }
        else:
            return {
                "twitch_downloader": "./raw_pipeline/bin/TwitchDownloaderCLI",
                "ffmpeg": "ffmpeg"  # Use system ffmpeg
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
