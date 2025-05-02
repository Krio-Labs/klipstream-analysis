"""
Downloader Module

This module handles downloading Twitch videos and extracting audio.
"""

import os
import platform
import logging
import asyncio
import json
import re
import shutil
from pathlib import Path
import subprocess
from tqdm import tqdm

from utils.config import (
    RAW_VIDEOS_DIR,
    RAW_AUDIO_DIR,
    BINARY_PATHS,
    DOWNLOADS_DIR,
    TEMP_DIR
)
from utils.logging_setup import setup_logger
from utils.helpers import (
    extract_video_id,
    set_executable_permissions,
    run_command
)

# Set up logger
logger = setup_logger("downloader", "twitch_downloader.log")

class TwitchVideoDownloader:
    """Class for downloading Twitch videos and extracting audio"""

    def __init__(self):
        """Initialize the downloader"""
        self._setup_environment()
        self._verify_ffmpeg_exists()

    def _setup_environment(self):
        """Set up the environment for downloading"""
        # Create cache directory
        cache_dir = Path.home() / ".cache" / "twitch_downloader"
        cache_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created cache directory at: {cache_dir}")

        # Log current working directory
        logger.info(f"Current working directory: {os.getcwd()}")

        # Log directory contents
        logger.info(f"Directory contents: {os.listdir()}")

        # Set up for macOS
        if platform.system() == "Darwin":
            # Set executable permissions for binaries
            set_executable_permissions(BINARY_PATHS["twitch_downloader"])
            set_executable_permissions(BINARY_PATHS["ffmpeg"])
            logger.info(f"Set executable permissions for {BINARY_PATHS['twitch_downloader']}")
            logger.info(f"Set executable permissions for {BINARY_PATHS['ffmpeg']}")

            # Create temp directory for .NET bundle extraction
            temp_dir = Path.home() / '.dotnet' / 'bundle_extract'
            temp_dir.mkdir(parents=True, exist_ok=True)
            os.environ['DOTNET_BUNDLE_EXTRACT_BASE_DIR'] = str(temp_dir)
            logger.info(f"Set DOTNET_BUNDLE_EXTRACT_BASE_DIR to {temp_dir}")

    def _verify_ffmpeg_exists(self):
        """Verify that ffmpeg is installed and accessible"""
        try:
            # Try system ffmpeg first
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                logger.info("ffmpeg found")
            else:
                # Try the local ffmpeg
                result = subprocess.run(
                    [BINARY_PATHS["ffmpeg"], "-version"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    logger.info("Local ffmpeg found")
                else:
                    raise FileNotFoundError("ffmpeg not found")

            # Also check for ffprobe
            result = subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                logger.info("System ffprobe found")
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg.")
            raise FileNotFoundError("ffmpeg not found. Please install ffmpeg.")

    def extract_video_id(self, url):
        """
        Extract the video ID from a Twitch VOD URL

        Args:
            url (str): Twitch VOD URL

        Returns:
            str: Video ID
        """
        video_id = extract_video_id(url)
        logger.info(f"Extracted video ID: {video_id}")
        return video_id

    def get_video_metadata(self, video_id):
        """
        Get metadata for a Twitch video

        Args:
            video_id (str): Twitch video ID

        Returns:
            dict: Video metadata
        """
        # Run the TwitchDownloaderCLI command to get video info
        command = [
            BINARY_PATHS["twitch_downloader"],
            "info",
            "--id", video_id,
            "--format", "Raw"
        ]

        try:
            # Run the command and capture the output
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the raw output
            output = result.stdout.strip()

            # Create a simple dictionary with the available info
            video_info = {
                "id": video_id
            }

            # Try to extract some basic info from the output
            if "Title:" in output:
                title_match = re.search(r"Title: (.+)", output)
                if title_match:
                    video_info["title"] = title_match.group(1)

            if "Channel:" in output:
                channel_match = re.search(r"Channel: (.+)", output)
                if channel_match:
                    video_info["user_name"] = channel_match.group(1)

            if "Duration:" in output:
                duration_match = re.search(r"Duration: (.+)", output)
                if duration_match:
                    video_info["duration"] = duration_match.group(1)

            return video_info
        except Exception as e:
            logger.error(f"Error getting video metadata: {str(e)}")
            return {}

    async def download_video(self, video_id):
        """
        Download a Twitch video

        Args:
            video_id (str): Twitch video ID

        Returns:
            Path: Path to the downloaded video file
        """
        logger.info(f"Downloading video for {video_id}...")

        # Define output path
        video_file = RAW_VIDEOS_DIR / f"{video_id}.mp4"

        # Create command
        command = [
            BINARY_PATHS["twitch_downloader"],
            "videodownload",
            "--id", video_id,
            "-o", str(video_file),
            "--quality", "worst",
            "--threads", "16",
            "--temp-path", str(TEMP_DIR)
        ]

        # Log the command
        logger.info(f"Running download command: {' '.join(command)}")

        # Log current directory contents
        logger.info(f"Current directory contents: {os.listdir()}")

        # Check if CLI file exists and has correct permissions
        cli_file_exists = os.path.exists(BINARY_PATHS["twitch_downloader"])
        cli_file_permissions = oct(os.stat(BINARY_PATHS["twitch_downloader"]).st_mode) if cli_file_exists else "N/A"
        logger.info(f"CLI file exists: {cli_file_exists}")
        logger.info(f"CLI file permissions: {cli_file_permissions}")

        # Run the command
        logger.info(f"Running command: {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Error downloading video: {stderr.decode()}")
            raise RuntimeError(f"Error downloading video: {stderr.decode()}")

        # Clean up temporary directory
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir(exist_ok=True, parents=True)
            logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")

        return video_file

    async def extract_audio(self, video_file, video_id):
        """
        Extract audio from a video file

        Args:
            video_file (Path): Path to the video file
            video_id (str): Twitch video ID

        Returns:
            Path: Path to the extracted audio file
        """
        # Define output path
        audio_file = RAW_AUDIO_DIR / f"audio_{video_id}.wav"

        # Get video duration
        duration_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_file)
        ]

        try:
            result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            logger.info(f"Video duration: {duration} seconds")
        except Exception as e:
            logger.warning(f"Could not get video duration: {str(e)}")
            duration = 0

        # Create command
        command = [
            BINARY_PATHS["ffmpeg"],
            "-i", str(video_file),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16 kHz
            "-ac", "1",  # Mono
            "-threads", "10",  # Use 10 threads
            "-y",  # Overwrite output file
            str(audio_file)
        ]

        # Log the command
        logger.info(f"Running command: {' '.join(command)}")

        # Create progress bar
        pbar = tqdm(total=100, desc="Converting to WAV")

        # Run the command
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Process output to update progress bar
        while True:
            line = await process.stderr.readline()
            if not line:
                break

            line_str = line.decode('utf-8', errors='replace')
            logger.info(f"Process output: {line_str.strip()}")

            # Update progress bar based on time
            if "time=" in line_str and duration > 0:
                time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line_str)
                if time_match:
                    h, m, s = map(float, time_match.groups())
                    current_time = h * 3600 + m * 60 + s
                    progress = min(100, (current_time / duration) * 100)
                    pbar.update(progress - pbar.n)

        # Wait for process to complete
        await process.wait()
        pbar.close()

        # Get process output
        stdout, stderr = await process.communicate()

        # Log stderr
        logger.info(f"Process stderr:\n{stderr.decode('utf-8', errors='replace')}")

        if process.returncode != 0:
            logger.error(f"Error extracting audio: {stderr.decode('utf-8', errors='replace')}")
            raise RuntimeError(f"Error extracting audio: {stderr.decode('utf-8', errors='replace')}")

        return audio_file

    async def process_video(self, url):
        """
        Process a Twitch video URL

        Args:
            url (str): Twitch VOD URL

        Returns:
            dict: Dictionary with video_id, video_file, and audio_file
        """
        # Extract video ID
        video_id = self.extract_video_id(url)

        # Get video metadata
        video_info = self.get_video_metadata(video_id)

        # Download video
        video_file = await self.download_video(video_id)

        # Extract audio
        audio_file = await self.extract_audio(video_file, video_id)

        # Return results
        return {
            "video_id": video_id,
            "video_file": video_file,
            "audio_file": audio_file,
            "twitch_info": video_info
        }
