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
import time
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

        # Check if the file already exists
        if video_file.exists():
            logger.info(f"Video file already exists: {video_file}")
            return video_file

        # Create command
        command = [
            BINARY_PATHS["twitch_downloader"],
            "videodownload",
            "--id", video_id,
            "-o", str(video_file),
            "--quality", "720p",  # Use 720p for faster downloads and less storage
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

        # Additional debugging for Cloud Run
        if not cli_file_exists:
            logger.error(f"TwitchDownloaderCLI not found at: {BINARY_PATHS['twitch_downloader']}")
            # List contents of the bin directory
            bin_dir = Path(BINARY_PATHS["twitch_downloader"]).parent
            if bin_dir.exists():
                logger.info(f"Contents of {bin_dir}: {list(bin_dir.iterdir())}")
            else:
                logger.error(f"Bin directory does not exist: {bin_dir}")
            raise FileNotFoundError(f"TwitchDownloaderCLI not found at: {BINARY_PATHS['twitch_downloader']}")

        # Test if the binary can be executed (with improved error handling)
        try:
            test_result = subprocess.run(
                [BINARY_PATHS["twitch_downloader"], "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Log detailed output for debugging
            logger.info(f"TwitchDownloaderCLI test - Return code: {test_result.returncode}")
            logger.info(f"TwitchDownloaderCLI test - Stdout: {test_result.stdout[:200]}...")
            logger.info(f"TwitchDownloaderCLI test - Stderr: {test_result.stderr[:200]}...")

            # TwitchDownloaderCLI --help returns exit code 1 even when working correctly
            # Check if the output contains expected help text instead of just return code
            if test_result.stdout and ("TwitchDownloaderCLI" in test_result.stdout or "Usage:" in test_result.stdout):
                logger.info("TwitchDownloaderCLI binary test passed (help text found)")
            elif test_result.returncode == 1 and "TwitchDownloaderCLI" in test_result.stderr:
                logger.info("TwitchDownloaderCLI binary test passed (binary responds correctly)")
            else:
                logger.warning(f"TwitchDownloaderCLI test unexpected result. Return code: {test_result.returncode}")
                logger.warning("Continuing anyway - will test during actual download")

        except subprocess.TimeoutExpired:
            logger.warning("TwitchDownloaderCLI test timed out - continuing anyway")
        except Exception as e:
            logger.warning(f"Error testing TwitchDownloaderCLI binary: {str(e)} - continuing anyway")
            # Don't raise an exception for binary test failures

        # Create progress bar
        pbar = tqdm(total=100, desc=f"Downloading VOD {video_id}")

        # Variables to track progress
        last_update_time = time.time()
        progress_detected = False

        # Run the command with timeout
        logger.info(f"Running command: {' '.join(command)}")
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        except Exception as e:
            logger.error(f"Failed to start video download process: {str(e)}")
            raise RuntimeError(f"Failed to start video download process: {str(e)}")

        # Set a timeout for the entire download process (30 minutes)
        download_timeout = 30 * 60  # 30 minutes in seconds
        start_time = time.time()

        # Process output to update progress bar
        while True:
            # Check for timeout
            current_time = time.time()
            if current_time - start_time > download_timeout:
                logger.error(f"Video download timed out after {download_timeout} seconds")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                raise RuntimeError(f"Video download timed out after {download_timeout} seconds")

            try:
                # Use asyncio.wait_for to add a timeout to readline
                line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
            except asyncio.TimeoutError:
                logger.warning("No output from video download process for 30 seconds, checking if process is still alive")
                if process.returncode is not None:
                    logger.info("Process has completed")
                    break
                continue

            if not line:
                # Check if stderr has any data
                try:
                    err_line = await asyncio.wait_for(process.stderr.readline(), timeout=5)
                except asyncio.TimeoutError:
                    err_line = None

                if not err_line:
                    break
                else:
                    err_str = err_line.decode('utf-8', errors='replace').strip()
                    logger.info(f"Process stderr: {err_str}")
                    continue

            line_str = line.decode('utf-8', errors='replace').strip()

            # Only log non-progress output to avoid duplicate logging
            progress_line = False

            # Look for progress information in the output
            # Pattern 1: "Downloaded: 10.5%"
            progress_match = re.search(r'Downloaded:\s+(\d+(?:\.\d+)?)%', line_str)
            if progress_match:
                progress = float(progress_match.group(1))
                pbar.update(progress - pbar.n)
                logger.info(f"Download progress: {progress}%")
                progress_detected = True
                progress_line = True
                continue

            # Pattern 2: "Downloading segment X/Y"
            segment_match = re.search(r'Downloading segment (\d+)/(\d+)', line_str)
            if segment_match:
                current, total = map(int, segment_match.groups())
                progress = min(100, (current / total) * 100)
                pbar.update(progress - pbar.n)
                logger.info(f"Download progress: {progress}% (segment {current}/{total})")
                progress_detected = True
                progress_line = True
                continue

            # Pattern 3: "Progress: X%"
            progress_match2 = re.search(r'Progress:\s+(\d+(?:\.\d+)?)%', line_str)
            if progress_match2:
                progress = float(progress_match2.group(1))
                pbar.update(progress - pbar.n)
                logger.info(f"Download progress: {progress}%")
                progress_detected = True
                progress_line = True
                continue

            # Pattern 4: "X% complete"
            progress_match3 = re.search(r'(\d+(?:\.\d+)?)%\s+complete', line_str)
            if progress_match3:
                progress = float(progress_match3.group(1))
                pbar.update(progress - pbar.n)
                logger.info(f"Download progress: {progress}%")
                progress_detected = True
                progress_line = True
                continue

            # Pattern 5: "Downloading: X%"
            progress_match4 = re.search(r'Downloading:\s+(\d+(?:\.\d+)?)%', line_str)
            if progress_match4:
                progress = float(progress_match4.group(1))
                pbar.update(progress - pbar.n)
                logger.info(f"Download progress: {progress}%")
                progress_detected = True
                progress_line = True
                continue

            # Log non-progress output
            if not progress_line:
                logger.info(f"Process output: {line_str}")

            # Fallback: If we haven't detected progress in the output but the process is still running,
            # update the progress bar slightly every 5 seconds to show activity
            current_time = time.time()
            if not progress_detected and (current_time - last_update_time) > 5:
                # Small increment (0.5%) to show activity
                pbar.update(0.5)
                last_update_time = current_time
                logger.info("No progress information detected, showing activity indicator")

            # Reset progress_detected flag after each line to ensure we detect if progress stops being reported
            progress_detected = False

        # Close the progress bar
        pbar.close()

        # Wait for process to complete and get return code
        return_code = await process.wait()

        if return_code != 0:
            # If we have an error, try to get any remaining stderr content
            stderr_data = await process.stderr.read()
            stderr_str = stderr_data.decode('utf-8', errors='replace')
            logger.error(f"Error downloading video: {stderr_str}")
            raise RuntimeError(f"Error downloading video: {stderr_str}")

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

        # Check if the file already exists
        if audio_file.exists():
            logger.info(f"Audio file already exists: {audio_file}")
            return audio_file

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

        # Set timeout for audio extraction (15 minutes)
        audio_timeout = 15 * 60  # 15 minutes in seconds
        start_time = time.time()

        # Process output to update progress bar
        while True:
            # Check for timeout
            current_time = time.time()
            if current_time - start_time > audio_timeout:
                logger.error(f"Audio extraction timed out after {audio_timeout} seconds")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                raise RuntimeError(f"Audio extraction timed out after {audio_timeout} seconds")

            try:
                line = await asyncio.wait_for(process.stderr.readline(), timeout=30)
            except asyncio.TimeoutError:
                logger.warning("No output from audio extraction process for 30 seconds, checking if process is still alive")
                if process.returncode is not None:
                    logger.info("Audio extraction process has completed")
                    break
                continue

            if not line:
                break

            line_str = line.decode('utf-8', errors='replace')

            # Update progress bar based on time
            if "time=" in line_str and duration > 0:
                time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line_str)
                if time_match:
                    h, m, s = map(float, time_match.groups())
                    current_time = h * 3600 + m * 60 + s
                    progress = min(100, (current_time / duration) * 100)
                    pbar.update(progress - pbar.n)
                    # Log progress updates
                    logger.info(f"Audio conversion progress: {progress:.2f}%")
                else:
                    # Log non-progress output
                    logger.info(f"Process output: {line_str.strip()}")
            else:
                # Log non-progress output
                logger.info(f"Process output: {line_str.strip()}")

        # Wait for process to complete
        return_code = await process.wait()
        pbar.close()

        if return_code != 0:
            # If we have an error, try to get any remaining stderr content
            stderr_data = await process.stderr.read()
            stderr_str = stderr_data.decode('utf-8', errors='replace')
            logger.error(f"Error extracting audio: {stderr_str}")
            raise RuntimeError(f"Error extracting audio: {stderr_str}")

        logger.info("Audio extraction completed successfully")

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
