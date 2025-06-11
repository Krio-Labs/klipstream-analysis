"""
Downloader Module

This module handles downloading Twitch videos and extracting audio.
"""

import os
import platform
import asyncio
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
    TEMP_DIR
)
from utils.logging_setup import setup_logger
from utils.helpers import (
    extract_video_id,
    set_executable_permissions
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

        # Set up for macOS
        if platform.system() == "Darwin":
            # Set executable permissions for binaries
            set_executable_permissions(BINARY_PATHS["twitch_downloader"])
            set_executable_permissions(BINARY_PATHS["ffmpeg"])

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
            if result.returncode != 0:
                # Try the local ffmpeg
                result = subprocess.run(
                    [BINARY_PATHS["ffmpeg"], "-version"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode != 0:
                    raise FileNotFoundError("ffmpeg not found")

            # Also check for ffprobe
            subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                text=True,
                check=False
            )
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg not found. Please install ffmpeg.")

    def extract_video_id(self, url):
        """
        Extract the video ID from a Twitch VOD URL

        Args:
            url (str): Twitch VOD URL

        Returns:
            str: Video ID
        """
        return extract_video_id(url)

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
            # Prepare environment for macOS .NET bundle extraction
            env = os.environ.copy()
            if platform.system() == "Darwin":
                temp_dir = Path.home() / '.dotnet' / 'bundle_extract'
                temp_dir.mkdir(parents=True, exist_ok=True)
                env['DOTNET_BUNDLE_EXTRACT_BASE_DIR'] = str(temp_dir)

            # Run the command and capture the output
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=env
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

    async def download_video_with_fallback(self, video_id: str) -> Path:
        """
        Download video with multiple fallback strategies optimized for this system
        """
        from utils.resource_optimizer import resource_optimizer

        # Get system-optimized strategies
        strategies = resource_optimizer.get_download_strategies()

        # Log system summary on first run
        resource_optimizer.log_system_summary()

        for i, strategy in enumerate(strategies):
            try:
                # Only log the first attempt to reduce verbosity
                if i == 0:
                    logger.info(f"🚀 Downloading with {strategy['quality']} quality, {strategy['threads']} threads")
                return await self.download_video_with_strategy(video_id, strategy)
            except Exception as e:
                logger.warning(f"❌ Strategy {i+1} failed: {str(e)}")
                if i == len(strategies) - 1:
                    raise e
                # Only log fallback attempts at debug level
                logger.debug(f"🔄 Trying next strategy: {strategy['description']}")

        raise RuntimeError("All download strategies failed")

    async def download_video_with_strategy(self, video_id: str, strategy: dict) -> Path:
        """
        Download video with specific strategy parameters
        """
        video_file = RAW_VIDEOS_DIR / f"{video_id}.mp4"

        # Update Convex status to "Downloading" before starting download
        try:
            from utils.convex_client_updated import ConvexManager
            convex_manager = ConvexManager()

            # Print to terminal immediately so user sees it
            print(f"📊 Updating status to 'Downloading' for video {video_id}...", flush=True)

            success = convex_manager.update_video_status(video_id, "Downloading")
            if success:
                print(f"✅ Status updated to 'Downloading'", flush=True)
                logger.info(f"📊 Updated Convex status to 'Downloading' for video {video_id}")
            else:
                print(f"⚠️  Status update failed", flush=True)
                logger.warning(f"Failed to update Convex status to 'Downloading'")
        except Exception as e:
            print(f"❌ Status update error: {e}", flush=True)
            logger.warning(f"Failed to update Convex status: {e}")

        # Check if the file already exists and has content (caching mechanism)
        if video_file.exists():
            file_size = video_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Only use cached file if it has content (> 1MB to be safe)
            if file_size > 1024 * 1024:  # At least 1MB
                logger.info(f"📹 Using cached video file: {file_size_mb:.1f} MB ({video_file})")
                return video_file
            else:
                logger.warning(f"📹 Cached file is too small ({file_size} bytes), re-downloading...")
                # Remove the invalid cached file
                video_file.unlink()
                logger.info(f"📹 Removed invalid cached file: {video_file}")

        # Create command with strategy-specific settings
        command = [
            BINARY_PATHS["twitch_downloader"],
            "videodownload",
            "--id", video_id,
            "-o", str(video_file),
            "--quality", strategy["quality"],
            "--threads", str(strategy["threads"]),
            "--bandwidth", "-1",  # Unlimited bandwidth
            "--temp-path", str(TEMP_DIR),
            "--collision", "Overwrite"
        ]

        return await self._execute_download_command(command, video_file)

    async def _execute_download_command(self, command: list, video_file: Path) -> Path:
        """
        Execute the TwitchDownloaderCLI command with system-optimized performance monitoring
        """
        from utils.resource_optimizer import resource_optimizer

        # Verify CLI exists before running
        if not os.path.exists(BINARY_PATHS["twitch_downloader"]):
            raise FileNotFoundError(f"TwitchDownloaderCLI not found at {BINARY_PATHS['twitch_downloader']}")

        # Test the binary before using it (minimal check)
        try:
            from api.services.subprocess_wrapper import subprocess_wrapper
            subprocess_wrapper.test_twitch_cli()
        except Exception:
            # Continue anyway - the actual download will fail if there's a real issue
            pass

        # Get system-optimized timeout and buffer settings
        timeout_settings = resource_optimizer.get_timeout_settings()
        buffer_settings = resource_optimizer.get_buffer_settings()

        download_timeout = timeout_settings['download_timeout']
        read_buffer_size = buffer_settings['read_buffer_size']
        read_timeout = buffer_settings['read_timeout']

        # Variables to track progress and create visual progress bar
        last_update_time = time.time()
        stage_progress = {1: 0, 2: 0, 3: 0, 4: 0}  # Track progress for each stage

        # Create progress bar for video download with cleaner format
        from tqdm import tqdm
        try:
            pbar = tqdm(total=100, desc="📹 Video Download", unit="%",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}%',
                       ncols=80, leave=True)
        except Exception:
            # Fallback to a simple progress tracker if tqdm fails
            class SimpleProgress:
                def __init__(self):
                    self.n = 0
                    self.desc = "📹 Video Download"
                def set_description(self, desc):
                    self.desc = desc
                    print(f"\r{desc}", end="", flush=True)
                def refresh(self):
                    print(f"\r{self.desc}: {self.n:.0f}%", end="", flush=True)
                def close(self):
                    print()  # New line
            pbar = SimpleProgress()

        # Log optimization settings at debug level only
        logger.debug(f"⚙️  Using optimized settings: {download_timeout}s timeout, {read_buffer_size}B buffer")

        start_time = time.time()

        # Prepare environment for macOS .NET bundle extraction
        env = os.environ.copy()
        if platform.system() == "Darwin":
            temp_dir = Path.home() / '.dotnet' / 'bundle_extract'
            temp_dir.mkdir(parents=True, exist_ok=True)
            env['DOTNET_BUNDLE_EXTRACT_BASE_DIR'] = str(temp_dir)

        # Run the command with timeout
        try:
            # Check if we're running in FastAPI context and use appropriate subprocess method
            try:
                from api.services.subprocess_wrapper import subprocess_wrapper
                # For FastAPI context, we need to use a different approach
                # Since we need streaming output, we'll set up the environment properly
                # and then use asyncio.create_subprocess_exec with the correct environment
                api_env = subprocess_wrapper.base_env.copy()
                api_env.update(env)  # Add our environment variables
                cwd = subprocess_wrapper.working_directory

                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=api_env,
                    cwd=cwd
                )
            except ImportError:
                # Fallback to regular subprocess for standalone execution
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
        except Exception as e:
            raise RuntimeError(f"Failed to start video download process: {str(e)}")

        # Process output with aggressive buffering for maximum performance
        buffer = b''
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

            # Read from stdout with aggressive buffering to catch rapid updates
            line_received = False

            try:
                # Read multiple bytes at once to catch rapid updates (using system-optimized buffer size)
                data = await asyncio.wait_for(process.stdout.read(read_buffer_size), timeout=read_timeout)
                if data:
                    buffer += data
                    # Process all complete lines in the buffer
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line_str = line.decode('utf-8', errors='replace').strip()
                        if line_str:
                            # Parse TwitchDownloaderCLI progress and update visual progress bar
                            progress_updated = self._update_progress_bar(line_str, pbar, stage_progress)
                            if not progress_updated:
                                # Only print non-progress lines to avoid clutter
                                if not any(keyword in line_str.lower() for keyword in ['status', 'downloading', 'verifying', 'finalizing']):
                                    print(line_str, flush=True)
                            last_update_time = current_time
                            line_received = True

                    # Also check for partial lines that might contain progress
                    if buffer and b'[STATUS]' in buffer:
                        partial_line = buffer.decode('utf-8', errors='replace').strip()
                        if partial_line:
                            self._update_progress_bar(partial_line, pbar, stage_progress)

            except asyncio.TimeoutError:
                pass

            # Try stderr
            try:
                err_line = await asyncio.wait_for(process.stderr.readline(), timeout=0.01)
                if err_line:
                    err_str = err_line.decode('utf-8', errors='replace').strip()
                    if err_str:
                        print(err_str, flush=True)  # Always show error output
                        last_update_time = current_time
                        line_received = True
            except asyncio.TimeoutError:
                pass

            # Check if process finished
            if not line_received:
                if process.returncode is not None:
                    # Process any remaining buffer
                    if buffer:
                        line_str = buffer.decode('utf-8', errors='replace').strip()
                        if line_str:
                            self._update_progress_bar(line_str, pbar, stage_progress)
                    break
                # Update progress bar periodically even without output
                if (current_time - last_update_time) > 2:
                    if hasattr(pbar, 'refresh'):
                        pbar.refresh()
                    last_update_time = current_time

        # Wait for process to complete and get return code
        return_code = await process.wait()

        # Close progress bar
        if hasattr(pbar, 'close'):
            pbar.close()

        if return_code != 0:
            # Try to get any remaining stderr content if stderr exists
            try:
                if process.stderr:
                    stderr_data = await process.stderr.read()
                    stderr_str = stderr_data.decode('utf-8', errors='replace')
                    if stderr_str.strip():
                        print(f"Error output: {stderr_str}", flush=True)
            except Exception:
                pass
            raise RuntimeError(f"Video download failed with return code: {return_code}")

        # Clean up temporary directory
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir(exist_ok=True, parents=True)
            logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")

        # Validate the downloaded file
        if video_file.exists():
            file_size_bytes = video_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            file_size_gb = file_size_bytes / (1024 * 1024 * 1024)

            # Check if file has reasonable content (at least 1MB)
            if file_size_bytes < 1024 * 1024:  # Less than 1MB
                raise RuntimeError(f"Downloaded video file is too small ({file_size_bytes} bytes). Download may have failed.")

            if file_size_gb >= 1.0:
                logger.info(f"📹 Video file downloaded: {file_size_gb:.2f} GB ({file_size_mb:.1f} MB)")
            else:
                logger.info(f"📹 Video file downloaded: {file_size_mb:.1f} MB")
            logger.info(f"📹 Video file path: {video_file}")
        else:
            raise RuntimeError(f"Video file not found after download: {video_file}")

        return video_file

    async def download_video(self, video_id):
        """
        Download a Twitch video using optimized fallback strategies

        Args:
            video_id (str): Twitch video ID

        Returns:
            Path: Path to the downloaded video file
        """
        # Use the optimized fallback strategy for maximum performance
        return await self.download_video_with_fallback(video_id)

    def _update_progress_bar(self, line_str, pbar, stage_progress):
        """
        Parse TwitchDownloaderCLI output and update the visual progress bar
        Returns True if progress was updated, False otherwise
        """
        import re

        # Parse different TwitchDownloaderCLI status patterns
        # Pattern: [STATUS] - Fetching Video Info [1/4]
        stage_match = re.search(r'\[STATUS\] - (.+?) \[(\d+)/4\]', line_str)
        if stage_match:
            stage_name = stage_match.group(1)
            stage_num = int(stage_match.group(2))

            # Update progress bar description and show progress
            stage_desc = f"📹 {stage_name}"
            if hasattr(pbar, 'set_description'):
                pbar.set_description(stage_desc)
            else:
                print(f"\r{stage_desc}", end="", flush=True)

            # Calculate overall progress based on stage
            base_progress = (stage_num - 1) * 25  # Each stage is 25% of total
            stage_progress[stage_num] = 0  # Reset stage progress

            # Update progress bar
            total_progress = base_progress
            if hasattr(pbar, 'n'):
                pbar.n = total_progress
            if hasattr(pbar, 'refresh'):
                pbar.refresh()
            else:
                print(f" [{total_progress:.0f}%]", end="", flush=True)

            return True

        # Pattern: [STATUS] - Downloading 45% [2/4]
        progress_match = re.search(r'\[STATUS\] - (.+?) (\d+)% \[(\d+)/4\]', line_str)
        if progress_match:
            stage_name = progress_match.group(1)
            progress_percent = int(progress_match.group(2))
            stage_num = int(progress_match.group(3))

            # Update progress bar description to show current percentage
            stage_desc = f"📹 {stage_name} {progress_percent}%"
            if hasattr(pbar, 'set_description'):
                pbar.set_description(stage_desc)
            else:
                print(f"\r{stage_desc}", end="", flush=True)

            # Calculate overall progress
            base_progress = (stage_num - 1) * 25  # Each stage is 25% of total
            stage_contribution = (progress_percent / 100) * 25  # This stage's contribution
            total_progress = base_progress + stage_contribution

            # Update stage progress tracking
            stage_progress[stage_num] = progress_percent

            # Update progress bar
            if hasattr(pbar, 'n'):
                pbar.n = total_progress
            if hasattr(pbar, 'refresh'):
                pbar.refresh()
            else:
                print(f" [{total_progress:.0f}%]", end="", flush=True)

            return True

        # Check for important info messages that should be shown
        if any(keyword in line_str.lower() for keyword in ['info', 'missing', 'corrupt', 'redownload']):
            print(f"ℹ️  {line_str}", flush=True)
            return True

        return False

    async def extract_audio(self, video_file, video_id):
        """
        Extract compressed audio from a video file optimized for transcription

        Args:
            video_file (Path): Path to the video file
            video_id (str): Twitch video ID

        Returns:
            Path: Path to the extracted audio file
        """
        # Use MP3 format for much smaller file sizes
        audio_file = RAW_AUDIO_DIR / f"audio_{video_id}.mp3"

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

        # Get system-optimized thread count for audio processing
        from utils.resource_optimizer import resource_optimizer
        cpu_cores = resource_optimizer.system_info['cpu']['logical_cores']
        audio_threads = min(cpu_cores, 16)  # Cap at 16 threads for audio processing

        # Create command for highly compressed audio optimized for speech transcription
        command = [
            BINARY_PATHS["ffmpeg"],
            "-i", str(video_file),
            "-vn",  # No video
            "-acodec", "mp3",  # MP3 codec for compression
            "-ar", "8000",  # 8 kHz sample rate (sufficient for speech)
            "-ac", "1",  # Mono
            "-ab", "32k",  # Very low bitrate (32 kbps) for maximum compression
            "-threads", str(audio_threads),  # Use system-optimized thread count
            "-y",  # Overwrite output file
            str(audio_file)
        ]

        # Log audio thread count at debug level only
        logger.debug(f"🎵 Using {audio_threads} threads for audio processing")
        logger.info(f"🎵 Audio extraction command: {' '.join(command)}")

        # Verify ffmpeg binary exists and is executable
        ffmpeg_path = command[0]
        if not os.path.exists(ffmpeg_path):
            raise FileNotFoundError(f"ffmpeg binary not found at: {ffmpeg_path}")
        if not os.access(ffmpeg_path, os.X_OK):
            logger.warning(f"ffmpeg binary not executable, attempting to fix: {ffmpeg_path}")
            try:
                os.chmod(ffmpeg_path, 0o755)
            except Exception as e:
                logger.error(f"Failed to make ffmpeg executable: {e}")

        # Update Convex status for audio processing
        from utils.convex_client_updated import ConvexManager
        convex_manager = ConvexManager()

        print(f"📊 Updating status to 'Processing audio' for video {video_id}...", flush=True)
        success = convex_manager.update_video_status(video_id, "Processing audio")
        if success:
            print(f"✅ Status updated to 'Processing audio'", flush=True)

        # Create progress bar for audio conversion
        pbar = tqdm(total=100, desc="🎵 Converting audio", unit="%")

        # Prepare environment for subprocess
        env = os.environ.copy()
        if platform.system() == "Darwin":
            temp_dir = Path.home() / '.dotnet' / 'bundle_extract'
            temp_dir.mkdir(parents=True, exist_ok=True)
            env['DOTNET_BUNDLE_EXTRACT_BASE_DIR'] = str(temp_dir)

        # Run the command
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        # Get system-optimized timeout for audio extraction
        timeout_settings = resource_optimizer.get_timeout_settings()
        audio_timeout = timeout_settings['audio_timeout']
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
                line = await asyncio.wait_for(process.stderr.readline(), timeout=60)
            except asyncio.TimeoutError:
                logger.warning("No output from audio extraction process for 60 seconds, checking if process is still alive")
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

        # Wait for process to complete
        return_code = await process.wait()
        pbar.close()

        if return_code != 0:
            # If we have an error, try to get any remaining stderr content
            stderr_data = await process.stderr.read()
            stderr_str = stderr_data.decode('utf-8', errors='replace')

            # Also try to get stdout content for more context
            stdout_data = await process.stdout.read()
            stdout_str = stdout_data.decode('utf-8', errors='replace')

            error_msg = f"Error extracting audio (return code: {return_code})"
            if stderr_str.strip():
                error_msg += f"\nStderr: {stderr_str}"
            if stdout_str.strip():
                error_msg += f"\nStdout: {stdout_str}"

            # Log the full command for debugging
            logger.error(f"Failed command: {' '.join(command)}")
            logger.error(f"Video file exists: {video_file.exists()}")
            logger.error(f"Video file size: {video_file.stat().st_size if video_file.exists() else 'N/A'}")

            raise RuntimeError(error_msg)

        # Log audio file size with enhanced formatting
        if audio_file.exists():
            file_size_bytes = audio_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)

            # Calculate compression ratio if video file exists
            compression_info = ""
            if video_file.exists():
                video_size_mb = video_file.stat().st_size / (1024 * 1024)
                compression_ratio = (video_size_mb / file_size_mb) if file_size_mb > 0 else 0
                compression_info = f" (compressed {compression_ratio:.1f}x from video)"

            logger.info(f"🎵 Audio file extracted: {file_size_mb:.1f} MB{compression_info}")
            logger.info(f"🎵 Audio file path: {audio_file}")
        else:
            logger.warning(f"Audio file not found: {audio_file}")

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

        # Verify video file exists and has content
        if not video_file.exists():
            raise RuntimeError(f"Video download failed: File does not exist at {video_file}")

        file_size = video_file.stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Video download failed: File is empty at {video_file}")

        logger.info(f"✅ Video download verified: {file_size / (1024*1024):.1f} MB")

        # Extract audio
        audio_file = await self.extract_audio(video_file, video_id)

        # Return results
        return {
            "video_id": video_id,
            "video_file": video_file,
            "audio_file": audio_file,
            "twitch_info": video_info
        }
