import subprocess
import os
import platform
import re
import logging
import asyncio
from tqdm import tqdm
import shutil
from pathlib import Path
import psutil

# Define global paths
DOWNLOADS_DIR = Path("downloads")
TEMP_DIR = DOWNLOADS_DIR / "temp"
OUTPUT_DIR = Path("outputs")

# Create necessary directories
DOWNLOADS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

class TwitchVideoDownloader:
    def __init__(self):
        self._setup_logging()
        
        # Add this check before setup_cli_path
        if platform.system() == "Darwin":  # macOS
            self._setup_macos_environment()
        
        self._setup_cli_path()
        self._verify_ffmpeg_exists()

    def _setup_logging(self):
        """Configure logging with timestamp, level, and message"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('twitch_downloader.log'),
                logging.StreamHandler()
            ]
        )
        logging.info("Initializing TwitchVideoDownloader")

    def _setup_cli_path(self):
        """Setup CLI path based on platform"""
        # Add debug logging
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Directory contents: {os.listdir(os.getcwd())}")
        
        system = platform.system()
        logging.info(f"Detected operating system: {system}")
        
        # Set CLI path based on platform
        if system == "Darwin":  # macOS
            self.cli_path = "./TwitchDownloaderCLI_mac"
            self.ffmpeg_path = "./ffmpeg_mac"
        elif system == "Windows":
            self.cli_path = "./TwitchDownloaderCLI.exe"
            self.ffmpeg_path = "./ffmpeg.exe"
        else:  # Linux (Cloud Functions uses Linux)
            self.cli_path = "./TwitchDownloaderCLI"
            self.ffmpeg_path = "./ffmpeg"
        
        # Check and set permissions for TwitchDownloaderCLI
        if os.path.exists(self.cli_path):
            try:
                current_permissions = os.stat(self.cli_path).st_mode
                os.chmod(self.cli_path, current_permissions | 0o111)
                logging.info(f"Set executable permissions for {self.cli_path}")
            except Exception as e:
                logging.error(f"Failed to set permissions for TwitchDownloaderCLI: {e}")
                raise
        else:
            logging.error(f"TwitchDownloaderCLI not found at {self.cli_path}")
            raise FileNotFoundError("TwitchDownloaderCLI not found")
        
        # Check and set permissions for ffmpeg
        if os.path.exists(self.ffmpeg_path):
            try:
                current_permissions = os.stat(self.ffmpeg_path).st_mode
                os.chmod(self.ffmpeg_path, current_permissions | 0o111)
                logging.info(f"Set executable permissions for {self.ffmpeg_path}")
            except Exception as e:
                logging.error(f"Failed to set permissions for ffmpeg: {e}")
                raise
        else:
            logging.error(f"ffmpeg not found at {self.ffmpeg_path}")
            raise FileNotFoundError("ffmpeg not found")

    def _setup_macos_environment(self):
        """Setup specific environment for macOS"""
        self.cli_path = "./TwitchDownloaderCLI_mac"
        cache_dir = os.path.expanduser("~/.cache/twitch_downloader")
        os.environ["DOTNET_BUNDLE_EXTRACT_BASE_DIR"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Created cache directory at: {cache_dir}")

    def _verify_cli_exists(self):
        """Verify CLI executable exists"""
        if not os.path.exists(self.cli_path):
            logging.error(f"TwitchDownloaderCLI not found at {self.cli_path}")
            raise FileNotFoundError(f"TwitchDownloaderCLI not found at {self.cli_path}")
        logging.info(f"Found TwitchDownloaderCLI at: {self.cli_path}")

    def _clean_existing_files(self, files):
        """Remove existing files if they exist"""
        for file in files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    logging.info(f"Removed existing file: {file}")
                except OSError as e:
                    logging.error(f"Error removing file {file}: {e}")
                    raise

    def _get_optimal_thread_count(self, is_download=True):
        """Determine optimal thread count based on CPU cores"""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            return min(cpu_count * 2, 16) if is_download else min(max(2, cpu_count - 2), 8)
        except Exception as e:
            logging.warning(f"Error determining thread count: {str(e)}")
            return 4  # Safe default

    async def _download_video(self, video_id, output_file, temp_path):
        """Download video using TwitchDownloaderCLI"""
        try:
            download_threads = self._get_optimal_thread_count(is_download=True)
            command = [
                self.cli_path,
                "videodownload",
                "--id", video_id,
                "-o", output_file,
                "--quality", "worst",
                "--threads", str(download_threads),
                "--temp-path", temp_path
            ]
            
            # Log the full command
            logging.info(f"Running download command: {' '.join(command)}")
            
            # Log directory contents and permissions
            logging.info(f"Current directory contents: {os.listdir('.')}")
            logging.info(f"CLI file exists: {os.path.exists(self.cli_path)}")
            logging.info(f"CLI file permissions: {oct(os.stat(self.cli_path).st_mode)}")
            
            process = await self._run_process_with_progress(
                command, 
                "Download Progress", 
                progress_pattern="Progress:"
            )
            
            if process.returncode != 0:
                logging.error(f"Download failed with return code: {process.returncode}")
                # Log stderr output
                stderr = await process.stderr.read()
                logging.error(f"Error output: {stderr.decode()}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Exception during download: {str(e)}")
            raise

    async def _convert_to_wav(self, input_file, output_file):
        """Convert video to WAV using optimized FFmpeg settings"""
        try:
            cpu_cores = max(1, psutil.cpu_count(logical=False) - 2)
            command = [
                'ffmpeg',
                '-i', input_file,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-threads', str(cpu_cores),
                '-y',
                output_file
            ]
            
            process = await self._run_process_with_progress(
                command,
                "Converting to WAV",
                progress_pattern="time=",
                duration=self._get_video_duration(input_file)
            )

            if process.returncode != 0:
                raise Exception("FFmpeg conversion failed")

            return output_file

        except Exception as e:
            logging.error(f"Error converting to WAV: {str(e)}")
            raise

    async def process_video(self, url):
        """Main method to process video: download and convert"""
        try:
            # Get video ID and create paths
            video_id = self.extract_video_id(url)
            video_file = OUTPUT_DIR / "video.mp4"
            audio_file = OUTPUT_DIR / f"audio_{video_id}.wav"
            
            # Clean existing files
            self._clean_existing_files([video_file, audio_file])
            
            # Process video
            if not await self._download_video(video_id, str(video_file), str(TEMP_DIR)):
                raise Exception("Video download failed")
                
            if not await self._convert_to_wav(str(video_file), str(audio_file)):
                raise Exception("Audio conversion failed")
                
            # Cleanup all temporary files and folders
            self._cleanup_files(TEMP_DIR, video_file)
            self._cleanup_downloads_folder()
            
            return audio_file
            
        except Exception as e:
            logging.error(f"Process failed: {str(e)}")
            self._cleanup_files(TEMP_DIR, video_file)
            self._cleanup_downloads_folder()
            raise

    def extract_video_id(self, url):
        """Extract video ID from Twitch URL"""
        # Match patterns like: https://www.twitch.tv/videos/2309350434
        pattern = r'(?:www\.)?twitch\.tv/videos/(\d+)'
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logging.info(f"Extracted video ID: {video_id}")
            return video_id
        else:
            logging.error(f"Could not extract video ID from URL: {url}")
            raise ValueError(f"Invalid Twitch video URL format: {url}")

    def _cleanup_files(self, temp_path, video_file):
        """Clean up temporary files and directories"""
        try:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                logging.info(f"Cleaned up temporary directory: {temp_path}")
            if os.path.exists(video_file):
                os.remove(video_file)
                logging.info(f"Cleaned up video file: {video_file}")
        except OSError as e:
            logging.warning(f"Error during cleanup: {e}")

    def _cleanup_downloads_folder(self):
        """Clean up the downloads folder"""
        try:
            if DOWNLOADS_DIR.exists():
                shutil.rmtree(DOWNLOADS_DIR)
                logging.info(f"Cleaned up downloads directory: {DOWNLOADS_DIR}")
        except OSError as e:
            logging.warning(f"Error cleaning up downloads directory: {e}")

    async def _run_process_with_progress(self, command, description, progress_pattern=None, duration=None):
        """Run a process with progress bar tracking"""
        logging.info(f"Running command: {' '.join(command)}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            pbar = tqdm(desc=description, total=100, unit="%")
            last_progress = 0
            
            # Capture both stdout and stderr
            stdout_data = []
            stderr_data = []
            
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                    
                line = line.decode('utf-8', errors='ignore').strip()
                stderr_data.append(line)
                logging.info(f"Process output: {line}")  # Log all output
                
                if duration and "time=" in line:
                    try:
                        time_str = re.search(r'time=(\d{2}:\d{2}:\d{2}.\d{2})', line)
                        if time_str:
                            h, m, s = map(float, time_str.group(1).split(':'))
                            current_time = h * 3600 + m * 60 + s
                            progress = min(100, (current_time / duration) * 100)
                            pbar.update(progress - last_progress)
                            last_progress = progress
                    except (ValueError, AttributeError):
                        continue
                    
            pbar.close()
            await process.wait()
            
            # Log all captured output
            if stdout_data:
                logging.info("Process stdout:")
                for line in stdout_data:
                    logging.info(line)
            if stderr_data:
                logging.info("Process stderr:")
                for line in stderr_data:
                    logging.info(line)
                
            return process
            
        except Exception as e:
            logging.error(f"Process execution failed: {str(e)}")
            raise

    def _get_video_duration(self, input_file):
        """Get video duration in seconds using ffprobe"""
        try:
            cmd = [
                'ffprobe', 
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_file
            ]
            
            # Run ffprobe command
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            logging.info(f"Video duration: {duration} seconds")
            return duration
            
        except (subprocess.SubprocessError, ValueError) as e:
            logging.error(f"Error getting video duration: {str(e)}")
            # Return a default duration if unable to get actual duration
            return 0

    def _verify_ffmpeg_exists(self):
        """Verify ffmpeg and ffprobe are installed"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True)
            subprocess.run(['ffprobe', '-version'], capture_output=True)
            logging.info("ffmpeg and ffprobe found")
        except FileNotFoundError:
            logging.error("ffmpeg or ffprobe not found. Please install ffmpeg.")
            raise FileNotFoundError("ffmpeg or ffprobe not found. Please install ffmpeg.")

async def main():
    logging.info("Starting TwitchVideoDownloader application")
    downloader = TwitchVideoDownloader()
    
    # Get URL from user
    url = input("Enter Twitch video URL: ")
    logging.info(f"User provided URL: {url}")
    
    try:
        # Note the await here
        result = await downloader.process_video(url)
        print(result)
        logging.info("Download process completed successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
