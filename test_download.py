import subprocess
import os
import platform
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Set up paths
    if platform.system() == "Darwin":  # macOS
        cli_path = "./TwitchDownloaderCLI_mac"
        ffmpeg_path = "./ffmpeg_mac"

        # Set up .NET environment for macOS
        cache_dir = os.path.expanduser("~/.cache/twitch_downloader")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["DOTNET_BUNDLE_EXTRACT_BASE_DIR"] = cache_dir
        logging.info(f"Set DOTNET_BUNDLE_EXTRACT_BASE_DIR to {cache_dir}")
    elif platform.system() == "Windows":
        cli_path = "./TwitchDownloaderCLI.exe"
        ffmpeg_path = "./ffmpeg.exe"
    else:  # Linux
        cli_path = "./TwitchDownloaderCLI"
        ffmpeg_path = "./ffmpeg"

    # Make sure executables are executable
    os.chmod(cli_path, 0o755)
    os.chmod(ffmpeg_path, 0o755)

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Video ID to download
    video_id = "2442637981"

    # Download command
    download_cmd = [
        cli_path,
        "videodownload",
        "--id", video_id,
        "-o", "outputs/video.mp4",
        "--quality", "worst",
        "--threads", "4"
    ]

    logging.info(f"Running command: {' '.join(download_cmd)}")

    # Run the download command
    try:
        result = subprocess.run(download_cmd, check=True, capture_output=True, text=True)
        logging.info("Download completed successfully")
        logging.info(f"Output: {result.stdout}")

        # Convert to audio
        convert_cmd = [
            ffmpeg_path,
            "-i", "outputs/video.mp4",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            f"outputs/audio_{video_id}.wav"
        ]

        logging.info(f"Running command: {' '.join(convert_cmd)}")

        result = subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        logging.info("Conversion completed successfully")

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
    except Exception as e:
        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
