import logging
import sys
import re
import argparse
import subprocess
import json
import csv
import platform
from pathlib import Path
import os

def get_cli_command():
    """
    Returns the appropriate TwitchDownloaderCLI command based on OS.
    """
    system = platform.system().lower()
    if system == 'windows':
        return "TwitchDownloaderCLI.exe"
    elif system == 'linux':
        return "./TwitchDownloaderCLI"
    elif system == 'darwin':  # macOS
        return "./TwitchDownloaderCLI_mac"
    else:
        raise OSError(f"Unsupported operating system: {system}")

def extract_vod_id(vod_link):
    """Extract the VOD ID from a Twitch VOD URL."""
    return vod_link.rstrip('/').split('/')[-1]

def is_valid_vod_link(vod_link):
    """Validates the Twitch VOD link format."""
    pattern = r'^https:\/\/www\.twitch\.tv\/videos\/\d+$'
    return re.match(pattern, vod_link) is not None

def download_chat(video_id, output_dir):
    """
    Downloads chat using TwitchDownloaderCLI and converts it to CSV format.

    Args:
        video_id (str): Twitch VOD ID
        output_dir (Path): Directory to save output files
    """
    json_path = output_dir / f"{video_id}_chat.json"
    csv_path = output_dir / f"{video_id}_chat.csv"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the appropriate CLI command for the current OS
    cli_command = get_cli_command()

    # Download chat in JSON format
    logging.info(f"Downloading chat for VOD: {video_id}")
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
        logging.info(f"Chat downloaded successfully to {json_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download chat: {e}")
        logging.error("Make sure TwitchDownloaderCLI is installed and in the correct location:")
        logging.error("Windows: Place TwitchDownloaderCLI.exe in the same directory or add to PATH")
        logging.error("Linux/MacOS: Place TwitchDownloaderCLI in the same directory and ensure it's executable (chmod +x)")
        raise
    except FileNotFoundError:
        logging.error("TwitchDownloaderCLI not found!")
        logging.error("Please download it from: https://github.com/lay295/TwitchDownloader/releases")
        logging.error("Windows: Place TwitchDownloaderCLI.exe in the same directory or add to PATH")
        logging.error("Linux/MacOS: Place TwitchDownloaderCLI in the same directory and ensure it's executable (chmod +x)")
        raise

    # Convert JSON to CSV
    convert_json_to_csv(json_path, csv_path)
    logging.info(f"Chat converted to CSV: {csv_path}")

    # Optionally remove JSON file to save space
    json_path.unlink()

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

def main(video_id=None, output_dir=None, keep_json=False):
    """Download chat data from a Twitch VOD.

    Args:
        video_id (str): Twitch VOD ID
        output_dir (str): Directory to save output files
        keep_json (bool): Whether to keep the intermediate JSON file
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / 'twitch_chat_downloader.log')
        ]
    )
    logging.info('Process started.')

    # If no video ID provided, prompt user for input
    if video_id is None:
        video_id = input("Please enter a Twitch VOD ID: ").strip()
        if not video_id:
            raise ValueError("VOD ID is required")

    # Convert output_dir to Path object
    output_dir = Path(output_dir) if output_dir else Path('data')

    try:
        csv_path = download_chat(video_id, output_dir)
        logging.info(f'Chat data saved to: {csv_path}')
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

    logging.info('Process completed successfully.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Twitch chat data from a VOD')
    parser.add_argument('--video-id', type=str, help='Twitch VOD ID (without the URL)', required=True)
    parser.add_argument('--output-dir', type=str, help='Output directory for chat files', default='data')
    parser.add_argument('--keep-json', action='store_true', help='Keep the intermediate JSON file')
    args = parser.parse_args()

    try:
        main(args.video_id, args.output_dir, args.keep_json)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)
