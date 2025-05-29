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
import gzip
import time
import pandas as pd

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
    # Use .txt extension for the initial download to avoid image embedding completely
    txt_path = output_dir / f"{video_id}_chat.txt"
    json_path = output_dir / f"{video_id}_chat.json"  # Use standard .json extension
    csv_path = output_dir / f"{video_id}_chat.csv"

    # Skip if files already exist
    if json_path.exists() and csv_path.exists():
        logger.info(f"Chat files already exist for VOD {video_id}, skipping download")
        return json_path, csv_path

    # Check if CSV file already exists
    if csv_path.exists():
        logger.info(f"Chat CSV file already exists: {csv_path}")
        # If JSON file doesn't exist, create it from the CSV for consistency
        if not json_path.exists():
            logger.info(f"JSON file doesn't exist, but we have CSV. Creating a placeholder JSON.")
            # Create a minimal JSON structure
            chat_data = {
                "comments": [],
                "video": {
                    "id": video_id,
                    "duration": 0
                }
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f)
        return json_path, csv_path

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading chat for VOD {video_id} to {output_dir}")

    # Get the appropriate CLI command for the current OS
    cli_command = BINARY_PATHS["twitch_downloader"]

    # Download chat in JSON format
    logger.info(f"Downloading chat for VOD: {video_id}")

    # First, download chat in TEXT format to avoid image embedding completely
    # The TwitchDownloaderCLI automatically uses text format when the output file has .txt extension
    download_txt_cmd = [
        cli_command,
        "chatdownload",
        "--id", video_id,
        "--output", str(txt_path),  # Use .txt extension to get text format
        "--timestamp-format", "Relative",
        "--threads", "32",
        "--collision", "Overwrite",
        "--log-level", "Status,Error"
    ]

    # Then we'll convert the text file to JSON format for our pipeline
    download_cmd = download_txt_cmd

    # Set DOTNET_BUNDLE_EXTRACT_BASE_DIR environment variable
    env = os.environ.copy()
    if platform.system().lower() == 'darwin':  # macOS specific fix
        temp_dir = Path.home() / '.twitch_downloader_temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        env['DOTNET_BUNDLE_EXTRACT_BASE_DIR'] = str(temp_dir)

    # Implement retry logic for network issues
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            # Add shell=True for Windows to properly execute the .exe
            is_windows = platform.system().lower() == 'windows'
            logger.info(f"Attempt {attempt}/{max_retries} to download chat")

            subprocess.run(
                download_cmd,
                check=True,
                shell=is_windows,
                env=env,  # Add environment variables
                timeout=1800  # 30-minute timeout
            )
            logger.info(f"Chat downloaded successfully to {txt_path}")

            # Now convert the text file to JSON format
            logger.info(f"Converting text chat to JSON format")
            convert_text_to_json(txt_path, json_path)
            logger.info(f"Chat converted to JSON format: {json_path}")

            # Remove the text file to save space
            if json_path.exists() and json_path.stat().st_size > 100:
                logger.info(f"Removing text file to save space: {txt_path}")
                txt_path.unlink()

            break  # Success, exit the retry loop

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download chat (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("Maximum retry attempts reached. Giving up.")
                logger.error("Make sure TwitchDownloaderCLI is installed and in the correct location:")
                logger.error("Windows: Place TwitchDownloaderCLI.exe in the same directory or add to PATH")
                logger.error("Linux/MacOS: Place TwitchDownloaderCLI in the same directory and ensure it's executable (chmod +x)")
                raise

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while downloading chat (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("Maximum retry attempts reached. Giving up.")
                raise

        except FileNotFoundError:
            logger.error("TwitchDownloaderCLI not found!")
            logger.error("Please download it from: https://github.com/lay295/TwitchDownloader/releases")
            logger.error("Windows: Place TwitchDownloaderCLI.exe in the same directory or add to PATH")
            logger.error("Linux/MacOS: Place TwitchDownloaderCLI in the same directory and ensure it's executable (chmod +x)")
            raise

        finally:
            # No temporary config file to clean up anymore
            pass

    # Convert JSON to CSV
    convert_json_to_csv(json_path, csv_path)
    logger.info(f"Chat converted to CSV: {csv_path}")

    # Check if CSV file exists and has content before removing JSON
    if csv_path.exists() and csv_path.stat().st_size > 100:  # Ensure CSV has at least some content
        # Keep the JSON file for now - we need it for the pipeline
        logger.info(f"Keeping JSON file for pipeline compatibility: {json_path}")
    else:
        logger.warning(f"CSV file missing or empty, keeping JSON file: {json_path}")

    # Also save a copy to the data directory for backward compatibility
    data_csv_path = DATA_DIR / f"{video_id}_chat.csv"
    if csv_path.exists():
        shutil.copy2(csv_path, data_csv_path)
        logger.info(f"Chat data also copied to: {data_csv_path} for backward compatibility")
    else:
        logger.error(f"Cannot copy CSV file to data directory: {csv_path} does not exist")

    return json_path, csv_path

def convert_json_to_csv(json_path, csv_path):
    """
    Converts the downloaded JSON chat file to CSV format.
    Uses a hybrid approach - direct loading for smaller files, streaming for larger ones.

    Args:
        json_path (Path): Path to input JSON file
        csv_path (Path): Path to output CSV file
    """

    # Check if the file is gzipped
    is_gzipped = str(json_path).endswith('.gz')

    # Get file size in MB
    file_size_mb = os.path.getsize(json_path) / (1024 * 1024)
    logger.info(f"Chat JSON file size: {file_size_mb:.2f} MB")

    # For smaller files (< 100MB), use direct loading which is faster
    if file_size_mb < 100:
        logger.info("Using direct JSON loading for small file")
        try:
            # Load the JSON file
            logger.info(f"Loading JSON file: {json_path}")
            if is_gzipped:
                with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                    chat_data = json.load(f)
            else:
                with open(json_path, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)

            # Log JSON structure for debugging
            if chat_data:
                logger.info(f"JSON loaded successfully, keys: {list(chat_data.keys())}")
            else:
                logger.warning("JSON file loaded but is empty or null")

            # Extract the relevant data
            comments = chat_data.get('comments', [])
            logger.info(f"Number of comments found in JSON: {len(comments)}")

            # Use pandas for faster CSV writing
            if comments:
                # Extract only the fields we need
                data = []
                error_count = 0
                for i, comment in enumerate(comments):
                    try:
                        time_in_seconds = comment['content_offset_seconds']
                        username = comment['commenter']['display_name']
                        message = comment['message']['body']
                        data.append([time_in_seconds, username, message])

                        # Log a sample of comments for debugging
                        if i < 5 or i % 10000 == 0:
                            logger.info(f"Sample comment {i}: time={time_in_seconds}, user={username}, message={message[:30]}...")
                    except (KeyError, TypeError) as e:
                        error_count += 1
                        if error_count < 10:  # Limit the number of error logs
                            logger.warning(f"Error processing comment {i}: {str(e)}")
                        continue

                # Create DataFrame and write to CSV
                logger.info(f"Creating DataFrame with {len(data)} comments")
                df = pd.DataFrame(data, columns=['time', 'username', 'message'])
                logger.info(f"Writing DataFrame to CSV: {csv_path}")
                df.to_csv(csv_path, index=False)
                logger.info(f"Converted {len(df)} chat messages to CSV using pandas (with {error_count} errors)")

                # Verify the CSV file was created
                if csv_path.exists():
                    logger.info(f"CSV file created successfully: {csv_path.stat().st_size} bytes")
                else:
                    logger.error(f"CSV file was not created: {csv_path}")
            else:
                # Create an empty CSV if no comments
                logger.warning("No comments found in the JSON file, creating empty CSV")
                with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['time', 'username', 'message'])
                logger.warning("Empty CSV file created with headers only")

        except Exception as e:
            logger.error(f"Error in direct JSON loading: {str(e)}")
            # Fallback to basic CSV writing
            _fallback_csv_writing(json_path, csv_path, is_gzipped)

    # For larger files, use chunked processing
    else:
        logger.info("Using chunked processing for large file")
        try:
            # Process in chunks using pandas
            import ijson
            logger.info(f"Using ijson for streaming large JSON file: {json_path}")

            # Open the CSV file for writing the header
            logger.info(f"Creating CSV file with headers: {csv_path}")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['time', 'username', 'message'])
                logger.info("CSV headers written successfully")

            # Process in chunks of 10,000 comments
            chunk_size = 10000
            chunk = []
            count = 0
            error_count = 0

            # Open the JSON file
            logger.info(f"Opening JSON file for streaming: {json_path}")
            if is_gzipped:
                json_file = gzip.open(json_path, 'rb')
                logger.info("Opened gzipped JSON file")
            else:
                json_file = open(json_path, 'rb')
                logger.info("Opened regular JSON file")

            # Use ijson for streaming
            try:
                logger.info("Starting ijson streaming with pattern 'comments.item'")
                comments = ijson.items(json_file, 'comments.item')

                # Process each comment
                for i, comment in enumerate(comments):
                    try:
                        time_in_seconds = comment['content_offset_seconds']
                        username = comment['commenter']['display_name']
                        message = comment['message']['body']
                        chunk.append([time_in_seconds, username, message])
                        count += 1

                        # Log sample comments for debugging
                        if count <= 5 or count % 10000 == 0:
                            logger.info(f"Sample comment {count}: time={time_in_seconds}, user={username}, message={message[:30]}...")

                        # Process in chunks
                        if len(chunk) >= chunk_size:
                            logger.info(f"Processing chunk of {len(chunk)} comments")
                            df = pd.DataFrame(chunk, columns=['time', 'username', 'message'])
                            df.to_csv(csv_path, mode='a', header=False, index=False)
                            chunk = []
                            logger.info(f"Processed {count} chat messages so far")

                            # Verify CSV file is growing
                            if csv_path.exists():
                                logger.info(f"CSV file size: {csv_path.stat().st_size} bytes")
                            else:
                                logger.error("CSV file does not exist after writing chunk!")
                    except (KeyError, TypeError) as e:
                        error_count += 1
                        if error_count < 10:  # Limit the number of error logs
                            logger.warning(f"Error processing comment {count}: {str(e)}")
                        continue

                # Process the remaining comments
                if chunk:
                    logger.info(f"Processing final chunk of {len(chunk)} comments")
                    df = pd.DataFrame(chunk, columns=['time', 'username', 'message'])
                    df.to_csv(csv_path, mode='a', header=False, index=False)
                    logger.info("Final chunk processed")

                logger.info(f"Converted {count} chat messages to CSV using chunked processing (with {error_count} errors)")

                # Verify final CSV file
                if csv_path.exists():
                    logger.info(f"Final CSV file size: {csv_path.stat().st_size} bytes")
                else:
                    logger.error("CSV file does not exist after processing!")
            finally:
                logger.info("Closing JSON file")
                json_file.close()

        except Exception as e:
            logger.error(f"Error in chunked processing: {str(e)}")
            # Fallback to basic CSV writing
            _fallback_csv_writing(json_path, csv_path, is_gzipped)

def convert_text_to_json(txt_path, json_path):
    """
    Convert the downloaded text chat file to JSON format.

    Args:
        txt_path (Path): Path to input text file
        json_path (Path): Path to output JSON file
    """
    logger.info(f"Converting text chat file {txt_path} to JSON format {json_path}")

    try:
        # Read the text file
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        logger.info(f"Read {len(lines)} lines from text file")

        # Parse the text file and create a JSON structure
        # Format is typically: [timestamp] username: message
        comments = []
        for i, line in enumerate(lines):
            try:
                # Skip empty lines
                if not line.strip():
                    continue

                # Extract timestamp, username, and message
                # Example format: [00:01:23] username: message
                match = re.match(r'\[([^\]]+)\] ([^:]+): (.*)', line.strip())
                if match:
                    timestamp_str, username, message = match.groups()

                    # Convert timestamp to seconds
                    # Format could be [HH:MM:SS] or [MM:SS]
                    parts = timestamp_str.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        hours, minutes, seconds = map(float, parts)
                        timestamp_seconds = hours * 3600 + minutes * 60 + seconds
                    elif len(parts) == 2:  # MM:SS
                        minutes, seconds = map(float, parts)
                        timestamp_seconds = minutes * 60 + seconds
                    else:
                        # Try to parse as seconds
                        timestamp_seconds = float(timestamp_str)

                    # Create a comment object
                    comment = {
                        "content_offset_seconds": timestamp_seconds,
                        "commenter": {
                            "display_name": username
                        },
                        "message": {
                            "body": message
                        }
                    }

                    comments.append(comment)

                    # Log a sample of comments for debugging
                    if i < 5 or i % 10000 == 0:
                        logger.info(f"Sample comment {i}: time={timestamp_seconds}, user={username}, message={message[:30]}...")
                else:
                    logger.warning(f"Could not parse line {i}: {line.strip()}")
            except Exception as e:
                logger.warning(f"Error processing line {i}: {str(e)}")
                continue

        # Create the JSON structure
        chat_data = {
            "comments": comments,
            "video": {
                "id": txt_path.stem.split('_')[0],  # Extract video ID from filename
                "duration": 0  # We don't know the duration, but it's not critical
            }
        }

        # Write the JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

        logger.info(f"Successfully converted {len(comments)} comments to JSON format")
        return True
    except Exception as e:
        logger.error(f"Error converting text to JSON: {str(e)}")
        return False

def _fallback_csv_writing(json_path, csv_path, is_gzipped):
    """
    Fallback method for CSV conversion when other methods fail.

    Args:
        json_path (Path): Path to input JSON file
        csv_path (Path): Path to output CSV file
        is_gzipped (bool): Whether the file is gzipped
    """
    logger.warning(f"Using fallback CSV writing method for {json_path}")

    # Open the CSV file for writing
    logger.info(f"Creating CSV file with headers: {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'username', 'message'])
        logger.info("CSV headers written successfully")

        try:
            # Load the JSON file
            logger.info(f"Loading JSON file in fallback method: {json_path}")
            if is_gzipped:
                with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                    logger.info("Reading gzipped JSON file")
                    chat_data = json.load(f)
            else:
                with open(json_path, 'r', encoding='utf-8') as f:
                    logger.info("Reading regular JSON file")
                    chat_data = json.load(f)

            # Log JSON structure
            if chat_data:
                logger.info(f"JSON loaded successfully in fallback, keys: {list(chat_data.keys())}")
            else:
                logger.warning("JSON file loaded in fallback but is empty or null")

            # Process the comments
            comments = chat_data.get('comments', [])
            logger.info(f"Number of comments found in JSON (fallback): {len(comments)}")

            count = 0
            error_count = 0
            for i, comment in enumerate(comments):
                try:
                    time_in_seconds = comment['content_offset_seconds']
                    username = comment['commenter']['display_name']
                    message = comment['message']['body']
                    writer.writerow([time_in_seconds, username, message])
                    count += 1

                    # Log sample comments
                    if count <= 5 or count % 10000 == 0:
                        logger.info(f"Fallback - sample comment {count}: time={time_in_seconds}, user={username}, message={message[:30]}...")
                except (KeyError, TypeError) as e:
                    error_count += 1
                    if error_count < 10:  # Limit error logs
                        logger.warning(f"Fallback - error processing comment {i}: {str(e)}")
                    continue

            logger.info(f"Fallback method processed {count} comments with {error_count} errors")

        except Exception as e:
            logger.error(f"Fallback method also failed: {str(e)}")
            # Create an empty CSV as last resort
            logger.error("Creating emergency CSV with error message")
            writer.writerow(['0', 'system', 'Chat conversion failed'])

    # Verify CSV file was created
    if csv_path.exists():
        logger.info(f"Fallback CSV file created: {csv_path.stat().st_size} bytes")
    else:
        logger.error(f"Failed to create CSV file in fallback method: {csv_path}")

def download_chat_data(video_id):
    """
    Download chat data for a video ID

    This function is called when the pipeline status is "Fetching chat"

    Args:
        video_id (str): Twitch VOD ID

    Returns:
        dict: Dictionary with chat_file path
    """

    try:
        start_time = time.time()
        logger.info(f"Starting chat download for {video_id}...")

        # Download chat - returns tuple of (json_path, csv_path)
        json_path, csv_path = download_chat(video_id, RAW_CHAT_DIR)

        # Calculate and log performance metrics
        end_time = time.time()
        duration = end_time - start_time

        # Get file size - use the CSV file for size calculation
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)

        logger.info(f"Chat download completed in {duration:.2f} seconds")
        logger.info(f"Chat file size: {file_size_mb:.2f} MB")
        logger.info(f"Chat data saved to JSON: {json_path}")
        logger.info(f"Chat data saved to CSV: {csv_path}")

        return {
            "chat_file": csv_path,
            "json_file": json_path
        }
    except Exception as e:
        logger.error(f"Error downloading chat: {str(e)}")
        raise
