#!/usr/bin/env python3
"""
Raw File Processor

This script handles downloading, processing, and uploading raw files for a Twitch VOD.
It performs the following steps:
1. Creates the Output/Raw directory structure
2. Downloads the Twitch video
3. Extracts audio from the video
4. Generates a transcript from the audio
5. Generates a waveform from the audio
6. Downloads the Twitch chat
7. Uploads all these files to Google Cloud Storage

No files are deleted during this process, and all raw files are stored in the Output/Raw directory.
"""

import os
import logging
import asyncio
import json
import concurrent.futures
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Import required modules
from audio_downloader import TwitchVideoDownloader
from audio_transcription import TranscriptionHandler
from audio_waveform import process_audio_file
from chat_download import download_chat
from gcs_upload import upload_files

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define directory structure
OUTPUT_DIR = Path("Output")
RAW_DIR = OUTPUT_DIR / "Raw"
RAW_VIDEOS_DIR = RAW_DIR / "Videos"
RAW_AUDIO_DIR = RAW_DIR / "Audio"
RAW_TRANSCRIPTS_DIR = RAW_DIR / "Transcripts"
RAW_WAVEFORMS_DIR = RAW_DIR / "Waveforms"
RAW_CHAT_DIR = RAW_DIR / "Chat"

def create_directory_structure():
    """Create the Output/Raw directory structure"""
    directories = [
        OUTPUT_DIR,
        RAW_DIR,
        RAW_VIDEOS_DIR,
        RAW_AUDIO_DIR,
        RAW_TRANSCRIPTS_DIR,
        RAW_WAVEFORMS_DIR,
        RAW_CHAT_DIR
    ]

    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created directory: {directory}")

async def download_video_and_audio(url):
    """Download video and extract audio"""
    try:
        # Get video ID
        downloader = TwitchVideoDownloader()
        video_id = downloader.extract_video_id(url)

        # Define output paths
        video_file = RAW_VIDEOS_DIR / f"{video_id}.mp4"
        audio_file = RAW_AUDIO_DIR / f"audio_{video_id}.wav"

        # Download video and extract audio
        logger.info(f"Downloading video for {video_id}...")
        download_result = await downloader.process_video(url)

        if not download_result:
            raise RuntimeError("Failed to download files")

        # Move files to the correct locations
        os.rename(download_result["video_file"], video_file)
        os.rename(download_result["audio_file"], audio_file)

        logger.info(f"Video downloaded to: {video_file}")
        logger.info(f"Audio extracted to: {audio_file}")

        return {
            "video_id": video_id,
            "video_file": video_file,
            "audio_file": audio_file
        }
    except Exception as e:
        logger.error(f"Error downloading video and audio: {str(e)}")
        raise

async def generate_transcript(video_id, audio_file):
    """Generate transcript from audio"""
    try:
        logger.info(f"Generating transcript for {video_id}...")

        # Create transcription handler
        transcriber = TranscriptionHandler()

        # Process audio file
        transcription_result = await transcriber.process_audio_files(video_id, str(audio_file), str(RAW_TRANSCRIPTS_DIR))

        if not transcription_result:
            raise RuntimeError("Transcription failed")

        # Get paths to transcript files
        words_file = RAW_TRANSCRIPTS_DIR / f"audio_{video_id}_words.csv"
        paragraphs_file = RAW_TRANSCRIPTS_DIR / f"audio_{video_id}_paragraphs.csv"

        logger.info(f"Transcript words saved to: {words_file}")
        logger.info(f"Transcript paragraphs saved to: {paragraphs_file}")

        return {
            "words_file": words_file,
            "paragraphs_file": paragraphs_file
        }
    except Exception as e:
        logger.error(f"Error generating transcript: {str(e)}")
        raise

def generate_waveform(video_id, audio_file):
    """Generate waveform from audio"""
    try:
        logger.info(f"Generating waveform for {video_id}...")

        # Process audio file to generate waveform data
        waveform_data = process_audio_file(video_id, str(audio_file))

        if not waveform_data:
            raise RuntimeError("Failed to generate waveform data")

        # Save waveform data to file
        waveform_file = RAW_WAVEFORMS_DIR / f"audio_{video_id}_waveform.json"
        with open(waveform_file, 'w') as f:
            json.dump(waveform_data, f)

        logger.info(f"Waveform data saved to: {waveform_file}")

        return {
            "waveform_file": waveform_file
        }
    except Exception as e:
        logger.error(f"Error generating waveform: {str(e)}")
        raise

def download_chat_data(video_id):
    """Download chat data"""
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

def upload_to_gcs(video_id, files):
    """
    Upload only the 6 specific raw files to Google Cloud Storage:
    1. Video file (.mp4)
    2. Audio file (.wav)
    3. Waveform file (.json)
    4. Transcript paragraphs file (.csv)
    5. Transcript words file (.csv)
    6. Chat file (.csv)
    """
    try:
        logger.info(f"Uploading raw files to GCS for {video_id}...")

        # Define the 6 specific files we want to upload
        raw_files_to_upload = {
            "video_file": files.get("video_file"),
            "audio_file": files.get("audio_file"),
            "waveform_file": files.get("waveform_file"),
            "paragraphs_file": files.get("paragraphs_file"),
            "words_file": files.get("words_file"),
            "chat_file": files.get("chat_file")
        }

        # Filter out any None values and convert to strings
        file_paths = [str(file) for file in raw_files_to_upload.values() if file is not None and isinstance(file, Path)]

        # Upload files
        uploaded_files = upload_files(video_id, file_paths)

        # Verify we have exactly 6 files
        if len(uploaded_files) != 6:
            logger.warning(f"Expected to upload 6 files, but uploaded {len(uploaded_files)} files")

        logger.info(f"Successfully uploaded {len(uploaded_files)} raw files to GCS")

        return uploaded_files
    except Exception as e:
        logger.error(f"Error uploading files to GCS: {str(e)}")
        raise

async def process_raw_files(url):
    """Main function to process raw files"""
    try:
        # Create directory structure
        create_directory_structure()

        # Download video and audio
        download_result = await download_video_and_audio(url)
        video_id = download_result["video_id"]

        # Create a dictionary to store all file paths
        files = {
            "video_id": video_id,
            "video_file": download_result["video_file"],
            "audio_file": download_result["audio_file"]
        }

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start chat download in parallel
            chat_future = executor.submit(download_chat_data, video_id)

            # Generate transcript (depends on audio)
            transcript_result = await generate_transcript(video_id, download_result["audio_file"])
            files.update(transcript_result)

            # Generate waveform (depends on audio)
            waveform_future = executor.submit(generate_waveform, video_id, download_result["audio_file"])

            # Wait for parallel tasks to complete
            chat_result = chat_future.result()
            waveform_result = waveform_future.result()

            # Update files dictionary
            files.update(chat_result)
            files.update(waveform_result)

        # Upload all files to GCS
        uploaded_files = upload_to_gcs(video_id, files)

        return {
            "status": "success",
            "video_id": video_id,
            "files": files,
            "uploaded_files": uploaded_files
        }
    except Exception as e:
        logger.error(f"Error processing raw files: {str(e)}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process raw files for a Twitch VOD')
    parser.add_argument('url', type=str, help='Twitch VOD URL to process')
    args = parser.parse_args()

    try:
        # Process raw files
        result = asyncio.run(process_raw_files(args.url))

        # Print results
        logger.info(f"Successfully processed raw files for video {result['video_id']}")
        logger.info(f"Files saved to Output/Raw directory")
        logger.info(f"Files uploaded to GCS: {len(result['uploaded_files'])}")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise
