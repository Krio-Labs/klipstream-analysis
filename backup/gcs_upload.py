"""
Google Cloud Storage Upload Module

This module handles uploading files to Google Cloud Storage buckets with appropriate folder structures.
Files are organized into three buckets:
- klipstream-vods-raw: For video, audio, and waveform files
- klipstream-transcripts: For transcript paragraph and word files
- klipstream-chatlogs: For chat logs

Each bucket uses a folder structure based on the VOD ID. For example, for a video with ID '123456789':
- klipstream-vods-raw/123456789/123456789.mp4 - The video file
- klipstream-vods-raw/123456789/audio_123456789.wav - The audio file
- klipstream-vods-raw/123456789/audio_123456789_waveform.json - The waveform file
- klipstream-transcripts/123456789/audio_123456789_paragraphs.csv - The transcript paragraphs
- klipstream-transcripts/123456789/audio_123456789_words.csv - The transcript words
- klipstream-chatlogs/123456789/123456789_chat.csv - The chat log
"""

import os
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GCS bucket names
VOD_BUCKET = os.environ.get('GCS_VOD_BUCKET', 'klipstream-vods-raw')
TRANSCRIPT_BUCKET = os.environ.get('GCS_TRANSCRIPT_BUCKET', 'klipstream-transcripts')
CHATLOG_BUCKET = os.environ.get('GCS_CHATLOG_BUCKET', 'klipstream-chatlogs')
GCS_PROJECT = os.environ.get('GCS_PROJECT', 'klipstream')

# Define the directories - use local directories instead of /tmp
OUTPUTS_DIR = os.environ.get('OUTPUTS_DIR', 'outputs')
DATA_DIR = os.environ.get('DATA_DIR', 'data')
RAW_DIR = Path("Output/Raw")
RAW_VIDEOS_DIR = RAW_DIR / "Videos"
RAW_AUDIO_DIR = RAW_DIR / "Audio"
RAW_TRANSCRIPTS_DIR = RAW_DIR / "Transcripts"
RAW_WAVEFORMS_DIR = RAW_DIR / "Waveforms"
RAW_CHAT_DIR = RAW_DIR / "Chat"

# Create directories if they don't exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(str(RAW_DIR), exist_ok=True)
os.makedirs(str(RAW_VIDEOS_DIR), exist_ok=True)
os.makedirs(str(RAW_AUDIO_DIR), exist_ok=True)
os.makedirs(str(RAW_TRANSCRIPTS_DIR), exist_ok=True)
os.makedirs(str(RAW_WAVEFORMS_DIR), exist_ok=True)
os.makedirs(str(RAW_CHAT_DIR), exist_ok=True)

def get_bucket_for_file(filename: str, video_id: str) -> str:
    """Determine which bucket a file should be uploaded to based on its type"""
    file_type = get_file_type(filename, video_id)

    # Video, audio, and waveform files go to VOD_BUCKET
    if file_type in ['video', 'audio', 'audiowave']:
        return VOD_BUCKET

    # Transcript files go to TRANSCRIPT_BUCKET
    elif file_type in ['transcript', 'transcriptWord']:
        return TRANSCRIPT_BUCKET

    # Chat files go to CHATLOG_BUCKET
    elif file_type in ['chat', 'chatAnalysis', 'chatSentiment']:
        return CHATLOG_BUCKET

    # Default to VOD_BUCKET for other files
    return VOD_BUCKET

def get_file_type(filename: str, video_id: str) -> str:
    """Determine the file type based on filename pattern"""
    if filename == f"{video_id}_chat.csv":
        return 'chat'
    elif filename == f"{video_id}_chat_analysis.csv":
        return 'chatAnalysis'
    elif filename == f"audio_{video_id}_waveform.json":
        return 'audiowave'
    elif filename == f"audio_{video_id}_paragraphs.csv":
        return 'transcript'
    elif filename == f"audio_{video_id}_words.csv":
        return 'transcriptWord'
    elif filename == f"{video_id}_chat_sentiment.csv":
        return 'chatSentiment'
    elif filename == f"audio_{video_id}.wav":
        return 'audio'
    elif filename.endswith('.mp4'):
        return 'video'
    return 'other'

def upload_file_to_gcs(file_path: str, video_id: str) -> Dict:
    """Upload a file to the appropriate GCS bucket"""
    try:
        # Initialize GCS client
        storage_client = storage.Client(project=GCS_PROJECT)

        # Get filename from path
        filename = os.path.basename(file_path)

        # Determine which bucket to use
        bucket_name = get_bucket_for_file(filename, video_id)
        bucket = storage_client.bucket(bucket_name)

        # Create folder structure based on video_id
        blob_path = f"{video_id}/{filename}"
        blob = bucket.blob(blob_path)

        # Detect content type based on file extension
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type:
            blob.content_type = content_type
            logger.debug(f"Set content type for {filename} to {content_type}")

        # Add metadata for tracking
        blob.metadata = {
            "uploaded_by": "twitch-analysis-service",
            "upload_timestamp": datetime.now().isoformat(),
            "video_id": video_id,
            "file_type": get_file_type(filename, video_id)
        }

        # Upload the file
        blob.upload_from_filename(file_path)

        # Get public URL
        url = f"gs://{bucket_name}/{blob_path}"

        logger.info(f"File {filename} uploaded to {url}")

        return {
            "filename": filename,
            "bucket": bucket_name,
            "path": blob_path,
            "url": url,
            "content_type": content_type
        }

    except Exception as e:
        logger.error(f"Error uploading file {file_path} to GCS: {str(e)}")
        return None

def upload_files(video_id: str, specific_files: List[str] = None) -> List[Dict]:
    """
    Upload only the 6 specific raw files for a video to GCS:
    1. Video file (.mp4)
    2. Audio file (.wav)
    3. Waveform file (.json)
    4. Transcript paragraphs file (.csv)
    5. Transcript words file (.csv)
    6. Chat file (.csv)

    Args:
        video_id: The Twitch video ID
        specific_files: Optional list of specific file paths to upload

    Returns:
        List of dictionaries with information about uploaded files
    """
    uploaded_files = []

    # Define the 6 specific files we want to upload from Output/Raw
    raw_files_to_upload = [
        # 1. Video file
        str(RAW_VIDEOS_DIR / f"{video_id}.mp4"),
        # 2. Audio file
        str(RAW_AUDIO_DIR / f"audio_{video_id}.wav"),
        # 3. Waveform file
        str(RAW_WAVEFORMS_DIR / f"audio_{video_id}_waveform.json"),
        # 4. Transcript paragraphs file
        str(RAW_TRANSCRIPTS_DIR / f"audio_{video_id}_paragraphs.csv"),
        # 5. Transcript words file
        str(RAW_TRANSCRIPTS_DIR / f"audio_{video_id}_words.csv"),
        # 6. Chat file
        str(RAW_CHAT_DIR / f"{video_id}_chat.csv")
    ]

    # If specific files are provided, filter them to ensure they match our expected files
    if specific_files:
        # Only upload files that are in our list of raw files
        filtered_files = []
        for file_path in specific_files:
            # Check if this file matches any of our expected raw files
            for raw_file in raw_files_to_upload:
                if os.path.basename(file_path) == os.path.basename(raw_file):
                    filtered_files.append(file_path)
                    break

        # Use the filtered list
        files_to_check = filtered_files
    else:
        # Use our predefined list
        files_to_check = raw_files_to_upload

    # Upload each file if it exists
    for file_path in files_to_check:
        if os.path.exists(file_path):
            logger.info(f"Uploading raw file: {file_path}")
            result = upload_file_to_gcs(file_path, video_id)
            if result:
                uploaded_files.append(result)

    # Log summary
    logger.info(f"Uploaded {len(uploaded_files)} raw files for video {video_id}")

    return uploaded_files

def update_video_status(video_id: str, status: str, twitch_info: dict = None) -> bool:
    """
    Update the video status in the database

    This function would typically update a database record, but for now
    it just logs the status change. In a real implementation, this would
    update a database with the GCS file URLs.
    """
    try:
        logger.info(f"Updating video {video_id} status to {status}")
        if twitch_info:
            logger.info(f"Twitch info: {twitch_info}")

        # In a real implementation, this would update a database
        # For now, just return success
        return True

    except Exception as e:
        logger.error(f"Error updating video status: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the module
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gcs_upload.py <video_id>")
        sys.exit(1)

    video_id = sys.argv[1]
    results = upload_files(video_id)

    print(f"Uploaded {len(results)} files for video {video_id}")
    for result in results:
        print(f"- {result['filename']} -> {result['url']}")
