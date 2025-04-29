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

# Create directories if they don't exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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
    Upload all files for a video to GCS

    Args:
        video_id: The Twitch video ID
        specific_files: Optional list of specific file paths to upload

    Returns:
        List of dictionaries with information about uploaded files
    """
    uploaded_files = []

    # If specific files are provided, upload them first
    if specific_files:
        for file_path in specific_files:
            if os.path.exists(file_path):
                logger.info(f"Uploading specific file: {file_path}")
                result = upload_file_to_gcs(file_path, video_id)
                if result:
                    uploaded_files.append(result)

    # Check for video file - try both naming patterns
    video_file_patterns = [
        os.path.join(OUTPUTS_DIR, "video.mp4"),
        os.path.join(OUTPUTS_DIR, f"{video_id}.mp4"),
        os.path.join(OUTPUTS_DIR, f"video_{video_id}.mp4")
    ]

    for video_file in video_file_patterns:
        if os.path.exists(video_file):
            logger.info(f"Found video file: {video_file}")
            result = upload_file_to_gcs(video_file, video_id)
            if result:
                uploaded_files.append(result)
            break

    # Check for audio file
    audio_file = os.path.join(OUTPUTS_DIR, f"audio_{video_id}.wav")
    if os.path.exists(audio_file):
        result = upload_file_to_gcs(audio_file, video_id)
        if result:
            uploaded_files.append(result)

    # Check for waveform file
    waveform_file = os.path.join(OUTPUTS_DIR, f"audio_{video_id}_waveform.json")
    if os.path.exists(waveform_file):
        logger.info(f"Found waveform file: {waveform_file}")
        result = upload_file_to_gcs(waveform_file, video_id)
        if result:
            uploaded_files.append(result)

    # Check chat file from data directory
    chat_filename = f"{video_id}_chat.csv"
    chat_path = os.path.join(DATA_DIR, chat_filename)

    logger.debug(f"Checking for chat file at: {chat_path}")
    if os.path.exists(chat_path):
        result = upload_file_to_gcs(chat_path, video_id)
        if result:
            uploaded_files.append(result)

    # Check files from outputs directory
    logger.debug(f"Checking outputs directory: {OUTPUTS_DIR}")
    for filename in os.listdir(OUTPUTS_DIR):
        if filename.startswith(f"{video_id}_") or filename.startswith(f"audio_{video_id}_") or filename.startswith("top_"):
            file_path = os.path.join(OUTPUTS_DIR, filename)
            logger.debug(f"Processing file: {file_path}")
            result = upload_file_to_gcs(file_path, video_id)
            if result:
                uploaded_files.append(result)

    # Also check the local outputs directory
    local_outputs_dir = "outputs"
    if os.path.exists(local_outputs_dir) and local_outputs_dir != OUTPUTS_DIR:
        logger.debug(f"Checking local outputs directory: {local_outputs_dir}")
        for filename in os.listdir(local_outputs_dir):
            if filename.startswith(f"{video_id}_") or filename.startswith(f"audio_{video_id}_") or filename.startswith("top_"):
                file_path = os.path.join(local_outputs_dir, filename)
                logger.debug(f"Processing file: {file_path}")
                result = upload_file_to_gcs(file_path, video_id)
                if result:
                    uploaded_files.append(result)

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
