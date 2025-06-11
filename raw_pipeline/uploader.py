"""
Uploader Module

This module handles uploading files to Google Cloud Storage.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from google.oauth2 import service_account

from utils.config import (
    VODS_BUCKET,
    TRANSCRIPTS_BUCKET,
    CHATLOGS_BUCKET,
    RAW_VIDEOS_DIR,
    RAW_AUDIO_DIR,
    RAW_WAVEFORMS_DIR,
    RAW_TRANSCRIPTS_DIR,
    RAW_CHAT_DIR,
    GCP_SERVICE_ACCOUNT_PATH
)
from utils.logging_setup import setup_logger
from utils.convex_client_updated import ConvexManager
# Video processing status constants
STATUS_QUEUED = "Queued"
STATUS_DOWNLOADING = "Downloading"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_ANALYZING = "Analyzing"
STATUS_FINDING_HIGHLIGHTS = "Finding highlights"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

# Set up logger
logger = setup_logger("uploader", "gcs_upload.log")

def get_bucket_for_file(file_path: str) -> str:
    """
    Determine the appropriate GCS bucket for a file based on its type

    Args:
        file_path (str): Path to the file

    Returns:
        str: Name of the GCS bucket
    """
    file_path = str(file_path).lower()

    # Check for transcript files first (paragraphs, words, segments)
    if 'transcript' in file_path or 'paragraphs' in file_path or 'words' in file_path or 'segments' in file_path:
        return TRANSCRIPTS_BUCKET
    # Check for chat files
    elif 'chat' in file_path:
        return CHATLOGS_BUCKET
    # All other files (video, audio, waveform) go to VODS_BUCKET
    elif any(ext in file_path for ext in ['.mp4', '.wav', '.json']):
        return VODS_BUCKET

    # Default to VODS_BUCKET
    return VODS_BUCKET

def upload_file_to_gcs(file_path: str, video_id: str) -> Dict:
    """
    Upload a file to Google Cloud Storage

    Args:
        file_path (str): Path to the file to upload
        video_id (str): The Twitch video ID

    Returns:
        dict: Information about the uploaded file
    """
    try:
        file_path = Path(file_path)

        # Skip if file doesn't exist
        if not file_path.exists():
            logger.warning(f"File not found, skipping upload: {file_path}")
            return None

        # Determine bucket based on file type
        bucket_name = get_bucket_for_file(file_path)

        # Create GCS client using environment-based authentication
        try:
            if GCP_SERVICE_ACCOUNT_PATH and os.path.exists(GCP_SERVICE_ACCOUNT_PATH):
                logger.info(f"Using service account credentials from {GCP_SERVICE_ACCOUNT_PATH}")
                credentials = service_account.Credentials.from_service_account_file(
                    GCP_SERVICE_ACCOUNT_PATH,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                client = storage.Client(credentials=credentials)
            else:
                # Use application default credentials (recommended for Cloud Run)
                logger.info("Using application default credentials (Cloud Run service account)")
                client = storage.Client()
        except Exception as e:
            logger.warning(f"Failed to initialize GCS client: {str(e)}")
            logger.info("Falling back to application default credentials")
            client = storage.Client()

        bucket = client.bucket(bucket_name)

        # Determine destination path in GCS
        blob_name = f"{video_id}/{file_path.name}"
        blob = bucket.blob(blob_name)

        # Upload file
        logger.info(f"Uploading {file_path} to gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(file_path)

        # Generate a GCS URI for the uploaded file
        gcs_uri = f"gs://{bucket_name}/{blob_name}"

        # Return information about the uploaded file
        return {
            "file_path": str(file_path),
            "bucket": bucket_name,
            "blob_name": blob_name,
            "gcs_uri": gcs_uri
        }

    except GoogleAPIError as e:
        logger.error(f"Google API error uploading {file_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error uploading {file_path}: {str(e)}")
        return None

def upload_files(video_id: str, specific_files: List[str] = None) -> List[Dict]:
    """
    Upload only the 7 specific raw files for a video to GCS:
    1. Video file (.mp4)
    2. Audio file (.wav)
    3. Waveform file (.json)
    4. Transcript paragraphs file (.csv)
    5. Transcript words file (.csv)
    6. Transcript segments file (.csv)
    7. Chat file (.csv)

    Args:
        video_id: The Twitch video ID
        specific_files: Optional list of specific file paths to upload

    Returns:
        List of dictionaries with information about uploaded files
    """
    uploaded_files = []

    # Define the 7 specific files we want to upload from Output/Raw
    raw_files_to_upload = [
        # 1. Video file
        str(RAW_VIDEOS_DIR / f"{video_id}.mp4"),
        # 2. Audio file (MP3 format for compression)
        str(RAW_AUDIO_DIR / f"audio_{video_id}.mp3"),
        # 3. Waveform file
        str(RAW_WAVEFORMS_DIR / f"audio_{video_id}_waveform.json"),
        # 4. Transcript paragraphs file
        str(RAW_TRANSCRIPTS_DIR / f"audio_{video_id}_paragraphs.csv"),
        # 5. Transcript words file
        str(RAW_TRANSCRIPTS_DIR / f"audio_{video_id}_words.csv"),
        # 6. Transcript segments file
        str(RAW_TRANSCRIPTS_DIR / f"audio_{video_id}_segments.csv"),
        # 7. Chat file
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

def upload_to_gcs(video_id, files):
    """
    Upload only the 7 specific raw files to Google Cloud Storage
    """
    # Initialize Convex client
    convex_manager = ConvexManager()

    try:
        logger.info(f"Uploading raw files to GCS for {video_id}...")

        # Check if service account key file is properly configured
        if GCP_SERVICE_ACCOUNT_PATH and os.path.exists(GCP_SERVICE_ACCOUNT_PATH):
            try:
                with open(GCP_SERVICE_ACCOUNT_PATH, 'r') as f:
                    key_data = json.load(f)
                    if key_data.get('private_key') == "REPLACE_WITH_ACTUAL_PRIVATE_KEY":
                        logger.warning("Service account key file contains placeholder values.")
                        logger.warning("Please update the key file with actual service account credentials.")
                        logger.warning("See decision_docs/gcp_authentication.md for instructions.")
                        logger.info("Skipping GCS upload due to invalid service account key.")
                        return []
            except Exception as e:
                logger.warning(f"Error reading service account key file: {str(e)}")

        # Define the 7 specific files we want to upload
        raw_files_to_upload = {
            "video_file": files.get("video_file"),
            "audio_file": files.get("audio_file"),
            "waveform_file": files.get("waveform_file"),
            "paragraphs_file": files.get("paragraphs_file"),
            "words_file": files.get("words_file"),
            "segments_file": files.get("segments_file"),
            "chat_file": files.get("chat_file")
        }

        # Filter out any None values and convert to strings
        file_paths = [str(file) for file in raw_files_to_upload.values() if file is not None and isinstance(file, Path)]

        # Upload files
        uploaded_files = upload_files(video_id, file_paths)

        # Verify we have exactly 7 files
        if len(uploaded_files) != 7:
            logger.warning(f"Expected to upload 7 files, but uploaded {len(uploaded_files)} files")

        logger.info(f"Successfully uploaded {len(uploaded_files)} raw files to GCS")

        # After uploading files, update Convex with URLs using CORRECT field names
        url_updates = {}

        # Check for video file (MP4)
        video_result = next((r for r in uploaded_files if r["file_path"].endswith(".mp4")), None)
        if video_result:
            url_updates["video_url"] = video_result["gcs_uri"]

        # Check for audio file (MP3 or WAV)
        audio_result = next((r for r in uploaded_files if r["file_path"].endswith(".mp3") or r["file_path"].endswith(".wav")), None)
        if audio_result:
            url_updates["audio_url"] = audio_result["gcs_uri"]

        # Check for transcript segments file (CSV)
        transcript_segments_result = next((r for r in uploaded_files if "segments.csv" in r["file_path"]), None)
        if transcript_segments_result:
            url_updates["transcript_url"] = transcript_segments_result["gcs_uri"]

        # Check for transcript words file (CSV)
        transcript_words_result = next((r for r in uploaded_files if "words.csv" in r["file_path"]), None)
        if transcript_words_result:
            url_updates["transcriptWords_url"] = transcript_words_result["gcs_uri"]

        # Check for chat file (CSV)
        chat_result = next((r for r in uploaded_files if "_chat.csv" in r["file_path"]), None)
        if chat_result:
            url_updates["chat_url"] = chat_result["gcs_uri"]

        # Check for waveform file (JSON)
        waveform_result = next((r for r in uploaded_files if "waveform.json" in r["file_path"]), None)
        if waveform_result:
            url_updates["waveform_url"] = waveform_result["gcs_uri"]

        # Update Convex if we have any URLs to update
        if url_updates:
            logger.info(f"Updating Convex with {len(url_updates)} URLs: {list(url_updates.keys())}")
            success = convex_manager.update_video_urls(video_id, url_updates)
            if success:
                logger.info("✅ Successfully updated Convex with raw pipeline URLs")
            else:
                logger.warning("⚠️ Failed to update Convex with raw pipeline URLs")
        else:
            logger.info("No URLs to update in Convex from raw pipeline")

        return uploaded_files
    except Exception as e:
        logger.warning(f"Error uploading files to GCS: {str(e)}")
        logger.warning("Continuing without GCS upload.")
        logger.warning("To fix this issue, either:")
        logger.warning("1. Run: gcloud auth application-default login")
        logger.warning("2. Or update the service account key file with valid credentials")
        logger.warning("See decision_docs/gcp_authentication.md for detailed instructions")
        # Return empty list instead of raising an exception
        return []
