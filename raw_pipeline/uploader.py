"""
Uploader Module

This module handles uploading files to Google Cloud Storage.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

from utils.config import (
    VODS_BUCKET,
    TRANSCRIPTS_BUCKET,
    CHATLOGS_BUCKET,
    RAW_VIDEOS_DIR,
    RAW_AUDIO_DIR,
    RAW_WAVEFORMS_DIR,
    RAW_TRANSCRIPTS_DIR,
    RAW_CHAT_DIR
)
from utils.logging_setup import setup_logger

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
    
    if any(ext in file_path for ext in ['.mp4', '.wav', '.json']):
        if 'waveform' in file_path:
            return VODS_BUCKET
        elif 'audio' in file_path or 'video' in file_path:
            return VODS_BUCKET
        elif 'transcript' in file_path or 'paragraphs' in file_path or 'words' in file_path:
            return TRANSCRIPTS_BUCKET
    elif 'chat' in file_path:
        return CHATLOGS_BUCKET
    
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
        
        # Create GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Determine destination path in GCS
        blob_name = f"{video_id}/{file_path.name}"
        blob = bucket.blob(blob_name)
        
        # Upload file
        logger.info(f"Uploading {file_path} to gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(file_path)
        
        # Make the blob publicly readable
        blob.make_public()
        
        # Return information about the uploaded file
        return {
            "file_path": str(file_path),
            "bucket": bucket_name,
            "blob_name": blob_name,
            "public_url": blob.public_url
        }
    
    except GoogleAPIError as e:
        logger.error(f"Google API error uploading {file_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error uploading {file_path}: {str(e)}")
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
