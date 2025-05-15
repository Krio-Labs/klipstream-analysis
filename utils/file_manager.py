"""
File Manager Module

This module provides a unified interface for file operations, handling both local and GCS storage.
It includes retry logic and fallback mechanisms for improved reliability.
"""

import os
import time
from pathlib import Path
import logging
from typing import Optional, Dict, List, Union, Tuple

from utils.config import (
    BASE_DIR, 
    USE_GCS, 
    GCS_PROJECT,
    VODS_BUCKET,
    TRANSCRIPTS_BUCKET,
    CHATLOGS_BUCKET,
    ANALYSIS_BUCKET
)

# Set up logger
logger = logging.getLogger(__name__)

class FileManager:
    """
    File Manager class for handling file operations with GCS integration.
    
    This class provides methods for:
    - Getting file paths (local or GCS)
    - Checking if files exist
    - Downloading files from GCS
    - Uploading files to GCS
    - Listing files
    
    It handles both local and GCS storage, with retry logic and fallback mechanisms.
    """
    
    def __init__(self, video_id: str, base_dir: Optional[Path] = None):
        """
        Initialize the FileManager.
        
        Args:
            video_id (str): The video ID to manage files for
            base_dir (Path, optional): Base directory for local files. Defaults to config.BASE_DIR.
        """
        self.video_id = video_id
        self.base_dir = base_dir or BASE_DIR
        self.use_gcs = USE_GCS
        
        # Define bucket mapping
        self.bucket_mapping = {
            "video": VODS_BUCKET,
            "audio": VODS_BUCKET,
            "transcript": TRANSCRIPTS_BUCKET,
            "segments": TRANSCRIPTS_BUCKET,
            "words": TRANSCRIPTS_BUCKET,
            "paragraphs": TRANSCRIPTS_BUCKET,
            "waveform": VODS_BUCKET,
            "chat": CHATLOGS_BUCKET,
            "audio_sentiment": ANALYSIS_BUCKET,
            "chat_sentiment": ANALYSIS_BUCKET,
            "highlights": ANALYSIS_BUCKET,
            "integrated_analysis": ANALYSIS_BUCKET
        }
        
    def get_local_path(self, file_type: str) -> Path:
        """
        Get the local path for a file type.
        
        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            
        Returns:
            Path: Local path for the file
        """
        # Define path mapping
        path_mapping = {
            "video": self.base_dir / f"output/Raw/Videos/{self.video_id}.mp4",
            "audio": self.base_dir / f"output/Raw/Audio/audio_{self.video_id}.wav",
            "transcript": self.base_dir / f"output/Raw/Transcripts/audio_{self.video_id}_transcript.json",
            "segments": self.base_dir / f"output/Raw/Transcripts/audio_{self.video_id}_segments.csv",
            "words": self.base_dir / f"output/Raw/Transcripts/audio_{self.video_id}_words.csv",
            "paragraphs": self.base_dir / f"output/Raw/Transcripts/audio_{self.video_id}_paragraphs.csv",
            "waveform": self.base_dir / f"output/Raw/Waveforms/audio_{self.video_id}_waveform.json",
            "chat": self.base_dir / f"output/Raw/Chat/{self.video_id}_chat.csv",
            "audio_sentiment": self.base_dir / f"output/Analysis/Audio/audio_{self.video_id}_sentiment.csv",
            "chat_sentiment": self.base_dir / f"output/Analysis/Chat/{self.video_id}_chat_sentiment.csv",
            "highlights": self.base_dir / f"output/Analysis/Audio/audio_{self.video_id}_highlights.csv",
            "integrated_analysis": self.base_dir / f"output/Analysis/integrated_{self.video_id}.json"
        }
        
        return path_mapping.get(file_type)
    
    def get_gcs_path(self, file_type: str) -> str:
        """
        Get the GCS path for a file type.
        
        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            
        Returns:
            str: GCS path for the file (without bucket name)
        """
        # Define GCS path mapping
        gcs_path_mapping = {
            "video": f"videos/{self.video_id}.mp4",
            "audio": f"audio/audio_{self.video_id}.wav",
            "transcript": f"transcripts/audio_{self.video_id}_transcript.json",
            "segments": f"transcripts/audio_{self.video_id}_segments.csv",
            "words": f"transcripts/audio_{self.video_id}_words.csv",
            "paragraphs": f"transcripts/audio_{self.video_id}_paragraphs.csv",
            "waveform": f"waveforms/audio_{self.video_id}_waveform.json",
            "chat": f"chat/{self.video_id}_chat.csv",
            "audio_sentiment": f"audio/audio_{self.video_id}_sentiment.csv",
            "chat_sentiment": f"chat/{self.video_id}_chat_sentiment.csv",
            "highlights": f"audio/audio_{self.video_id}_highlights.csv",
            "integrated_analysis": f"integrated/integrated_{self.video_id}.json"
        }
        
        return gcs_path_mapping.get(file_type)
    
    def get_bucket_name(self, file_type: str) -> str:
        """
        Get the bucket name for a file type.
        
        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            
        Returns:
            str: Bucket name for the file
        """
        return self.bucket_mapping.get(file_type)
    
    def file_exists_locally(self, file_type: str) -> bool:
        """
        Check if a file exists locally.
        
        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            
        Returns:
            bool: True if the file exists locally, False otherwise
        """
        local_path = self.get_local_path(file_type)
        return local_path.exists() if local_path else False
    
    def file_exists_in_gcs(self, file_type: str) -> bool:
        """
        Check if a file exists in GCS.
        
        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            
        Returns:
            bool: True if the file exists in GCS, False otherwise
        """
        if not self.use_gcs:
            return False
            
        try:
            from google.cloud import storage
            
            gcs_path = self.get_gcs_path(file_type)
            bucket_name = self.get_bucket_name(file_type)
            
            if not gcs_path or not bucket_name:
                return False
                
            client = storage.Client(project=GCS_PROJECT)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)
            
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking if file exists in GCS: {str(e)}")
            return False
    
    def get_file_path(self, file_type: str, download_if_missing: bool = True) -> Optional[Path]:
        """
        Get the path to a file, downloading from GCS if necessary.
        
        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            download_if_missing (bool, optional): Whether to download from GCS if missing locally. Defaults to True.
            
        Returns:
            Path: Path to the file, or None if not found
        """
        local_path = self.get_local_path(file_type)
        
        # If file exists locally, return it
        if local_path and local_path.exists():
            return local_path
            
        # If download_if_missing is False, don't try to download from GCS
        if not download_if_missing or not self.use_gcs:
            return None
            
        # Try to download from GCS
        if self.download_from_gcs(file_type):
            return local_path
            
        return None
    
    def download_from_gcs(self, file_type: str, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Download a file from GCS to local storage with retry logic.
        
        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.use_gcs:
            return False
            
        try:
            from google.cloud import storage
            
            local_path = self.get_local_path(file_type)
            gcs_path = self.get_gcs_path(file_type)
            bucket_name = self.get_bucket_name(file_type)
            
            if not local_path or not gcs_path or not bucket_name:
                return False
                
            client = storage.Client(project=GCS_PROJECT)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)
            
            # Check if blob exists
            if not blob.exists():
                logger.warning(f"File {gcs_path} does not exist in bucket {bucket_name}")
                return False
                
            # Ensure directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to download with retries
            for attempt in range(max_retries):
                try:
                    blob.download_to_filename(local_path)
                    logger.info(f"Downloaded {gcs_path} to {local_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Download attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {max_retries} download attempts failed")
                        return False
                        
        except Exception as e:
            logger.error(f"Error downloading from GCS: {str(e)}")
            return False
            
    def upload_to_gcs(self, file_type: str, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Upload a file to GCS with retry logic.
        
        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.
            
        Returns:
            bool: True if upload was successful, False otherwise
        """
        if not self.use_gcs:
            return False
            
        try:
            from google.cloud import storage
            
            local_path = self.get_local_path(file_type)
            gcs_path = self.get_gcs_path(file_type)
            bucket_name = self.get_bucket_name(file_type)
            
            if not local_path or not gcs_path or not bucket_name:
                return False
                
            # Check if local file exists
            if not local_path.exists():
                logger.warning(f"Local file {local_path} does not exist")
                return False
                
            client = storage.Client(project=GCS_PROJECT)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)
            
            # Try to upload with retries
            for attempt in range(max_retries):
                try:
                    blob.upload_from_filename(local_path)
                    logger.info(f"Uploaded {local_path} to {bucket_name}/{gcs_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Upload attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {max_retries} upload attempts failed")
                        return False
                        
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            return False
