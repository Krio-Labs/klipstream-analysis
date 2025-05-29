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

        # Check if we're running in Cloud Run (which uses /tmp as writable directory)
        self.in_cloud_run = os.environ.get('K_SERVICE') is not None

        # If we're in Cloud Run and base_dir doesn't already include /tmp, prepend it
        if self.in_cloud_run and str(self.base_dir) == '.':
            self.base_dir = Path('/tmp')
            logger.info(f"Running in Cloud Run, using base directory: {self.base_dir}")

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

        path = path_mapping.get(file_type)

        # If we're in Cloud Run but the path doesn't start with /tmp, check if it exists at /tmp
        if path and self.in_cloud_run and not str(path).startswith('/tmp'):
            tmp_path = Path(f"/tmp/{path}")
            if tmp_path.exists():
                logger.info(f"Found file at {tmp_path} instead of {path}")
                return tmp_path

        return path

    def get_gcs_path(self, file_type: str) -> str:
        """
        Get the GCS path for a file type.

        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)

        Returns:
            str: GCS path for the file (without bucket name)
        """
        # Define GCS path mapping
        # Use VOD ID as the top-level folder for organization
        gcs_path_mapping = {
            # VODs bucket - organize by media type
            "video": f"{self.video_id}/video/{self.video_id}.mp4",
            "audio": f"{self.video_id}/audio/audio_{self.video_id}.wav",
            "waveform": f"{self.video_id}/waveform/audio_{self.video_id}_waveform.json",

            # Transcripts bucket - organize by VOD ID
            "transcript": f"{self.video_id}/audio_{self.video_id}_transcript.json",
            "segments": f"{self.video_id}/audio_{self.video_id}_segments.csv",
            "words": f"{self.video_id}/audio_{self.video_id}_words.csv",
            "paragraphs": f"{self.video_id}/audio_{self.video_id}_paragraphs.csv",

            # Chat logs bucket
            "chat": f"{self.video_id}/{self.video_id}_chat.csv",

            # Analysis bucket
            "audio_sentiment": f"{self.video_id}/audio/audio_{self.video_id}_sentiment.csv",
            "chat_sentiment": f"{self.video_id}/chat/{self.video_id}_chat_sentiment.csv",
            "highlights": f"{self.video_id}/audio/audio_{self.video_id}_highlights.csv",
            "integrated_analysis": f"{self.video_id}/integrated_{self.video_id}.json"
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
        This method checks multiple possible locations for the file.

        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            download_if_missing (bool, optional): Whether to download from GCS if missing locally. Defaults to True.

        Returns:
            Path: Path to the file, or None if not found
        """
        # Get the standard local path
        local_path = self.get_local_path(file_type)

        # Check if file exists at the standard path
        if local_path and local_path.exists():
            logger.info(f"Found {file_type} at standard path: {local_path}")
            return local_path

        # If we're in Cloud Run, check both with and without /tmp prefix
        if self.in_cloud_run:
            # Check with /tmp prefix if not already there
            if local_path and not str(local_path).startswith('/tmp'):
                tmp_path = Path(f"/tmp/{local_path}")
                if tmp_path.exists():
                    logger.info(f"Found {file_type} at Cloud Run path: {tmp_path}")
                    return tmp_path

            # Check without /tmp prefix if it's already there
            if local_path and str(local_path).startswith('/tmp'):
                non_tmp_path = Path(str(local_path).replace('/tmp/', '', 1))
                if non_tmp_path.exists():
                    logger.info(f"Found {file_type} at non-tmp path: {non_tmp_path}")
                    return non_tmp_path

        # Try alternative paths based on file type
        alt_paths = []

        if file_type == "audio_sentiment":
            # Try different casing and locations
            alt_paths = [
                Path(f"/tmp/output/Analysis/audio/audio_{self.video_id}_sentiment.csv"),
                Path(f"/tmp/output/Analysis/Audio/audio_{self.video_id}_sentiment.csv"),
                Path(f"output/Analysis/audio/audio_{self.video_id}_sentiment.csv"),
                Path(f"output/Analysis/Audio/audio_{self.video_id}_sentiment.csv")
            ]
        elif file_type == "chat_sentiment":
            # Try different naming patterns
            alt_paths = [
                Path(f"/tmp/output/Analysis/Chat/{self.video_id}_chat_sentiment.csv"),
                Path(f"/tmp/output/Analysis/chat/{self.video_id}_chat_sentiment.csv"),
                Path(f"output/Analysis/Chat/{self.video_id}_chat_sentiment.csv"),
                Path(f"output/Analysis/chat/{self.video_id}_chat_sentiment.csv")
            ]

        # Check alternative paths
        for alt_path in alt_paths:
            if alt_path.exists():
                logger.info(f"Found {file_type} at alternative path: {alt_path}")
                return alt_path

        # If download_if_missing is False, don't try to download from GCS
        if not download_if_missing or not self.use_gcs:
            logger.warning(f"Could not find {file_type} locally and download_if_missing is {download_if_missing}")
            return None

        # Try to download from GCS
        logger.info(f"Attempting to download {file_type} from GCS")
        if self.download_from_gcs(file_type):
            # Get the path again after download
            downloaded_path = self.get_local_path(file_type)
            if downloaded_path and downloaded_path.exists():
                logger.info(f"Successfully downloaded {file_type} to {downloaded_path}")
                return downloaded_path
            else:
                logger.warning(f"Downloaded {file_type} but file not found at expected path")

        logger.warning(f"Could not find or download {file_type}")
        return None

    def download_from_gcs(self, file_type: str, max_retries: int = 3, retry_delay: int = 5,
                       chunk_size: int = 5 * 1024 * 1024) -> bool:
        """
        Download a file from GCS to local storage with retry logic and chunked downloads.

        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.
            chunk_size (int, optional): Size of chunks for large file downloads in bytes. Defaults to 5MB.

        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.use_gcs:
            return False

        try:
            from google.cloud import storage
            from google.cloud.exceptions import GoogleCloudError

            local_path = self.get_local_path(file_type)
            gcs_path = self.get_gcs_path(file_type)
            bucket_name = self.get_bucket_name(file_type)

            if not local_path or not gcs_path or not bucket_name:
                return False

            # Use service account credentials instead of application default credentials
            import os
            service_account_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'new-service-account-key.json')

            if os.path.exists(service_account_path):
                logger.info(f"Using service account credentials from {service_account_path}")
                client = storage.Client.from_service_account_json(service_account_path, project=GCS_PROJECT)
            else:
                logger.warning(f"Service account key file not found at {service_account_path}, falling back to application default credentials")
                client = storage.Client(project=GCS_PROJECT)

            bucket = client.bucket(bucket_name)

            # Try to download with retries and exponential backoff
            for attempt in range(max_retries):
                try:
                    # Get blob with appropriate chunk size for large files
                    blob = bucket.blob(gcs_path, chunk_size=chunk_size)

                    # Check if blob exists
                    if not blob.exists():
                        logger.warning(f"File {gcs_path} does not exist in bucket {bucket_name}")
                        return False

                    # Get blob size for logging and timeout calculation
                    blob.reload()
                    file_size = blob.size if hasattr(blob, 'size') else 0
                    logger.info(f"Downloading file {gcs_path} ({file_size/1024/1024:.2f} MB) to {local_path}")

                    # Ensure directory exists
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    # Set timeout based on file size
                    timeout = max(300, int(file_size / (100 * 1024)))  # Scale timeout with file size

                    # Download with appropriate settings
                    try:
                        # First try with the newer API (google-cloud-storage >= 2.0.0)
                        from google.api_core.retry import Retry
                        blob.download_to_filename(
                            local_path,
                            timeout=timeout,
                            retry=Retry(
                                initial=1.0,
                                maximum=60.0,
                                multiplier=2.0,
                                deadline=timeout
                            )
                        )
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Retry import failed: {str(e)}, trying alternative method")
                        try:
                            # Try with DEFAULT_RETRY (google-cloud-storage >= 1.31.0)
                            from google.cloud.storage.retry import DEFAULT_RETRY
                            blob.download_to_filename(
                                local_path,
                                timeout=timeout,
                                retry=DEFAULT_RETRY
                            )
                        except (ImportError, AttributeError) as e:
                            logger.warning(f"DEFAULT_RETRY import failed: {str(e)}, downloading without retry")
                            # Fallback to no retry
                            blob.download_to_filename(
                                local_path,
                                timeout=timeout
                            )

                    # Verify file was downloaded correctly
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        logger.info(f"Successfully downloaded {gcs_path} to {local_path}")
                        return True
                    else:
                        raise Exception("Downloaded file is empty or does not exist")

                except GoogleCloudError as e:
                    # Handle specific Google Cloud errors
                    logger.warning(f"Download attempt {attempt+1} failed with Google Cloud error: {str(e)}")
                    if "Broken pipe" in str(e) or "Connection reset" in str(e) or "timed out" in str(e):
                        # Network errors - retry with longer delay
                        retry_delay = min(retry_delay * 2, 60)  # Exponential backoff, max 60 seconds

                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {max_retries} download attempts failed")
                        return False

                except Exception as e:
                    logger.warning(f"Download attempt {attempt+1} failed: {str(e)}")

                    # Clean up partial downloads
                    if os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                            logger.info(f"Removed partial download: {local_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up partial download: {str(cleanup_error)}")

                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {max_retries} download attempts failed")
                        return False

        except Exception as e:
            logger.error(f"Error downloading from GCS: {str(e)}")
            return False

    def upload_to_gcs(self, file_type: str, max_retries: int = 3, retry_delay: int = 5,
                    chunk_size: int = 5 * 1024 * 1024) -> bool:
        """
        Upload a file to GCS with retry logic and chunked uploads for large files.

        Args:
            file_type (str): Type of file (video, audio, transcript, etc.)
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.
            chunk_size (int, optional): Size of chunks for large file uploads in bytes. Defaults to 5MB.

        Returns:
            bool: True if upload was successful, False otherwise
        """
        if not self.use_gcs:
            return False

        try:
            # Import required modules
            import os
            from google.cloud import storage
            from google.cloud.exceptions import GoogleCloudError

            local_path = self.get_local_path(file_type)
            gcs_path = self.get_gcs_path(file_type)
            bucket_name = self.get_bucket_name(file_type)

            if not local_path or not gcs_path or not bucket_name:
                return False

            # Check if local file exists
            if not local_path.exists():
                logger.warning(f"Local file {local_path} does not exist")
                return False

            # Get file size
            file_size = os.path.getsize(local_path)
            logger.info(f"Uploading file {local_path} ({file_size/1024/1024:.2f} MB) to {bucket_name}/{gcs_path}")

            # Use service account credentials instead of application default credentials
            service_account_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'new-service-account-key.json')

            if os.path.exists(service_account_path):
                logger.info(f"Using service account credentials from {service_account_path}")
                client = storage.Client.from_service_account_json(service_account_path, project=GCS_PROJECT)
            else:
                logger.warning(f"Service account key file not found at {service_account_path}, falling back to application default credentials")
                client = storage.Client(project=GCS_PROJECT)

            bucket = client.bucket(bucket_name)

            # Large files will use resumable uploads automatically

            # Try to upload with retries and exponential backoff
            for attempt in range(max_retries):
                try:
                    # For large files, use resumable uploads with chunking
                    if file_size > 10 * 1024 * 1024:  # 10MB threshold for chunked upload
                        blob = bucket.blob(gcs_path, chunk_size=chunk_size)
                        logger.info(f"Using chunked upload with {chunk_size/1024/1024:.2f}MB chunks")
                    else:
                        blob = bucket.blob(gcs_path)

                    # Set timeout for large files
                    timeout = max(300, int(file_size / (100 * 1024)))  # Scale timeout with file size

                    # Upload with appropriate settings
                    try:
                        # First try with the newer API (google-cloud-storage >= 2.0.0)
                        from google.api_core.retry import Retry
                        blob.upload_from_filename(
                            local_path,
                            timeout=timeout,
                            if_generation_match=None,  # Avoid generation matching errors
                            retry=Retry(
                                initial=1.0,
                                maximum=60.0,
                                multiplier=2.0,
                                deadline=timeout
                            )
                        )
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Retry import failed: {str(e)}, trying alternative method")
                        try:
                            # Try with DEFAULT_RETRY (google-cloud-storage >= 1.31.0)
                            from google.cloud.storage.retry import DEFAULT_RETRY
                            blob.upload_from_filename(
                                local_path,
                                timeout=timeout,
                                if_generation_match=None,  # Avoid generation matching errors
                                retry=DEFAULT_RETRY
                            )
                        except (ImportError, AttributeError) as e:
                            logger.warning(f"DEFAULT_RETRY import failed: {str(e)}, uploading without retry")
                            # Fallback to no retry
                            blob.upload_from_filename(
                                local_path,
                                timeout=timeout,
                                if_generation_match=None  # Avoid generation matching errors
                            )

                    logger.info(f"Successfully uploaded {local_path} to {bucket_name}/{gcs_path}")
                    return True

                except GoogleCloudError as e:
                    # Handle specific Google Cloud errors
                    logger.warning(f"Upload attempt {attempt+1} failed with Google Cloud error: {str(e)}")
                    if "Broken pipe" in str(e) or "Connection reset" in str(e) or "timed out" in str(e):
                        # Network errors - retry with longer delay
                        retry_delay = min(retry_delay * 2, 60)  # Exponential backoff, max 60 seconds

                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {max_retries} upload attempts failed")
                        return False

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
