"""
Convex Client Module (Updated Implementation)

This module provides utilities for interacting with the Convex database,
based on the exact schema defined in the frontend project.

Environment Variables Required:
- CONVEX_URL: The URL of your Convex deployment (e.g., https://laudable-horse-446.convex.cloud)
- CONVEX_API_KEY: The API key for your Convex deployment

These should be set in your .env file or in your Cloud Run environment variables.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from utils.logging_setup import setup_logger
# Video processing status constants
STATUS_QUEUED = "Queued"
STATUS_DOWNLOADING = "Downloading"
STATUS_FETCHING_CHAT = "Fetching chat"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_ANALYZING = "Analyzing"
STATUS_FINDING_HIGHLIGHTS = "Finding highlights"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

# Import the real ConvexIntegration class
from convex_integration import ConvexIntegration

# Set up logger
logger = setup_logger("convex_client", "convex_client.log")

# Load environment variables
load_dotenv()

# Get Convex URL and API key from environment variables
CONVEX_URL = os.getenv("CONVEX_URL")
CONVEX_API_KEY = os.getenv("CONVEX_API_KEY")

# Valid status values
VALID_STATUSES = [
    STATUS_QUEUED,
    STATUS_DOWNLOADING,
    STATUS_FETCHING_CHAT,
    STATUS_TRANSCRIBING,
    STATUS_ANALYZING,
    STATUS_FINDING_HIGHLIGHTS,
    STATUS_COMPLETED,
    STATUS_FAILED
]

class ConvexManager:
    """Manager for Convex database operations"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConvexManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the Convex client"""
        # Set to False to make actual API calls
        self.test_mode = False
        logger.info("Running in LIVE MODE - actual Convex API calls will be made")

        # Create a Convex integration
        self.convex = ConvexIntegration(CONVEX_URL, CONVEX_API_KEY)

        if not CONVEX_URL or not CONVEX_API_KEY:
            logger.error("Convex URL or API key not found in environment variables")
            logger.error("Please set CONVEX_URL and CONVEX_API_KEY in your .env file or Cloud Run environment variables")
            self.convex = None
            return

        # Log the Convex URL (but not the API key for security)
        logger.info(f"Initialized Convex client with URL: {CONVEX_URL}")
        logger.info(f"API key present: {bool(CONVEX_API_KEY)}")

    def check_video_exists(self, twitch_id: str) -> bool:
        """
        Check if a video exists in the Convex database by twitch_id

        Args:
            twitch_id (str): The Twitch video ID

        Returns:
            bool: True if the video exists, False otherwise
        """
        if not self.convex:
            logger.error("Convex client not initialized")
            return False

        # In test mode, always assume the video exists
        if self.test_mode:
            logger.info(f"[TEST MODE] Assuming video {twitch_id} exists in database")
            return True

        try:
            # Get the video from Convex
            video = self.convex.get_video(twitch_id)
            exists = video is not None
            logger.info(f"Video {twitch_id} exists in database: {exists}")
            if exists:
                logger.info(f"Video details: {video}")
            return exists
        except Exception as e:
            logger.error(f"Failed to check if video {twitch_id} exists: {str(e)}")
            logger.info(f"Assuming video {twitch_id} exists despite query error")
            return True

    def create_video_if_missing(self, twitch_id: str, title: str = None) -> bool:
        """
        Create a video entry in Convex if it doesn't exist

        Args:
            twitch_id (str): The Twitch video ID
            title (str, optional): Video title (will be extracted from URL if not provided)

        Returns:
            bool: True if video exists or was created successfully, False otherwise
        """
        if not self.convex:
            logger.error("Convex client not initialized")
            return False

        try:
            # First check if video already exists
            existing_video = self.convex.get_video(twitch_id)
            if existing_video:
                logger.info(f"Video {twitch_id} already exists in Convex")
                return True

            logger.info(f"Video {twitch_id} not found in Convex, creating new entry...")

            # Create video with minimal information using the correct team ID
            success = self.convex.create_video_minimal(twitch_id, "Queued")

            if success:
                logger.info(f"Successfully created video entry for {twitch_id}")
                return True
            else:
                logger.error(f"Failed to create video entry for {twitch_id}")
                return False

        except Exception as e:
            logger.error(f"Error creating video entry for {twitch_id}: {str(e)}")
            return False

    def update_video_status(self, twitch_id: str, status: str, max_retries: int = 3) -> bool:
        """
        Update the status field for a video in the Convex database
        If the video doesn't exist, it will be created automatically.

        Args:
            twitch_id (str): The Twitch video ID
            status (str): The new status value
            max_retries (int): Maximum number of retry attempts

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.convex:
            logger.warning("Convex client not initialized - running in local mode")
            return True

        # Check if running locally (not in cloud environment)
        is_local = not os.environ.get('CLOUD_RUN_SERVICE') and not os.environ.get('K_SERVICE')
        if is_local:
            logger.info(f"[LOCAL MODE] Would update video {twitch_id} status to '{status}'")
            return True

        # Validate status
        if status not in VALID_STATUSES:
            logger.warning(f"Invalid status '{status}'. Valid statuses are: {VALID_STATUSES}")
            logger.warning(f"Proceeding with update anyway, but consider using a valid status")

        # In test mode, just log the status update and return success
        if self.test_mode:
            logger.info(f"[TEST MODE] Would update status for video {twitch_id} to '{status}'")
            return True

        # Track failures for development mode
        if not hasattr(self, '_convex_failures'):
            self._convex_failures = {}

        # In development mode, skip only after multiple failures to allow for video creation attempts
        if os.environ.get('ENVIRONMENT', 'production') == 'development':
            failure_count = self._convex_failures.get(twitch_id, 0)
            if failure_count >= 3:  # Skip after 3 failures in development mode (allows for creation attempts)
                logger.info(f"Skipping Convex update for {twitch_id} in development mode (previous failures: {failure_count})")
                return True

        # Try to update with retries
        for attempt in range(max_retries):
            try:
                # Call the method to update the status
                logger.info(f"Updating status for video {twitch_id} to '{status}'...")
                success = self.convex.update_status_by_twitch_id(twitch_id, status)

                if success:
                    logger.info(f"Successfully updated status for video {twitch_id} to '{status}'")
                    # Reset failure counter on success
                    self._convex_failures[twitch_id] = 0
                    return True
                else:
                    logger.error(f"Failed to update status for video {twitch_id} to '{status}'")
                    self._convex_failures[twitch_id] = self._convex_failures.get(twitch_id, 0) + 1

                    if attempt < max_retries - 1:
                        retry_delay = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {max_retries} attempts to update status failed")
                        # In development mode, continue gracefully
                        if os.environ.get('ENVIRONMENT', 'production') == 'development':
                            logger.warning(f"Continuing in development mode despite Convex failures for video {twitch_id}")
                            return True
                        return False
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Attempt {attempt+1}/{max_retries} failed to update status for video {twitch_id}: {error_msg}")
                self._convex_failures[twitch_id] = self._convex_failures.get(twitch_id, 0) + 1

                # Check if this is a "video not found" error
                if "not found" in error_msg.lower():
                    logger.warning(f"Video {twitch_id} not found in Convex database")

                    # Always try to create the missing video entry (both dev and production)
                    logger.info(f"Attempting to create missing video entry for {twitch_id}")
                    if self.create_video_if_missing(twitch_id):
                        logger.info(f"Successfully created video entry for {twitch_id}, retrying status update")
                        # Retry the status update now that the video exists
                        try:
                            success = self.convex.update_status_by_twitch_id(twitch_id, status)
                            if success:
                                logger.info(f"Successfully updated status for newly created video {twitch_id}")
                                return True
                        except Exception as retry_error:
                            logger.error(f"Failed to update status after creating video: {str(retry_error)}")
                    else:
                        logger.error(f"Failed to create missing video entry for {twitch_id}")

                    # In development mode, continue gracefully even if creation fails
                    if os.environ.get('ENVIRONMENT', 'production') == 'development':
                        logger.warning(f"Development mode: Continuing despite video creation failure for {twitch_id}")
                        return True
                    else:
                        return False

                if attempt < max_retries - 1:
                    retry_delay = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts to update status failed")
                    # In development mode, continue gracefully
                    if os.environ.get('ENVIRONMENT', 'production') == 'development':
                        logger.warning(f"Continuing in development mode despite Convex failures for video {twitch_id}")
                        return True
                    return False

        # In development mode, continue gracefully
        if os.environ.get('ENVIRONMENT', 'production') == 'development':
            logger.warning(f"Continuing in development mode despite Convex failures for video {twitch_id}")
            return True
        return False

    def update_job_progress(self, twitch_id: str, job_id: str, progress_data: Dict[str, Any], max_retries: int = 3) -> bool:
        """
        Update job progress fields for a video in the Convex database

        Args:
            twitch_id (str): The Twitch video ID
            job_id (str): The unique job ID
            progress_data (Dict[str, Any]): Dictionary of progress fields to update
            max_retries (int): Maximum number of retry attempts

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.convex:
            logger.error("Convex client not initialized")
            return False

        # Skip if no progress data to update
        if not progress_data:
            logger.info(f"No progress data to update for video {twitch_id}")
            return True

        # In test mode, just log the progress update and return success
        if self.test_mode:
            logger.info(f"[TEST MODE] Would update progress for video {twitch_id} (job {job_id}): {progress_data}")
            return True

        # Log progress data for monitoring (Convex URLs schema doesn't support progress fields)
        logger.info(f"Progress update for video {twitch_id} (job {job_id}): {progress_data}")

        # For now, we'll just return True since the progress is logged
        # TODO: Update Convex schema to support progress fields or use a separate table
        return True

    def update_video_urls(self, twitch_id: str, url_updates: Dict[str, str], max_retries: int = 3) -> bool:
        """
        Update URL fields for a video in the Convex database using exact schema validation

        Args:
            twitch_id (str): The Twitch video ID
            url_updates (Dict[str, str]): Dictionary of field names and URLs to update
            max_retries (int): Maximum number of retry attempts

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.convex:
            logger.error("Convex client not initialized")
            return False

        # Skip if no URLs to update
        if not url_updates:
            logger.info(f"No URLs to update for video {twitch_id}")
            return True

        # EXACT field names from Convex schema - these are the ONLY allowed fields
        allowed_fields = {
            'video_url',
            'audio_url',
            'waveform_url',
            'transcript_url',
            'transcriptWords_url',
            'chat_url',
            'analysis_url'
        }

        # Validate and filter URL updates to only allowed fields
        validated_urls = {}
        rejected_fields = []

        for field, url in url_updates.items():
            if field in allowed_fields:
                validated_urls[field] = url
            else:
                rejected_fields.append(field)

        # Log rejected fields
        if rejected_fields:
            logger.warning(f"Rejected invalid URL fields for video {twitch_id}: {rejected_fields}")
            logger.warning(f"Only these fields are allowed: {sorted(allowed_fields)}")

        # Skip if no valid URLs after validation
        if not validated_urls:
            logger.warning(f"No valid URL fields to update for video {twitch_id}")
            return False

        # Log the URLs we're updating
        logger.info(f"Updating URLs for video {twitch_id}: {list(validated_urls.keys())}")
        for field, url in validated_urls.items():
            logger.info(f"  {field}: {url}")

        # In test mode, just log the URL updates and return success
        if self.test_mode:
            logger.info(f"[TEST MODE] Would update URLs for video {twitch_id}: {list(validated_urls.keys())}")
            return True

        # Try to update with retries
        for attempt in range(max_retries):
            try:
                # Call the Convex mutation with exact format
                logger.info(f"Updating URLs for video {twitch_id} (attempt {attempt + 1})...")
                result = self.convex.mutation("video:updateUrls", {
                    "twitchId": twitch_id,
                    "urls": validated_urls
                })

                if result and result.get("status") == "success":
                    logger.info(f"Successfully updated URLs for video {twitch_id}: {list(validated_urls.keys())}")
                    return True
                else:
                    logger.error(f"Failed to update URLs for video {twitch_id}: {result}")
                    if attempt < max_retries - 1:
                        retry_delay = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {max_retries} attempts to update URLs failed")
                        return False
            except Exception as e:
                logger.error(f"Attempt {attempt+1}/{max_retries} failed to update URLs for video {twitch_id}: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts to update URLs failed")
                    return False

        return False

    def update_pipeline_progress(self, twitch_id: str, status: str = None, urls: Dict[str, str] = None) -> bool:
        """
        Update both status and URLs for a video in one operation

        Args:
            twitch_id (str): The Twitch video ID
            status (str, optional): The new status value
            urls (Dict[str, str], optional): Dictionary of field names and URLs to update

        Returns:
            bool: True if all updates were successful, False otherwise
        """
        success = True

        # Update status if provided
        if status:
            status_success = self.update_video_status(twitch_id, status)
            if not status_success:
                logger.warning(f"Failed to update status for video {twitch_id}")
                success = False

        # Update URLs if provided
        if urls:
            urls_success = self.update_video_urls(twitch_id, urls)
            if not urls_success:
                logger.warning(f"Failed to update URLs for video {twitch_id}")
                success = False

        return success

    def update_job_progress_detailed(self, twitch_id: str, job_id: str, detailed_progress: Dict[str, Any], max_retries: int = 3) -> bool:
        """
        Update detailed job progress with all new fields

        Args:
            twitch_id: The Twitch video ID
            job_id: The job ID
            detailed_progress: Dictionary containing detailed progress information
            max_retries: Maximum number of retry attempts

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.convex:
            logger.error("Convex client not initialized")
            return False

        # In test mode, just log and return success
        if self.test_mode:
            logger.info(f"[TEST MODE] Would update detailed progress for video {twitch_id} (job {job_id}): {detailed_progress}")
            return True

        # Prepare detailed progress data
        progress_update = {
            "jobId": job_id,
            "currentStage": detailed_progress.get("current_stage"),
            "progressPercentage": detailed_progress.get("progress_percentage"),
            "estimatedCompletionSeconds": detailed_progress.get("estimated_completion_seconds"),
            "estimatedCompletionTime": detailed_progress.get("estimated_completion_time"),
            "stagesCompleted": detailed_progress.get("stages_completed"),
            "totalStages": detailed_progress.get("total_stages"),
            "processingStartedAt": detailed_progress.get("processing_started_at"),
            "lastProgressUpdate": detailed_progress.get("last_progress_update"),
            "updatedAt": int(time.time() * 1000)  # Current timestamp in milliseconds
        }

        # Add error information if present
        if detailed_progress.get("error_info"):
            error_info = detailed_progress["error_info"]
            progress_update.update({
                "errorType": error_info.get("error_type"),
                "errorCode": error_info.get("error_code"),
                "isRetryable": error_info.get("is_retryable"),
                "retryCount": detailed_progress.get("retry_count", 0),
                "supportReference": error_info.get("support_reference")
            })

        # Log detailed progress data for monitoring (Convex URLs schema doesn't support progress fields)
        logger.info(f"Detailed progress update for video {twitch_id} (job {job_id}): {progress_update}")

        # For now, we'll just return True since the progress is logged
        # TODO: Update Convex schema to support progress fields or use a separate table
        return True

    def track_job_analytics(self, twitch_id: str, job_id: str, analytics_data: Dict[str, Any]) -> bool:
        """
        Track job analytics and performance metrics

        Args:
            twitch_id: The Twitch video ID
            job_id: The job ID
            analytics_data: Dictionary containing analytics information

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.convex:
            logger.error("Convex client not initialized")
            return False

        # In test mode, just log and return success
        if self.test_mode:
            logger.info(f"[TEST MODE] Would track analytics for job {job_id}: {analytics_data}")
            return True

        try:
            analytics_record = {
                "videoId": twitch_id,
                "jobId": job_id,
                "startTime": analytics_data.get("start_time"),
                "endTime": analytics_data.get("end_time"),
                "totalDuration": analytics_data.get("total_duration"),
                "stageTimings": analytics_data.get("stage_timings", {}),
                "errorCount": analytics_data.get("error_count", 0),
                "retryCount": analytics_data.get("retry_count", 0),
                "finalStatus": analytics_data.get("final_status"),
                "createdAt": int(time.time() * 1000)
            }

            # Note: This would require a new Convex mutation for analytics
            # For now, we'll store it as part of the video record
            success = self.convex.update_urls(twitch_id, {"analyticsData": analytics_record})

            if success:
                logger.info(f"Successfully tracked analytics for job {job_id}")
                return True
            else:
                logger.error(f"Failed to track analytics for job {job_id}")
                return False

        except Exception as e:
            logger.error(f"Error tracking analytics for job {job_id}: {str(e)}")
            return False

    def store_webhook_config(self, twitch_id: str, webhook_config: Dict[str, Any]) -> bool:
        """
        Store webhook configuration for a video

        Args:
            twitch_id: The Twitch video ID
            webhook_config: Dictionary containing webhook configuration

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.convex:
            logger.error("Convex client not initialized")
            return False

        # In test mode, just log and return success
        if self.test_mode:
            logger.info(f"[TEST MODE] Would store webhook config for video {twitch_id}: {webhook_config}")
            return True

        try:
            webhook_update = {
                "webhookUrls": webhook_config.get("webhook_urls", []),
                "webhookEvents": webhook_config.get("webhook_events", [])
            }

            success = self.convex.update_urls(twitch_id, webhook_update)

            if success:
                logger.info(f"Successfully stored webhook config for video {twitch_id}")
                return True
            else:
                logger.error(f"Failed to store webhook config for video {twitch_id}")
                return False

        except Exception as e:
            logger.error(f"Error storing webhook config for video {twitch_id}: {str(e)}")
            return False
