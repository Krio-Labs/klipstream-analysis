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
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from utils.logging_setup import setup_logger
from convex_integration import ConvexIntegration, STATUS_QUEUED, STATUS_DOWNLOADING, STATUS_FETCHING_CHAT, STATUS_TRANSCRIBING, STATUS_ANALYZING, STATUS_FINDING_HIGHLIGHTS, STATUS_COMPLETED, STATUS_FAILED

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

    def update_video_status(self, twitch_id: str, status: str, max_retries: int = 3) -> bool:
        """
        Update the status field for a video in the Convex database

        Args:
            twitch_id (str): The Twitch video ID
            status (str): The new status value
            max_retries (int): Maximum number of retry attempts

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.convex:
            logger.error("Convex client not initialized")
            return False

        # Validate status
        if status not in VALID_STATUSES:
            logger.warning(f"Invalid status '{status}'. Valid statuses are: {VALID_STATUSES}")
            logger.warning(f"Proceeding with update anyway, but consider using a valid status")

        # In test mode, just log the status update and return success
        if self.test_mode:
            logger.info(f"[TEST MODE] Would update status for video {twitch_id} to '{status}'")
            return True

        # Try to update with retries
        for attempt in range(max_retries):
            try:
                # Call the method to update the status
                logger.info(f"Updating status for video {twitch_id} to '{status}'...")
                success = self.convex.update_status_by_twitch_id(twitch_id, status)

                if success:
                    logger.info(f"Successfully updated status for video {twitch_id} to '{status}'")
                    return True
                else:
                    logger.error(f"Failed to update status for video {twitch_id} to '{status}'")
                    if attempt < max_retries - 1:
                        retry_delay = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {max_retries} attempts to update status failed")
                        return False
            except Exception as e:
                logger.error(f"Attempt {attempt+1}/{max_retries} failed to update status for video {twitch_id}: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts to update status failed")
                    return False

        return False

    def update_video_urls(self, twitch_id: str, url_updates: Dict[str, str], max_retries: int = 3) -> bool:
        """
        Update URL fields for a video in the Convex database

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

        # Log the URLs we're updating
        logger.info(f"Updating URLs for video {twitch_id}: {list(url_updates.keys())}")
        for field, url in url_updates.items():
            logger.info(f"  {field}: {url}")

        # In test mode, just log the URL updates and return success
        if self.test_mode:
            logger.info(f"[TEST MODE] Would update URLs for video {twitch_id}: {list(url_updates.keys())}")
            return True

        # Try to update with retries
        for attempt in range(max_retries):
            try:
                # Call the method to update the URLs
                logger.info(f"Updating URLs for video {twitch_id}...")
                success = self.convex.update_urls(twitch_id, url_updates)

                if success:
                    logger.info(f"Successfully updated URLs for video {twitch_id}: {list(url_updates.keys())}")
                    return True
                else:
                    logger.error(f"Failed to update URLs for video {twitch_id}")
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
