#!/usr/bin/env python3
"""
Convex Integration Module

This module provides functions for integrating with the Convex database.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv
from convex_api import ConvexAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Video processing status constants
STATUS_QUEUED = "Queued"
STATUS_DOWNLOADING = "Downloading"
STATUS_FETCHING_CHAT = "Fetching chat"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_ANALYZING = "Analyzing"
STATUS_FINDING_HIGHLIGHTS = "Finding highlights"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

class ConvexIntegration:
    """Integration with the Convex database."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Convex integration.

        Args:
            url: The Convex deployment URL. If not provided, will try to get from CONVEX_URL env var.
            api_key: The Convex API key. If not provided, will try to get from CONVEX_API_KEY env var.
        """
        # Create a Convex API client
        self.client = ConvexAPIClient(url, api_key)

    def get_video(self, twitch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a video by its Twitch ID.

        Args:
            twitch_id: The Twitch ID of the video.

        Returns:
            The video data if found, None otherwise.
        """
        return self.client.get_video_by_twitch_id(twitch_id)

    def update_status(self, video_id: str, status: str) -> bool:
        """
        Update the status of a video using its Convex ID.

        Args:
            video_id: The Convex ID of the video.
            status: The new status.

        Returns:
            True if the update was successful, False otherwise.
        """
        logger.info(f"Updating status for video {video_id} to '{status}'")
        return self.client.update_video_status(video_id, status)

    def update_status_by_twitch_id(self, twitch_id: str, status: str) -> bool:
        """
        Update the status of a video using its Twitch ID.

        Args:
            twitch_id: The Twitch ID of the video.
            status: The new status.

        Returns:
            True if the update was successful, False otherwise.
        """
        logger.info(f"Updating status for video with Twitch ID {twitch_id} to '{status}'")
        return self.client.update_status_by_twitch_id(twitch_id, status)

    def update_urls(self, twitch_id: str, urls: Dict[str, str]) -> bool:
        """
        Update the URLs of a video.

        Args:
            twitch_id: The Twitch ID of the video.
            urls: A dictionary of URL fields to update, e.g. {"transcriptUrl": "https://..."}

        Returns:
            True if the update was successful, False otherwise.
        """
        logger.info(f"Updating URLs for video with Twitch ID {twitch_id}")
        return self.client.update_video_urls(twitch_id, urls)

    def update_transcript_url(self, twitch_id: str, url: str) -> bool:
        """
        Update the transcript URL of a video.

        Args:
            twitch_id: The Twitch ID of the video.
            url: The URL of the transcript.

        Returns:
            True if the update was successful, False otherwise.
        """
        return self.update_urls(twitch_id, {"transcriptUrl": url})

    def update_chat_url(self, twitch_id: str, url: str) -> bool:
        """
        Update the chat URL of a video.

        Args:
            twitch_id: The Twitch ID of the video.
            url: The URL of the chat log.

        Returns:
            True if the update was successful, False otherwise.
        """
        return self.update_urls(twitch_id, {"chatUrl": url})

    def update_audiowave_url(self, twitch_id: str, url: str) -> bool:
        """
        Update the audiowave URL of a video.

        Args:
            twitch_id: The Twitch ID of the video.
            url: The URL of the audiowave visualization.

        Returns:
            True if the update was successful, False otherwise.
        """
        return self.update_urls(twitch_id, {"audiowaveUrl": url})

    def update_transcript_analysis_url(self, twitch_id: str, url: str) -> bool:
        """
        Update the transcript analysis URL of a video.

        Args:
            twitch_id: The Twitch ID of the video.
            url: The URL of the transcript analysis.

        Returns:
            True if the update was successful, False otherwise.
        """
        return self.update_urls(twitch_id, {"transcriptAnalysisUrl": url})

    def update_transcript_word_url(self, twitch_id: str, url: str) -> bool:
        """
        Update the transcript word URL of a video.

        Args:
            twitch_id: The Twitch ID of the video.
            url: The URL of the transcript word data.

        Returns:
            True if the update was successful, False otherwise.
        """
        return self.update_urls(twitch_id, {"transcriptWordUrl": url})

    def update_all_urls(self, twitch_id: str, transcript_url: Optional[str] = None,
                       chat_url: Optional[str] = None, audiowave_url: Optional[str] = None,
                       transcript_analysis_url: Optional[str] = None,
                       transcript_word_url: Optional[str] = None) -> bool:
        """
        Update all URLs of a video.

        Args:
            twitch_id: The Twitch ID of the video.
            transcript_url: The URL of the transcript.
            chat_url: The URL of the chat log.
            audiowave_url: The URL of the audiowave visualization.
            transcript_analysis_url: The URL of the transcript analysis.
            transcript_word_url: The URL of the transcript word data.

        Returns:
            True if the update was successful, False otherwise.
        """
        urls = {}

        if transcript_url:
            urls["transcriptUrl"] = transcript_url

        if chat_url:
            urls["chatUrl"] = chat_url

        if audiowave_url:
            urls["audiowaveUrl"] = audiowave_url

        if transcript_analysis_url:
            urls["transcriptAnalysisUrl"] = transcript_analysis_url

        if transcript_word_url:
            urls["transcriptWordUrl"] = transcript_word_url

        if not urls:
            logger.warning("No URLs provided for update")
            return False

        return self.update_urls(twitch_id, urls)

    def create_video(self, video_data: Dict[str, Any]) -> bool:
        """
        Create a new video entry in the database.

        Args:
            video_data: Dictionary containing video data

        Returns:
            True if successful, False otherwise.
        """
        return self.client.create_video(video_data)

# Example usage
if __name__ == "__main__":
    import sys

    # Create a Convex integration
    convex = ConvexIntegration()

    # Get a video by Twitch ID
    twitch_id = sys.argv[1] if len(sys.argv) > 1 else "2434635255"
    print(f"Getting video with Twitch ID: {twitch_id}")

    video = convex.get_video(twitch_id)

    if video:
        print(f"Found video: {video['_id']} - {video['title']}")

        # Update the video status using Convex ID
        video_id = video.get("_id")
        if video_id:
            print(f"Updating status for video {video_id}")
            success = convex.update_status(video_id, "Testing Convex Integration (ID)")
            print(f"Status update using ID {'successful' if success else 'failed'}")

        # Update the video status using Twitch ID
        print(f"Updating status for video with Twitch ID {twitch_id}")
        success = convex.update_status_by_twitch_id(twitch_id, "Testing Convex Integration (Twitch ID)")
        print(f"Status update using Twitch ID {'successful' if success else 'failed'}")

        # Update the video URLs
        print(f"Updating URLs for video with Twitch ID {twitch_id}")
        success = convex.update_all_urls(
            twitch_id,
            transcript_url="https://storage.googleapis.com/klipstream-transcripts/2434635255/transcript.json",
            chat_url="https://storage.googleapis.com/klipstream-chatlogs/2434635255/chat.json",
            audiowave_url="https://storage.googleapis.com/klipstream-vods-raw/2434635255/audiowave.png",
            transcript_analysis_url="https://storage.googleapis.com/klipstream-analysis/2434635255/analysis.json",
            transcript_word_url="https://storage.googleapis.com/klipstream-transcripts/2434635255/words.json"
        )
        print(f"URL update {'successful' if success else 'failed'}")
    else:
        print(f"No video found with Twitch ID: {twitch_id}")
