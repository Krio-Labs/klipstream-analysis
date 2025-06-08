#!/usr/bin/env python3
"""
Convex API Client

This module provides a client for interacting with the Convex API.
"""

import os
import requests
import json
import logging
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConvexAPIClient:
    """Client for interacting with the Convex API."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Convex API client.

        Args:
            url: The Convex deployment URL. If not provided, will try to get from CONVEX_URL env var.
            api_key: The Convex API key. If not provided, will try to get from CONVEX_API_KEY env var.
        """
        # Load environment variables if not already loaded
        load_dotenv()

        # Get URL and API key from parameters or environment variables
        self.url = url or os.environ.get("CONVEX_URL")
        self.api_key = api_key or os.environ.get("CONVEX_API_KEY")

        # Validate URL and API key
        if not self.url:
            raise ValueError("Convex URL not provided and CONVEX_URL environment variable not set")

        if not self.api_key:
            logger.warning("Convex API key not provided and CONVEX_API_KEY environment variable not set")

        # Set default timeout for requests
        self.timeout = 10  # seconds

        # Set base headers
        self.headers = {
            "Content-Type": "application/json"
        }

        # Note: We're not using the API key in the headers as it seems the Convex API
        # doesn't require authentication for the functions we're calling

    def query(self, function_path: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a Convex query function.

        Args:
            function_path: The path to the function in the format "module:function".
            args: The arguments to pass to the function.

        Returns:
            The response from the Convex API.

        Raises:
            requests.exceptions.RequestException: If the request fails.
            ValueError: If the response contains an error.
        """
        url = f"{self.url}/api/query"
        payload = {
            "path": function_path,
            "args": args or {}
        }

        logger.debug(f"Executing query {function_path} with args: {args}")

        response = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()

        # Check for errors in the response
        if isinstance(result, dict) and result.get("status") == "error":
            raise ValueError(f"Convex API error: {result.get('errorMessage')}")

        return result

    def mutation(self, function_path: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a Convex mutation function.

        Args:
            function_path: The path to the function in the format "module:function".
            args: The arguments to pass to the function.

        Returns:
            The response from the Convex API.

        Raises:
            requests.exceptions.RequestException: If the request fails.
            ValueError: If the response contains an error.
        """
        url = f"{self.url}/api/mutation"
        payload = {
            "path": function_path,
            "args": args or {}
        }

        logger.debug(f"Executing mutation {function_path} with args: {args}")

        response = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()

        # Check for errors in the response
        if isinstance(result, dict) and result.get("status") == "error":
            raise ValueError(f"Convex API error: {result.get('errorMessage')}")

        return result

    def get_video_by_twitch_id(self, twitch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a video by its Twitch ID.

        Args:
            twitch_id: The Twitch ID of the video.

        Returns:
            The video data if found, None otherwise.
        """
        try:
            result = self.query("video:getByTwitchIdPublic", {"twitchId": twitch_id})

            if result.get("status") == "success" and "value" in result:
                return result["value"]

            return None
        except Exception as e:
            logger.error(f"Error getting video by Twitch ID {twitch_id}: {str(e)}")
            return None

    def update_video_status(self, video_id: str, status: str) -> bool:
        """
        Update the status of a video using its Convex ID.

        Args:
            video_id: The Convex ID of the video.
            status: The new status.

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            result = self.mutation("video:updateStatus", {"id": video_id, "status": status})
            return result.get("status") == "success"
        except Exception as e:
            logger.error(f"Error updating video status for {video_id}: {str(e)}")
            return False

    def update_status_by_twitch_id(self, twitch_id: str, status: str) -> bool:
        """
        Update the status of a video using its Twitch ID.

        Args:
            twitch_id: The Twitch ID of the video.
            status: The new status.

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            result = self.mutation("video:updateStatusByTwitchId", {"twitchId": twitch_id, "status": status})
            return result.get("status") == "success"
        except Exception as e:
            logger.error(f"Error updating video status for Twitch ID {twitch_id}: {str(e)}")
            return False

    def update_video_urls(self, twitch_id: str, urls: Dict[str, str]) -> bool:
        """
        Update the URLs of a video.

        Args:
            twitch_id: The Twitch ID of the video.
            urls: A dictionary of URL fields to update, e.g. {"transcriptUrl": "https://..."}

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            # Format the arguments according to the expected schema
            args = {
                "twitchId": twitch_id,
                "urls": urls
            }
            result = self.mutation("video:updateUrls", args)
            return result.get("status") == "success"
        except Exception as e:
            logger.error(f"Error updating video URLs for Twitch ID {twitch_id}: {str(e)}")
            return False

    def create_video(self, video_data: Dict[str, Any]) -> bool:
        """
        Create a new video entry in the database.

        Note: This is a placeholder for future implementation.
        Creating videos requires proper team and thumbnail setup.

        Args:
            video_data: Dictionary containing video data

        Returns:
            True if the creation was successful, False otherwise.
        """
        logger.warning("Video creation not implemented yet - requires proper team and thumbnail setup")
        logger.info(f"Would create video with data: {video_data}")
        return False

    def create_video_minimal(self, twitch_id: str, status: str = "Queued") -> bool:
        """
        Create a new video entry with minimal information (used by pipeline).

        Args:
            twitch_id: The Twitch video ID
            status: Initial status for the video

        Returns:
            True if the creation was successful, False otherwise.
        """
        try:
            # Use the existing insert mutation with minimal required fields
            # Using the provided valid team ID
            result = self.mutation("video:insert", {
                "team": "js7bj9zgdkyj9ykvr4m6jarxkh7ep9fa",  # Valid team ID
                "twitch_id": twitch_id,
                "title": f"Twitch VOD {twitch_id}",
                "thumbnail_id": "placeholder",
                "duration": "unknown",
                "status": status
            })

            if result:
                logger.info(f"Successfully created video entry for Twitch ID {twitch_id}: {result}")
                return True
            else:
                logger.error(f"Failed to create video entry for Twitch ID {twitch_id}: {result}")
                return False

        except Exception as e:
            logger.error(f"Error creating video entry for Twitch ID {twitch_id}: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    import sys

    # Create a client
    client = ConvexAPIClient()

    # Get a video by Twitch ID
    twitch_id = sys.argv[1] if len(sys.argv) > 1 else "2434635255"
    print(f"Getting video with Twitch ID: {twitch_id}")

    video = client.get_video_by_twitch_id(twitch_id)

    if video:
        print(f"Found video: {json.dumps(video, indent=2)}")

        # Update the video status
        video_id = video.get("_id")
        if video_id:
            print(f"Updating status for video {video_id}")
            success = client.update_video_status(video_id, "Testing Convex API Client")
            print(f"Status update {'successful' if success else 'failed'}")

            # Update the video URLs
            twitch_id = video.get("twitch_id")
            print(f"Updating URLs for video with Twitch ID {twitch_id}")
            urls = {
                "transcriptUrl": "https://example.com/transcript.json",
                "chatUrl": "https://example.com/chat.json",
                "audiowaveUrl": "https://example.com/audiowave.png",
                "transcriptAnalysisUrl": "https://example.com/analysis.json",
                "transcriptWordUrl": "https://example.com/words.json"
            }
            success = client.update_video_urls(twitch_id, urls)
            print(f"URL update {'successful' if success else 'failed'}")
    else:
        print(f"No video found with Twitch ID: {twitch_id}")
