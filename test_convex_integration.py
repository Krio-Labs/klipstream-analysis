#!/usr/bin/env python3
"""
Test script for the Convex integration
"""

import os
import json
import base64
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_convex_upload():
    """Test uploading a file to Convex"""
    # Get the Convex upload URL from environment
    convex_url = os.getenv("CONVEX_UPLOAD_URL")
    if not convex_url:
        convex_url = input("Enter Convex upload URL: ")

    # Get the Twitch video ID to update
    video_id = input("Enter Twitch video ID to update (must already exist in Convex): ")

    # Create a test file
    test_file_path = "test_file.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test file for Convex upload")

    # Read the file
    with open(test_file_path, "rb") as f:
        file_content = f.read()

    # Encode the file content
    encoded_content = base64.b64encode(file_content).decode("utf-8")

    # Prepare the payload
    payload = {
        "fileName": "test_file.txt",
        "fileContent": encoded_content,
        "videoId": video_id,
        "fileType": "other"
    }

    # Send the request
    print(f"Sending request to {convex_url}")
    try:
        response = requests.post(
            convex_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # Print the response
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

    # Clean up
    os.remove(test_file_path)

def test_update_video_status():
    """Test updating video status in Convex"""
    # Get the Convex upload URL from environment
    convex_url = os.getenv("CONVEX_UPLOAD_URL")
    if not convex_url:
        convex_url = input("Enter Convex upload URL: ")

    # Get the video ID
    video_id = input("Enter Twitch video ID to update: ")

    # Prepare the update URL
    update_url = convex_url.replace("uploadFile", "updateVideoStatus")

    # Prepare the payload
    payload = {
        "videoId": video_id,
        "status": "completed",
        "twitchInfo": json.dumps({
            "id": video_id,
            "title": "Test Video",
            "user_name": "TestUser",
            "published_at": "2023-01-01T00:00:00Z",
            "duration": 3600,
            "view_count": 1000,
            "thumbnail_url": f"https://static-cdn.jtvnw.net/cf_vods/d2nvs31859zcd8/twitchcdn/{video_id}/thumb/thumb0.jpg"
        }),
        "duration": "3600",
        "title": "Test Video",
        "user_name": "TestUser",
        "view_count": 1000,
        "published_at": "2023-01-01T00:00:00Z",
        "language": "en",
        "thumbnail": f"https://static-cdn.jtvnw.net/cf_vods/d2nvs31859zcd8/twitchcdn/{video_id}/thumb/thumb0.jpg"
    }

    # Send the request
    print(f"Sending request to {update_url}")
    try:
        response = requests.post(
            update_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # Print the response
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Convex Integration Test")
    print("======================")
    print("1. Test file upload")
    print("2. Test update video status")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        test_convex_upload()
    elif choice == "2":
        test_update_video_status()
    else:
        print("Invalid choice. Exiting.")
