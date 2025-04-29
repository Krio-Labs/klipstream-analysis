#!/usr/bin/env python3
"""
Test script for Google Cloud Storage integration
"""

import os
import argparse
import logging
from gcs_upload import upload_files, update_video_status

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_gcs_upload(video_id, specific_files=False):
    """
    Test uploading files to Google Cloud Storage

    Args:
        video_id: The Twitch video ID
        specific_files: If True, test uploading specific files
    """
    # Create test directories if they don't exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Create a test file in outputs directory
    test_file_path = f"outputs/audio_{video_id}_test.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test file for GCS upload")

    # Create a test video file in outputs directory
    test_video_path = f"outputs/{video_id}.mp4"
    with open(test_video_path, "w") as f:
        f.write("This is a test video file for GCS upload")

    # Create a test audio file in outputs directory
    test_audio_path = f"outputs/audio_{video_id}.wav"
    with open(test_audio_path, "w") as f:
        f.write("This is a test audio file for GCS upload")

    # Create a test waveform file in outputs directory
    test_waveform_path = f"outputs/audio_{video_id}_waveform.json"
    with open(test_waveform_path, "w") as f:
        f.write('{"waveform": [0.1, 0.2, 0.3, 0.4, 0.5]}')

    # Create a test chat file in data directory
    test_chat_path = f"data/{video_id}_chat.csv"
    with open(test_chat_path, "w") as f:
        f.write("timestamp,username,message\n")
        f.write("0,testuser,Hello world!")

    # Upload the files
    print(f"Uploading test files for video ID: {video_id}")

    if specific_files:
        # Test uploading specific files
        print("Testing upload of specific files...")
        specific_file_paths = [test_video_path, test_audio_path, test_waveform_path]
        uploaded_files = upload_files(video_id, specific_file_paths)
    else:
        # Test regular upload
        uploaded_files = upload_files(video_id)

    # Print results
    print(f"Uploaded {len(uploaded_files)} files:")
    for file_info in uploaded_files:
        print(f"- {file_info['filename']} -> {file_info['url']}")

    # Clean up
    os.remove(test_file_path)
    os.remove(test_video_path)
    os.remove(test_audio_path)
    os.remove(test_waveform_path)
    os.remove(test_chat_path)

    return uploaded_files

def test_update_status(video_id):
    """Test updating video status"""
    # Create test twitch info
    twitch_info = {
        "id": video_id,
        "title": "Test Video",
        "user_name": "TestUser",
        "duration": 3600,
        "view_count": 1000,
        "published_at": "2023-01-01T00:00:00Z",
        "language": "en",
        "thumbnail_url": f"https://static-cdn.jtvnw.net/cf_vods/d2nvs31859zcd8/twitchcdn/{video_id}/thumb/thumb0.jpg"
    }

    # Update status
    print(f"Updating status for video ID: {video_id}")
    success = update_video_status(video_id, "completed", twitch_info)

    if success:
        print("Status updated successfully")
    else:
        print("Failed to update status")

    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Google Cloud Storage integration")
    parser.add_argument("--video-id", required=True, help="Twitch video ID to use for testing")
    parser.add_argument("--test-upload", action="store_true", help="Test file upload")
    parser.add_argument("--test-specific", action="store_true", help="Test uploading specific files")
    parser.add_argument("--test-status", action="store_true", help="Test status update")

    args = parser.parse_args()

    if args.test_upload:
        test_gcs_upload(args.video_id, specific_files=False)

    if args.test_specific:
        test_gcs_upload(args.video_id, specific_files=True)

    if args.test_status:
        test_update_status(args.video_id)

    if not (args.test_upload or args.test_specific or args.test_status):
        print("No tests specified. Use --test-upload, --test-specific, or --test-status")
