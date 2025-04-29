#!/usr/bin/env python3
"""
Test script to verify GCS authentication and upload functionality
"""

import os
import sys
from google.cloud import storage
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_file(filename, content="This is a test file for GCS upload"):
    """Create a test file with the given content"""
    with open(filename, "w") as f:
        f.write(content)
    return filename

def upload_to_bucket(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        logger.info(f"Uploading {source_file_name} to {bucket_name}/{destination_blob_name}")
        
        # Initialize GCS client
        storage_client = storage.Client()
        
        # Get bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Create blob
        blob = bucket.blob(destination_blob_name)
        
        # Upload file
        blob.upload_from_filename(source_file_name)
        
        # Get public URL
        url = f"gs://{bucket_name}/{destination_blob_name}"
        
        logger.info(f"File {source_file_name} uploaded to {url}")
        return url
    
    except Exception as e:
        logger.error(f"Error uploading file {source_file_name} to GCS: {str(e)}")
        return None

def test_all_buckets():
    """Test uploading to all required buckets"""
    # Create test directory if it doesn't exist
    test_dir = "test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    # Define buckets to test
    buckets = [
        "klipstream-vods-raw",
        "klipstream-transcripts",
        "klipstream-chatlogs"
    ]
    
    # Test video ID
    video_id = "test_video_123"
    
    # Create and upload test files for each bucket
    results = []
    
    # Test VOD bucket (video, audio, waveform)
    test_video = os.path.join(test_dir, f"{video_id}.mp4")
    create_test_file(test_video, "Test video content")
    video_url = upload_to_bucket("klipstream-vods-raw", test_video, f"{video_id}/{video_id}.mp4")
    results.append({"file": test_video, "url": video_url, "success": video_url is not None})
    
    test_audio = os.path.join(test_dir, f"audio_{video_id}.wav")
    create_test_file(test_audio, "Test audio content")
    audio_url = upload_to_bucket("klipstream-vods-raw", test_audio, f"{video_id}/audio_{video_id}.wav")
    results.append({"file": test_audio, "url": audio_url, "success": audio_url is not None})
    
    test_waveform = os.path.join(test_dir, f"audio_{video_id}_waveform.json")
    create_test_file(test_waveform, '{"waveform": [0.1, 0.2, 0.3, 0.4, 0.5]}')
    waveform_url = upload_to_bucket("klipstream-vods-raw", test_waveform, f"{video_id}/audio_{video_id}_waveform.json")
    results.append({"file": test_waveform, "url": waveform_url, "success": waveform_url is not None})
    
    # Test transcripts bucket
    test_paragraphs = os.path.join(test_dir, f"audio_{video_id}_paragraphs.csv")
    create_test_file(test_paragraphs, "start,end,text\n0,5,Test paragraph")
    paragraphs_url = upload_to_bucket("klipstream-transcripts", test_paragraphs, f"{video_id}/audio_{video_id}_paragraphs.csv")
    results.append({"file": test_paragraphs, "url": paragraphs_url, "success": paragraphs_url is not None})
    
    # Test chatlogs bucket
    test_chat = os.path.join(test_dir, f"{video_id}_chat.csv")
    create_test_file(test_chat, "timestamp,username,message\n0,user1,Hello")
    chat_url = upload_to_bucket("klipstream-chatlogs", test_chat, f"{video_id}/{video_id}_chat.csv")
    results.append({"file": test_chat, "url": chat_url, "success": chat_url is not None})
    
    # Print results
    logger.info("Upload test results:")
    all_success = True
    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        logger.info(f"{status}: {result['file']} -> {result['url']}")
        if not result["success"]:
            all_success = False
    
    # Clean up test files
    for result in results:
        try:
            os.remove(result["file"])
            logger.info(f"Removed test file: {result['file']}")
        except Exception as e:
            logger.warning(f"Error removing test file {result['file']}: {str(e)}")
    
    # Try to remove test directory
    try:
        os.rmdir(test_dir)
        logger.info(f"Removed test directory: {test_dir}")
    except Exception as e:
        logger.warning(f"Error removing test directory {test_dir}: {str(e)}")
    
    return all_success

if __name__ == "__main__":
    logger.info("Starting GCS authentication and upload test")
    success = test_all_buckets()
    if success:
        logger.info("All tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed. Check the logs for details.")
        sys.exit(1)
