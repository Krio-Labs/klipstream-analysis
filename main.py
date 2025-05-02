import asyncio
from pathlib import Path
import warnings
import multiprocessing
import atexit
import os
import functions_framework
from dotenv import load_dotenv

# Import utilities
from utils.logging_setup import setup_logger
from utils.helpers import extract_video_id

# Import raw file processor from reorganized structure
from raw_pipeline import process_raw_files

# Import other modules for analysis
from audio_sentiment import sentiment_analysis
from audio_analysis import analyze_transcription_highlights
from audio_analysis import plot_metrics
from chat_processor import process_chat_data
from chat_sentiment import analyze_chat_sentiment
from chat_analysis import analyze_chat_intervals, analyze_chat_highlights

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("main", "main.log")

def cleanup():
    try:
        multiprocessing.resource_tracker._resource_tracker = None
    except Exception:
        pass

atexit.register(cleanup)

import asyncio
from convex_upload import upload_files

async def upload_to_convex_async(video_id):
    """Use the existing convex_upload.py to upload files to Convex"""
    try:
        logger.info(f"Uploading files for video {video_id} using convex_upload.py")
        await upload_files(video_id)
        return True
    except Exception as e:
        logger.error(f"Error uploading files to Convex: {str(e)}")
        return False

def upload_to_convex(video_id):
    """Synchronous wrapper for upload_to_convex_async"""
    return asyncio.run(upload_to_convex_async(video_id))

async def update_video_status_async(video_id, status, twitch_info=None):
    """Updates the status of a video in Convex using convex_upload.py"""
    try:
        # Import the update_video function from convex_upload
        from convex_upload import update_video

        # Prepare update data - use the format expected by the Convex endpoint
        update_data = {"id": video_id, "status": status}

        # Add twitch info if available
        if twitch_info:
            # Store the complete twitch_info object
            update_data["twitch_info"] = twitch_info

            # Also add individual fields for backward compatibility
            if "title" in twitch_info:
                update_data["title"] = twitch_info["title"]
            if "user_name" in twitch_info:
                update_data["user_name"] = twitch_info["user_name"]
            if "duration" in twitch_info:
                update_data["duration"] = str(twitch_info["duration"])
            if "view_count" in twitch_info:
                update_data["view_count"] = twitch_info["view_count"]
            if "published_at" in twitch_info:
                update_data["published_at"] = twitch_info["published_at"]
            if "language" in twitch_info:
                update_data["language"] = twitch_info["language"]
            if "thumbnail_url" in twitch_info:
                update_data["thumbnail"] = twitch_info["thumbnail_url"]

        # Update the video
        logger.info(f"Updating video {video_id} status to {status}")
        success = await update_video(update_data)

        if success:
            logger.info(f"Successfully updated video {video_id} status to {status}")
            return True
        else:
            logger.error(f"Failed to update video {video_id} status")
            return False

    except Exception as e:
        logger.error(f"Error updating video status: {str(e)}")
        logger.exception("Full exception details:")
        return False

def update_video_status(video_id, status, twitch_info=None):
    """Synchronous wrapper for update_video_status_async"""
    return asyncio.run(update_video_status_async(video_id, status, twitch_info))

@functions_framework.http
def run_pipeline(request):
    """
    Cloud Function entry point
    """
    try:
        # Parse the request JSON
        request_json = request.get_json()
        if not request_json or 'url' not in request_json:
            return 'No url provided', 400

        url = request_json['url']
        # Note: We're using convex_upload.py which handles the URL internally now

        # Note: We're using convex_upload.py which doesn't need team_id

        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")  # Use project-relative path
        output_dir.mkdir(exist_ok=True, parents=True)

        # Create data directory for chat files
        data_dir = Path("data")  # Use project-relative path
        data_dir.mkdir(exist_ok=True, parents=True)

        # Create downloads directory
        downloads_dir = Path("downloads")
        downloads_dir.mkdir(exist_ok=True, parents=True)

        # Extract video ID
        video_id = extract_video_id(url)

        # Update video status to "processing"
        update_video_status(video_id, "processing")

        try:
            # Run the async pipeline using asyncio
            result = asyncio.run(process_video(url))

            # Get Twitch metadata
            twitch_info = result.get("twitch_info", {})

            # Upload results to Convex using convex_upload.py
            # This will handle both outputs and data directories
            upload_success = upload_to_convex(video_id)

            # Update video status to "completed"
            update_success = update_video_status(video_id, "completed", twitch_info)

            return {
                "status": "success",
                "result": result,
                "video_id": video_id,
                "upload_success": upload_success,
                "update_success": update_success
            }

        except Exception as e:
            # Update video status to "failed"
            update_video_status(video_id, "failed")
            raise e

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}, 500

async def process_video(url: str):
    """
    Process a Twitch VOD URL

    This function first processes the raw files using the raw file processor,
    then performs analysis on those files.

    Args:
        url (str): The Twitch VOD URL

    Returns:
        dict: Dictionary with results and metadata
    """
    try:
        # First, process the raw files using our reorganized module
        logger.info(f"Step 1: Processing raw files for URL: {url}")
        raw_result = await process_raw_files(url)

        # Extract video_id and file paths from the raw processing result
        video_id = raw_result["video_id"]
        logger.info(f"Raw processing completed for video ID: {video_id}")

        # Get file paths from the raw processing result
        files = raw_result["files"]

        # Get Twitch video metadata
        twitch_info = raw_result.get("twitch_info", {})

        # Step 2: Sentiment analysis
        logger.info("Step 2: Performing sentiment analysis...")
        sentiment_result = sentiment_analysis(video_id=video_id)
        if not sentiment_result:
            logger.error("Sentiment analysis failed to complete successfully")
            raise RuntimeError("Sentiment analysis failed")
        logger.info("Sentiment analysis completed successfully")

        # Step 3: Audio analysis
        logger.info("Step 3: Performing audio analysis...")

        # Get paragraphs file path
        paragraphs_file = files["paragraphs_file"]

        # Verify the file exists
        if not Path(paragraphs_file).exists():
            raise FileNotFoundError(f"Required file not found: {paragraphs_file}")

        # Get output directory
        output_dir = Path("outputs")

        # Analyze transcript highlights
        grouped_data = analyze_transcription_highlights(str(paragraphs_file), str(output_dir))
        if grouped_data is not None:
            logger.info("Transcript highlights analysis completed")

            # Plot metrics with video_id
            plot_metrics(str(output_dir), video_id)
            logger.info("Metrics plotting completed")
        else:
            raise RuntimeError("Transcript highlights analysis failed")

        logger.info("Audio analysis completed")

        # Step 4: Process chat data
        logger.info("Step 4: Processing chat data...")
        try:
            # Process chat data
            process_chat_data(video_id)
            logger.info("Chat data processed successfully")

            # Add chat sentiment analysis
            logger.info("Step 5: Analyzing chat sentiment...")
            sentiment_output = analyze_chat_sentiment(video_id)
            if sentiment_output:
                logger.info(f"Chat sentiment analysis completed. Results saved to: {sentiment_output}")

                # Only run chat analysis if sentiment analysis succeeded
                logger.info("Step 6: Running chat analysis...")
                interval_stats = analyze_chat_intervals(video_id, str(output_dir))
                if interval_stats is not None:
                    logger.info("Chat intervals analysis completed")

                    # Run highlight analysis
                    analyze_chat_highlights(video_id, str(output_dir))
                    logger.info("Chat highlights analysis completed")

        except Exception as e:
            logger.error(f"Failed to process chat data: {e}")
            # Continue pipeline even if chat operations fail
            logger.warning("Continuing pipeline without chat data")

        logger.info("Pipeline completed successfully!")

        # Return a dictionary with results and metadata
        return {
            "status": "completed",
            "video_id": video_id,
            "twitch_info": twitch_info,
            "files": files,
            "uploaded_files": raw_result.get("uploaded_files", [])
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

@functions_framework.http
def list_files(request):
    """List all files in the current directory

    This is a utility function for debugging purposes.
    """
    # Get query parameters
    request_args = request.args if request.args else {}
    max_depth = int(request_args.get('max_depth', '3'))

    # Get the current working directory
    current_dir = os.getcwd()

    # List all files and directories
    files = []
    for root, dirs, filenames in os.walk(current_dir):
        # Calculate depth
        depth = root.count(os.sep) - current_dir.count(os.sep)
        if depth > max_depth:
            continue

        # Add directories
        for dir_name in dirs:
            files.append(os.path.join(root, dir_name))
        # Add files
        for filename in filenames:
            files.append(os.path.join(root, filename))

    # Print or return the files
    print("Files in function:")
    for file in files:
        print(file)

    return {"files": files}

@functions_framework.http
def list_output_files(request):
    """List all files in the output directories

    This is a utility function for debugging purposes.
    """
    # Get query parameters
    request_args = request.args if request.args else {}
    max_depth = int(request_args.get('max_depth', '3'))

    output_dirs = ['outputs', 'data', 'downloads']
    files = []

    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            continue

        for root, dirs, filenames in os.walk(output_dir):
            # Calculate depth
            depth = root.count(os.sep) - output_dir.count(os.sep)
            if depth > max_depth:
                continue

            for dir_name in dirs:
                files.append(os.path.join(root, dir_name))
            for filename in filenames:
                files.append(os.path.join(root, filename))

    return {"output_files": files}

if __name__ == "__main__":
    # Filter resource tracker warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    import argparse

    # Logging is already set up at the top of the file

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process Twitch VOD audio analysis')
    parser.add_argument('url', type=str, help='Twitch VOD URL to process')
    args = parser.parse_args()

    # Run the pipeline directly with process_video for local execution
    try:
        # Process the video
        result = asyncio.run(process_video(args.url))

        # Extract video_id from the result
        video_id = result.get("video_id")
        if video_id:
            # Upload results to Convex
            logger.info("Uploading results to Convex...")
            upload_success = upload_to_convex(video_id)
            if upload_success:
                logger.info("Successfully uploaded all files to Convex")
            else:
                logger.error("Failed to upload some or all files to Convex")

            # Update video status
            twitch_info = result.get("twitch_info", {})
            update_success = update_video_status(video_id, "completed", twitch_info)
            if update_success:
                logger.info("Successfully updated video status in Convex")
            else:
                logger.error("Failed to update video status in Convex")
        else:
            logger.error("No video_id found in result, cannot upload to Convex")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise