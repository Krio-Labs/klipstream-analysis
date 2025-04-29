import asyncio
import logging
from pathlib import Path
import json
import warnings
import multiprocessing
import atexit
import os
import functions_framework
import tempfile
import shutil
from dotenv import load_dotenv

from audio_downloader import TwitchVideoDownloader
from audio_transcription import TranscriptionHandler
from audio_sentiment import sentiment_analysis
from audio_analysis import analyze_transcription_highlights
from audio_analysis import plot_metrics
from audio_waveform import process_audio_file
from chat_download import download_chat
from chat_processor import process_chat_data
from chat_sentiment import analyze_chat_sentiment
from chat_analysis import analyze_chat_intervals, analyze_chat_highlights

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cleanup():
    try:
        multiprocessing.resource_tracker._resource_tracker = None
    except Exception:
        pass

atexit.register(cleanup)

import asyncio
from gcs_upload import upload_files, update_video_status as gcs_update_video_status

def upload_to_gcs(video_id, video_file=None, audio_file=None, waveform_file=None):
    """
    Upload files to Google Cloud Storage

    Args:
        video_id: The Twitch video ID
        video_file: Optional path to the video file
        audio_file: Optional path to the audio file
        waveform_file: Optional path to the waveform file
    """
    try:
        logging.info(f"Uploading files for video {video_id} to Google Cloud Storage")

        # Prepare specific files to upload if provided
        specific_files = []
        if video_file and os.path.exists(video_file):
            specific_files.append(video_file)
            logging.info(f"Adding video file to upload: {video_file}")

        if audio_file and os.path.exists(audio_file):
            specific_files.append(audio_file)
            logging.info(f"Adding audio file to upload: {audio_file}")

        if waveform_file and os.path.exists(waveform_file):
            specific_files.append(waveform_file)
            logging.info(f"Adding waveform file to upload: {waveform_file}")

        # Upload files
        uploaded_files = upload_files(video_id, specific_files if specific_files else None)
        logging.info(f"Successfully uploaded {len(uploaded_files)} files to GCS")
        return True
    except Exception as e:
        logging.error(f"Error uploading files to GCS: {str(e)}")
        return False

def update_video_status(video_id, status, twitch_info=None):
    """Update the status of a video in the database"""
    try:
        logging.info(f"Updating video {video_id} status to {status}")
        success = gcs_update_video_status(video_id, status, twitch_info)

        if success:
            logging.info(f"Successfully updated video {video_id} status to {status}")
            return True
        else:
            logging.error(f"Failed to update video {video_id} status")
            return False
    except Exception as e:
        logging.error(f"Error updating video status: {str(e)}")
        logging.exception("Full exception details:")
        return False

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
        # Extract the Twitch video URL from the request

        # Clean up directories from previous runs
        logging.info("Cleaning up directories from previous runs...")
        try:
            from cleanup import cleanup_directories
            cleanup_directories()
        except Exception as e:
            logging.warning(f"Error during cleanup: {str(e)}")
            # Continue with the pipeline even if cleanup fails

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
        video_id = TwitchVideoDownloader().extract_video_id(url)

        # Update video status to "processing"
        update_video_status(video_id, "processing")

        try:
            # Run the async pipeline using asyncio
            result = asyncio.run(process_video(url))

            # Get Twitch metadata
            twitch_info = result.get("twitch_info", {})

            # Upload results to Google Cloud Storage
            # This will handle both outputs and data directories
            video_file = result.get("video_file")
            audio_file = result.get("audio_file")
            waveform_file = result.get("waveform_file")
            upload_success = upload_to_gcs(video_id, video_file, audio_file, waveform_file)

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
        logging.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}, 500

async def process_video(url: str):
    """
    Contains the original run_pipeline logic
    """
    try:
        # Clean up directories from previous runs
        logging.info("Cleaning up directories from previous runs...")
        try:
            from cleanup import cleanup_directories
            cleanup_directories()
        except Exception as e:
            logging.warning(f"Error during cleanup: {str(e)}")
            # Continue with the pipeline even if cleanup fails

        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")  # Use project-relative path
        output_dir.mkdir(exist_ok=True, parents=True)

        # Create data directory for chat files
        data_dir = Path("data")  # Use project-relative path
        data_dir.mkdir(exist_ok=True, parents=True)

        # Create downloads directory
        downloads_dir = Path("downloads")
        downloads_dir.mkdir(exist_ok=True, parents=True)

        # Get video ID early since we'll need it for multiple steps
        downloader = TwitchVideoDownloader()
        video_id = downloader.extract_video_id(url)

        # Get Twitch video metadata
        twitch_info = downloader.get_video_metadata(video_id)

        # Step 1: Download audio and video
        logging.info("Step 1: Downloading audio and video from Twitch...")
        download_result = await downloader.process_video(url)
        if not download_result:
            raise RuntimeError("Failed to download files")

        audio_file = download_result["audio_file"]
        video_file = download_result["video_file"]
        logging.info(f"Audio downloaded successfully: {audio_file}")
        logging.info(f"Video downloaded successfully: {video_file}")

        # Step 2: Transcribe audio with retry logic
        logging.info("Step 2: Transcribing audio...")
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                transcriber = TranscriptionHandler()
                transcription_result = await transcriber.process_audio_files(video_id)
                if transcription_result:
                    logging.info("Transcription completed successfully")
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Transcription attempt {attempt + 1} failed: {str(e)}")
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logging.error("All transcription attempts failed")
                    raise RuntimeError("Transcription failed after all retry attempts") from e

        # Step 3: Sentiment analysis
        logging.info("Step 3: Performing sentiment analysis...")
        sentiment_result = sentiment_analysis(video_id=video_id)
        if not sentiment_result:
            logging.error("Sentiment analysis failed to complete successfully")
            raise RuntimeError("Sentiment analysis failed")
        logging.info("Sentiment analysis completed successfully")

        # Step 4: Generate waveform
        logging.info("Step 4: Generating audio waveform...")
        waveform_data = process_audio_file(video_id)
        waveform_file = None
        if waveform_data:
            waveform_file = output_dir / f"audio_{video_id}_waveform.json"
            with open(waveform_file, 'w') as f:
                json.dump(waveform_data, f)
            logging.info(f"Waveform data generated: {len(waveform_data)} points")
        else:
            logging.error("Failed to generate waveform data")

        # Step 5: Audio analysis
        logging.info("Step 5: Performing audio analysis...")
        # Check for paragraphs file in outputs directory
        paragraphs_file = output_dir / f"audio_{video_id}_paragraphs.csv"

        # Verify the file exists
        if not paragraphs_file.exists():
            raise FileNotFoundError(f"Required file not found: {paragraphs_file}")

        grouped_data = analyze_transcription_highlights(str(paragraphs_file), str(output_dir))
        if grouped_data is not None:
            logging.info("Chat highlights analysis completed")

            # Plot metrics with video_id
            plot_metrics(str(output_dir), video_id)
            logging.info("Metrics plotting completed")
        else:
            raise RuntimeError("Chat highlights analysis failed")

        logging.info("Audio analysis completed")

        # Step 6: Download and process chat
        logging.info("Step 6: Downloading and processing chat data...")
        try:
            chat_file = download_chat(video_id, data_dir)
            logging.info(f"Chat data downloaded successfully: {chat_file}")

            # Process chat data
            process_chat_data(video_id)
            logging.info("Chat data processed successfully")

            # Add chat sentiment analysis
            logging.info("Step 7: Analyzing chat sentiment...")
            sentiment_output = analyze_chat_sentiment(video_id)
            if sentiment_output:
                logging.info(f"Chat sentiment analysis completed. Results saved to: {sentiment_output}")

                # Only run chat analysis if sentiment analysis succeeded
                logging.info("Step 8: Running chat analysis...")
                interval_stats = analyze_chat_intervals(video_id, str(output_dir))
                if interval_stats is not None:
                    logging.info("Chat intervals analysis completed")

                    # Run highlight analysis
                    analyze_chat_highlights(video_id, str(output_dir))
                    logging.info("Chat highlights analysis completed")

        except Exception as e:
            logging.error(f"Failed to process chat data: {e}")
            # Continue pipeline even if chat operations fail
            logging.warning("Continuing pipeline without chat data")

        logging.info("Pipeline completed successfully!")

        # Return a dictionary with results and metadata
        return {
            "status": "completed",
            "video_id": video_id,
            "twitch_info": twitch_info,
            "video_file": str(video_file),
            "audio_file": str(audio_file),
            "waveform_file": str(waveform_file) if waveform_file else None
        }

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
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

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
            # Upload results to Google Cloud Storage
            logging.info("Uploading results to Google Cloud Storage...")
            video_file = result.get("video_file")
            audio_file = result.get("audio_file")
            waveform_file = result.get("waveform_file")
            upload_success = upload_to_gcs(video_id, video_file, audio_file, waveform_file)
            if upload_success:
                logging.info("Successfully uploaded all files to Google Cloud Storage")
            else:
                logging.error("Failed to upload some or all files to Google Cloud Storage")

            # Update video status
            twitch_info = result.get("twitch_info", {})
            update_success = update_video_status(video_id, "completed", twitch_info)
            if update_success:
                logging.info("Successfully updated video status in database")
            else:
                logging.error("Failed to update video status in database")
        else:
            logging.error("No video_id found in result, cannot upload to Google Cloud Storage")

    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Process failed: {str(e)}")
        raise