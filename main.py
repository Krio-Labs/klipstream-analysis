#!/usr/bin/env python3
"""
Klipstream Analysis Main Module

This module serves as the main entry point for the Klipstream Analysis project.
It orchestrates the entire pipeline, from downloading raw files to analyzing them
and uploading the results to Google Cloud Storage.

The pipeline consists of two main stages:
1. Raw Pipeline: Downloads video, extracts audio, generates transcripts, etc.
2. Analysis Pipeline: Performs sentiment analysis, highlight detection, etc.

This module can be run as a standalone script or as a Cloud Function.
"""

import asyncio
from pathlib import Path
import warnings
import multiprocessing
import atexit
import os
import time
import argparse
import yaml
import functions_framework
from dotenv import load_dotenv

# Import utilities
from utils.logging_setup import setup_logger
from utils.helpers import extract_video_id
from utils.config import create_directories, BASE_DIR, USE_GCS
from utils.file_manager import FileManager
from utils.convex_client_updated import ConvexManager

# Video processing status constants
STATUS_QUEUED = "Queued"
STATUS_DOWNLOADING = "Downloading"
STATUS_FETCHING_CHAT = "Fetching chat"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_ANALYZING = "Analyzing"
STATUS_FINDING_HIGHLIGHTS = "Finding highlights"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

# Import raw file processor from reorganized structure
from raw_pipeline import process_raw_files

# Import analysis pipeline
from analysis_pipeline import process_analysis

# Load environment variables from .env file
load_dotenv()

# Load environment variables from .env.yaml file
# This is important for Cloud Functions deployment
try:
    if os.path.exists('.env.yaml'):
        with open('.env.yaml', 'r') as f:
            yaml_env = yaml.safe_load(f)
            for key, value in yaml_env.items():
                os.environ[key] = str(value)
except Exception as e:
    print(f"Error loading .env.yaml: {str(e)}")

# Set up logger
logger = setup_logger("main", "main.log")

def cleanup():
    """
    Clean up multiprocessing resources to prevent warnings
    This is registered as an atexit handler
    """
    try:
        multiprocessing.resource_tracker._resource_tracker = None
    except Exception:
        pass

# Register cleanup function to run at exit
atexit.register(cleanup)

async def run_integrated_pipeline(url):
    """
    Run the integrated pipeline (raw + analysis) for a Twitch VOD URL

    Args:
        url (str): Twitch VOD URL

    Returns:
        dict: Results of the pipeline execution
    """
    start_time = time.time()
    stage_times = {}

    # Initialize Convex client
    convex_manager = ConvexManager()

    try:
        # Extract video ID
        video_id = extract_video_id(url)
        logger.info(f"Starting integrated pipeline for video ID: {video_id}")

        # Initialize file manager
        file_manager = FileManager(video_id)

        # Log configuration
        logger.info(f"Using base directory: {BASE_DIR}")
        logger.info(f"Using GCS for file storage: {USE_GCS}")

        # Update Convex status to "Queued"
        logger.info(f"Updating Convex status to '{STATUS_QUEUED}' for video ID: {video_id}")
        convex_manager.update_video_status(video_id, STATUS_QUEUED)

        # Create required directories
        output_dir = BASE_DIR / "output"
        output_dir.mkdir(exist_ok=True, parents=True)

        data_dir = BASE_DIR / "data"
        data_dir.mkdir(exist_ok=True, parents=True)

        downloads_dir = BASE_DIR / "downloads"
        downloads_dir.mkdir(exist_ok=True, parents=True)

        # Create all necessary subdirectories
        create_directories()

        # STAGE 1: Raw Pipeline
        logger.info("STAGE 1: Starting Raw Pipeline")
        logger.info("This stage includes: video download, audio extraction, transcription, waveform generation, and chat download")
        raw_start = time.time()

        try:
            raw_result = await process_raw_files(url)
            if not raw_result or raw_result.get("status") != "completed":
                raise RuntimeError("Raw pipeline failed to complete successfully")

            raw_end = time.time()
            raw_duration = raw_end - raw_start
            stage_times["raw_pipeline"] = raw_duration
            logger.info(f"Raw pipeline completed successfully in {raw_duration:.2f} seconds")

            # Extract necessary information from raw pipeline result
            video_id = raw_result["video_id"]
            files = raw_result["files"]
            twitch_info = raw_result.get("twitch_info", {})

            logger.info(f"Raw files processed for video ID: {video_id}")
            logger.info(f"Generated files: {list(files.keys())}")

        except Exception as e:
            logger.error(f"Raw pipeline failed: {str(e)}")
            raise

        # STAGE 2: Analysis Pipeline
        logger.info("STAGE 2: Starting Analysis Pipeline")
        analysis_start = time.time()

        try:
            analysis_result = await process_analysis(video_id)
            if not analysis_result or analysis_result.get("status") != "completed":
                # Check if this is a development environment
                if os.environ.get('ENVIRONMENT', 'production') == 'development':
                    logger.warning("Analysis pipeline had issues but continuing in development mode")
                    analysis_result = {"status": "completed", "video_id": video_id}
                else:
                    raise RuntimeError("Analysis pipeline failed to complete successfully")

            analysis_end = time.time()
            analysis_duration = analysis_end - analysis_start
            stage_times["analysis_pipeline"] = analysis_duration
            logger.info(f"Analysis pipeline completed in {analysis_duration:.2f} seconds")

        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            # In development mode, continue with a warning
            if os.environ.get('ENVIRONMENT', 'production') == 'development':
                logger.warning("Analysis pipeline failed but continuing in development mode")
                analysis_result = {"status": "completed", "video_id": video_id}
                analysis_end = time.time()
                analysis_duration = analysis_end - analysis_start
                stage_times["analysis_pipeline"] = analysis_duration
            else:
                raise

        # Calculate total time
        end_time = time.time()
        total_duration = end_time - start_time

        # Generate summary report
        logger.info("=" * 50)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Video ID: {video_id}")
        logger.info(f"Raw Pipeline: {stage_times.get('raw_pipeline', 0):.2f} seconds")
        logger.info(f"Analysis Pipeline: {stage_times.get('analysis_pipeline', 0):.2f} seconds")
        logger.info(f"Total Execution Time: {total_duration:.2f} seconds")
        logger.info("=" * 50)

        # Update Convex status to "Completed"
        logger.info(f"Updating Convex status to '{STATUS_COMPLETED}' for video ID: {video_id}")
        convex_manager.update_video_status(video_id, STATUS_COMPLETED)

        return {
            "status": "completed",
            "video_id": video_id,
            "twitch_info": twitch_info,
            "files": files,
            "stage_times": stage_times,
            "total_duration": total_duration
        }

    except Exception as e:
        logger.error(f"Integrated pipeline failed: {str(e)}")
        logger.exception("Full exception details:")

        end_time = time.time()
        total_duration = end_time - start_time

        # Update Convex status to "Failed"
        try:
            # Try to get video_id from the exception context
            if 'video_id' in locals():
                logger.info(f"Updating Convex status to '{STATUS_FAILED}' for video ID: {video_id}")
                convex_manager.update_video_status(video_id, STATUS_FAILED)
            else:
                # Try to extract video ID from URL
                try:
                    extracted_id = extract_video_id(url)
                    logger.info(f"Updating Convex status to '{STATUS_FAILED}' for video ID: {extracted_id}")
                    convex_manager.update_video_status(extracted_id, STATUS_FAILED)
                except Exception as id_error:
                    logger.error(f"Could not extract video ID from URL for status update: {str(id_error)}")
        except Exception as status_error:
            logger.error(f"Failed to update status to 'Failed': {str(status_error)}")

        return {
            "status": "failed",
            "error": str(e),
            "stage_times": stage_times,
            "total_duration": total_duration
        }

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

        # Run the integrated pipeline
        result = asyncio.run(run_integrated_pipeline(url))

        # Convert Path objects to strings for JSON serialization
        serializable_result = convert_paths_to_strings(result)

        return {
            "status": serializable_result.get("status", "failed"),
            "result": serializable_result
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}, 500

def convert_paths_to_strings(obj):
    """
    Recursively convert Path objects to strings for JSON serialization

    Args:
        obj: The object to convert

    Returns:
        The converted object with Path objects replaced by strings
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_paths_to_strings(item) for item in obj)
    else:
        return obj

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

    # Convert any Path objects to strings
    serializable_files = convert_paths_to_strings(files)
    return {"files": serializable_files}

@functions_framework.http
def list_output_files(request):
    """List all files in the output directories

    This is a utility function for debugging purposes.
    """
    # Get query parameters
    request_args = request.args if request.args else {}
    max_depth = int(request_args.get('max_depth', '3'))

    # Check if running in Cloud Functions environment
    is_cloud_function = os.environ.get('K_SERVICE') is not None
    base_dir = Path("/tmp") if is_cloud_function else Path(".")

    output_dirs = [base_dir / 'output', base_dir / 'data', base_dir / 'downloads']
    files = []

    for output_dir in output_dirs:
        if not output_dir.exists():
            continue

        for root, dirs, filenames in os.walk(output_dir):
            # Calculate depth
            root_path = Path(root)
            depth = len(root_path.relative_to(output_dir).parts)
            if depth > max_depth:
                continue

            for dir_name in dirs:
                files.append(Path(root) / dir_name)
            for filename in filenames:
                files.append(Path(root) / filename)

    # Convert Path objects to strings for JSON serialization
    serializable_files = convert_paths_to_strings(files)
    return {"output_files": serializable_files}

if __name__ == "__main__":
    # Filter resource tracker warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process Twitch VOD for analysis')
    parser.add_argument('url', type=str, help='Twitch VOD URL to process')
    args = parser.parse_args()

    try:
        # Run the integrated pipeline
        result = asyncio.run(run_integrated_pipeline(args.url))

        if result.get("status") == "completed":
            print(f"Pipeline completed successfully for video ID: {result.get('video_id')}")
            print(f"Total execution time: {result.get('total_duration', 0):.2f} seconds")
        else:
            print(f"Pipeline failed: {result.get('error', 'Unknown error')}")
            exit(1)

    except KeyboardInterrupt:
        print("Process interrupted by user")
        exit(1)
    except Exception as e:
        print(f"Process failed: {str(e)}")
        exit(1)
