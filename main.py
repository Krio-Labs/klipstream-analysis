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
import platform
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

def detect_gpu_capabilities():
    """
    Detect available GPU capabilities for transcription optimization

    Returns:
        dict: GPU capabilities including type, availability, and memory
    """
    gpu_info = {
        "nvidia_cuda_available": False,
        "apple_metal_available": False,
        "gpu_memory_gb": 0,
        "recommended_method": "deepgram"
    }

    try:
        # Check for NVIDIA CUDA
        import torch
        if torch.cuda.is_available():
            gpu_info["nvidia_cuda_available"] = True
            gpu_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info["recommended_method"] = "parakeet"
            logger.info(f"NVIDIA CUDA detected: {gpu_info['gpu_memory_gb']:.1f}GB memory")
        else:
            logger.info("NVIDIA CUDA not available")
    except ImportError:
        logger.info("PyTorch not available for CUDA detection")

    try:
        # Check for Apple Silicon Metal Performance Shaders
        if platform.system() == "Darwin":  # macOS
            import subprocess
            result = subprocess.run(["system_profiler", "SPHardwareDataType"],
                                  capture_output=True, text=True)
            if "Apple" in result.stdout and ("M1" in result.stdout or "M2" in result.stdout or "M3" in result.stdout):
                gpu_info["apple_metal_available"] = True
                gpu_info["recommended_method"] = "parakeet"
                logger.info("Apple Silicon with Metal Performance Shaders detected")
    except Exception as e:
        logger.debug(f"Apple Silicon detection failed: {e}")

    return gpu_info

def setup_gcs_authentication():
    """
    Set up Google Cloud Storage authentication
    """
    try:
        # Check if we're in Cloud Run (service account automatically available)
        if os.environ.get('K_SERVICE'):
            logger.info("🔐 Running in Cloud Run - using service account authentication")
            return True

        # Check if service account key file exists
        service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if service_account_path and Path(service_account_path).exists():
            logger.info(f"🔐 Using service account key: {service_account_path}")
            return True

        # Try to use gcloud auth application-default login
        import subprocess
        try:
            result = subprocess.run(['gcloud', 'auth', 'application-default', 'print-access-token'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("🔐 Using gcloud application-default credentials")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # If no authentication found, try to authenticate
        logger.warning("🔐 No GCS authentication found. Attempting to authenticate...")
        try:
            auth_result = subprocess.run(['gcloud', 'auth', 'application-default', 'login'],
                                       timeout=60)
            if auth_result.returncode == 0:
                logger.info("🔐 GCS authentication successful")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"🔐 GCS authentication failed: {e}")
            logger.error("Please run: gcloud auth application-default login")
            return False

        return False

    except Exception as e:
        logger.error(f"🔐 GCS authentication setup failed: {e}")
        return False

def configure_transcription_environment():
    """
    Configure transcription environment based on available hardware and user preferences
    """
    # Detect GPU capabilities
    gpu_info = detect_gpu_capabilities()

    # Set environment variables for transcription configuration
    transcription_config = {
        "TRANSCRIPTION_METHOD": os.environ.get("TRANSCRIPTION_METHOD", "auto"),
        "ENABLE_GPU_TRANSCRIPTION": os.environ.get("ENABLE_GPU_TRANSCRIPTION", "true"),
        "ENABLE_FALLBACK": os.environ.get("ENABLE_FALLBACK", "true"),
        "COST_OPTIMIZATION": os.environ.get("COST_OPTIMIZATION", "true"),
        "GPU_MEMORY_LIMIT_GB": os.environ.get("GPU_MEMORY_LIMIT_GB", "20"),
        "PARAKEET_MODEL_NAME": os.environ.get("PARAKEET_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
    }

    # Auto-configure based on detected hardware
    if transcription_config["TRANSCRIPTION_METHOD"] == "auto":
        if gpu_info["nvidia_cuda_available"] and gpu_info["gpu_memory_gb"] >= 4:
            transcription_config["TRANSCRIPTION_METHOD"] = "parakeet"
            logger.info("Auto-configured for NVIDIA Parakeet GPU transcription")
        elif gpu_info["apple_metal_available"]:
            transcription_config["TRANSCRIPTION_METHOD"] = "parakeet"
            logger.info("Auto-configured for Apple Silicon Parakeet transcription")
        else:
            transcription_config["TRANSCRIPTION_METHOD"] = "deepgram"
            logger.info("Auto-configured for Deepgram cloud transcription")

    # Apply configuration to environment
    for key, value in transcription_config.items():
        os.environ[key] = str(value)
        logger.debug(f"Set {key}={value}")

    return transcription_config, gpu_info

def configure_gpu_acceleration(gpu_info):
    """
    Configure GPU acceleration for various pipeline processes

    Args:
        gpu_info: GPU capabilities information

    Returns:
        dict: GPU acceleration configuration
    """
    gpu_config = {
        "transcription_gpu": False,
        "audio_processing_gpu": False,
        "sentiment_analysis_gpu": False,
        "waveform_gpu": False
    }

    # Enable GPU acceleration based on available hardware
    if gpu_info["nvidia_cuda_available"] or gpu_info["apple_metal_available"]:
        gpu_config["transcription_gpu"] = True
        logger.info("🚀 GPU acceleration enabled for transcription")

        # Enable audio processing GPU acceleration
        if gpu_info["gpu_memory_gb"] >= 4:  # Require at least 4GB for audio processing
            gpu_config["audio_processing_gpu"] = True
            logger.info("🚀 GPU acceleration enabled for audio processing")

        # Enable sentiment analysis GPU acceleration for larger models
        if gpu_info["gpu_memory_gb"] >= 8:  # Require 8GB+ for sentiment models
            gpu_config["sentiment_analysis_gpu"] = True
            logger.info("🚀 GPU acceleration enabled for sentiment analysis")

        # Enable waveform GPU acceleration (lightweight)
        gpu_config["waveform_gpu"] = True
        logger.info("🚀 GPU acceleration enabled for waveform generation")

    # Set environment variables for GPU acceleration
    for process, enabled in gpu_config.items():
        env_var = f"ENABLE_{process.upper()}"
        os.environ[env_var] = "true" if enabled else "false"
        logger.debug(f"Set {env_var}={enabled}")

    return gpu_config

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
        logger.info(f"🎬 Starting pipeline for video: {video_id}")

        # Set up Google Cloud Storage authentication
        gcs_auth_success = setup_gcs_authentication()
        if not gcs_auth_success:
            logger.warning("⚠️  GCS authentication failed - file uploads may not work")

        # Configure transcription environment based on available hardware
        transcription_config, gpu_info = configure_transcription_environment()
        logger.info(f"🎤 Transcription method: {transcription_config['TRANSCRIPTION_METHOD']}")
        logger.info(f"🖥️  GPU capabilities: CUDA={gpu_info['nvidia_cuda_available']}, Metal={gpu_info['apple_metal_available']}")

        # Configure GPU acceleration for other pipeline processes
        gpu_config = configure_gpu_acceleration(gpu_info)
        logger.info(f"🚀 GPU acceleration: {sum(gpu_config.values())}/{len(gpu_config)} processes enabled")

        # Initialize file manager
        file_manager = FileManager(video_id)

        # Update Convex status to "Queued"
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
        logger.info("📥 Stage 1: Downloading and processing raw files...")
        raw_start = time.time()

        try:
            raw_result = await process_raw_files(url)
            if not raw_result or raw_result.get("status") != "completed":
                raise RuntimeError("Raw pipeline failed to complete successfully")

            raw_end = time.time()
            raw_duration = raw_end - raw_start
            stage_times["raw_pipeline"] = raw_duration
            logger.info(f"✅ Stage 1 completed in {raw_duration:.1f}s")

            # Extract necessary information from raw pipeline result
            video_id = raw_result["video_id"]
            files = raw_result["files"]
            twitch_info = raw_result.get("twitch_info", {})
            transcription_metadata = raw_result.get("transcription_metadata", {})

        except Exception as e:
            logger.error(f"❌ Stage 1 failed: {str(e)}")
            raise

        # STAGE 2: Analysis Pipeline
        logger.info("🔍 Stage 2: Analyzing content...")
        analysis_start = time.time()

        try:
            analysis_result = await process_analysis(video_id)
            if not analysis_result or analysis_result.get("status") != "completed":
                # Check if this is a development environment
                if os.environ.get('ENVIRONMENT', 'production') == 'development':
                    logger.warning("⚠️  Analysis had issues but continuing in development mode")
                    analysis_result = {"status": "completed", "video_id": video_id}
                else:
                    raise RuntimeError("Analysis pipeline failed to complete successfully")

            analysis_end = time.time()
            analysis_duration = analysis_end - analysis_start
            stage_times["analysis_pipeline"] = analysis_duration
            logger.info(f"✅ Stage 2 completed in {analysis_duration:.1f}s")

        except Exception as e:
            logger.error(f"❌ Stage 2 failed: {str(e)}")
            # In development mode, continue with a warning
            if os.environ.get('ENVIRONMENT', 'production') == 'development':
                logger.warning("⚠️  Analysis failed but continuing in development mode")
                analysis_result = {"status": "completed", "video_id": video_id}
                analysis_end = time.time()
                analysis_duration = analysis_end - analysis_start
                stage_times["analysis_pipeline"] = analysis_duration
            else:
                raise

        # Calculate total time
        end_time = time.time()
        total_duration = end_time - start_time

        # Log final file size summary
        if files:
            log_final_file_summary(video_id, files)

        # Generate summary report
        logger.info(f"🎉 Pipeline completed successfully!")
        logger.info(f"   Video ID: {video_id}")
        logger.info(f"   Stage 1: {stage_times.get('raw_pipeline', 0):.1f}s")
        logger.info(f"   Stage 2: {stage_times.get('analysis_pipeline', 0):.1f}s")
        logger.info(f"   Total: {total_duration:.1f}s")

        # Update Convex status to "Completed"
        convex_manager.update_video_status(video_id, STATUS_COMPLETED)

        # Prepare final result with transcription metadata
        result = {
            "status": "completed",
            "video_id": video_id,
            "twitch_info": twitch_info,
            "files": files,
            "stage_times": stage_times,
            "total_duration": total_duration
        }

        # Add transcription metadata if available
        if transcription_metadata:
            result["transcription_metadata"] = transcription_metadata

        return result

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

def log_final_file_summary(video_id: str, files: dict):
    """
    Log a final comprehensive summary of all generated files and their sizes

    Args:
        video_id (str): The video ID
        files (dict): Dictionary containing file paths
    """
    logger.info("")
    logger.info("🎯 FINAL FILE SIZE SUMMARY")
    logger.info("=" * 60)

    total_size_bytes = 0
    file_count = 0

    # Define file categories for better organization
    file_categories = {
        "📹 Video & Audio": ["video_file", "audio_file"],
        "📝 Transcripts": ["segments_file", "words_file", "paragraphs_file"],
        "💬 Chat Data": ["chat_file", "json_file"],
        "🌊 Waveforms": ["waveform_file"],
        "📊 Analysis": ["analysis_file", "integrated_file"]
    }

    for category, file_keys in file_categories.items():
        category_size = 0
        category_files = []

        for key in file_keys:
            if key in files and files[key]:
                file_path = Path(files[key])
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    category_size += file_size
                    total_size_bytes += file_size
                    file_count += 1
                    category_files.append(f"    • {file_path.name}: {file_size_mb:.1f} MB")

        if category_files:
            category_size_mb = category_size / (1024 * 1024)
            logger.info(f"{category} ({category_size_mb:.1f} MB):")
            for file_info in category_files:
                logger.info(file_info)

    # Overall summary
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)

    logger.info("=" * 60)
    if total_size_gb >= 1.0:
        logger.info(f"🎯 TOTAL OUTPUT: {file_count} files, {total_size_gb:.2f} GB ({total_size_mb:.1f} MB)")
    else:
        logger.info(f"🎯 TOTAL OUTPUT: {file_count} files, {total_size_mb:.1f} MB")
    logger.info("=" * 60)
    logger.info("")

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
