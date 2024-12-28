import asyncio
import logging
from pathlib import Path
import json
import warnings
import multiprocessing
import atexit
import os

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

def cleanup():
    try:
        multiprocessing.resource_tracker._resource_tracker = None
    except Exception:
        pass

atexit.register(cleanup)

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
        
        # Create outputs directory if it doesn't exist
        output_dir = Path("/tmp/outputs")  # Use /tmp for Cloud Functions
        output_dir.mkdir(exist_ok=True)
        
        # Create data directory for chat files
        data_dir = Path("/tmp/data")  # Use /tmp for Cloud Functions
        data_dir.mkdir(exist_ok=True)
        
        # Run the async pipeline using asyncio
        result = asyncio.run(process_video(url))
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}, 500

async def process_video(url: str):
    """
    Contains the original run_pipeline logic
    """
    try:
        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Create data directory for chat files
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Get video ID early since we'll need it for multiple steps
        downloader = TwitchVideoDownloader()
        video_id = downloader.extract_video_id(url)
        
        # Step 1: Download audio
        logging.info("Step 1: Downloading audio from Twitch...")
        audio_file = await downloader.process_video(url)
        if not audio_file:
            raise RuntimeError("Failed to download audio file")
        logging.info(f"Audio downloaded successfully: {audio_file}")

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
        if waveform_data:
            waveform_file = output_dir / f"audio_{video_id}_waveform.json"
            with open(waveform_file, 'w') as f:
                json.dump(waveform_data, f)
            logging.info(f"Waveform data generated: {len(waveform_data)} points")
        else:
            logging.error("Failed to generate waveform data")

        # Step 5: Audio analysis
        logging.info("Step 5: Performing audio analysis...")
        paragraphs_file = output_dir / f"audio_{video_id}_paragraphs.csv"
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
        return True

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

def list_files(event, context):
    # Get the current working directory
    current_dir = os.getcwd()
    
    # List all files and directories
    files = []
    for root, dirs, filenames in os.walk(current_dir):
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

def list_tmp_files(event, context):
    tmp_dir = '/tmp'
    files = []
    for root, dirs, filenames in os.walk(tmp_dir):
        for dir_name in dirs:
            files.append(os.path.join(root, dir_name))
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    return {"tmp_files": files}

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
        asyncio.run(process_video(args.url))  # Call process_video directly instead of run_pipeline
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Process failed: {str(e)}")
        raise