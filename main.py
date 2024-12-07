import asyncio
import logging
from pathlib import Path
import json
import warnings
import multiprocessing

from audio_downloader import TwitchVideoDownloader
from audio_transcription import TranscriptionHandler
from audio_sentiment import sentiment_analysis
from audio_analysis import analyze_transcription_highlights
from audio_analysis import plot_audio_waveform, plot_metrics
from audio_waveform import process_audio_file

async def run_pipeline(url: str):
    """
    Run the complete audio analysis pipeline
    """
    try:
        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Download audio
        logging.info("Step 1: Downloading audio from Twitch...")
        downloader = TwitchVideoDownloader()
        audio_file = await downloader.process_video(url)
        if not audio_file:
            raise RuntimeError("Failed to download audio file")
        video_id = downloader.extract_video_id(url)
        logging.info(f"Audio downloaded successfully: {audio_file}")

        # Step 2: Transcribe audio
        logging.info("Step 2: Transcribing audio...")
        transcriber = TranscriptionHandler()
        transcription_result = await transcriber.process_audio_files(video_id)
        if not transcription_result:
            raise RuntimeError("Transcription failed")
        logging.info("Transcription completed")

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
            
            # Plot audio visualizations
            audio_path = str(audio_file)
            plot_audio_waveform(audio_path, str(output_dir), video_id)
            logging.info("Audio waveform plotted")
            
            # Plot metrics with video_id
            plot_metrics(grouped_data, str(output_dir), video_id)
            logging.info("Metrics plotting completed")
        else:
            raise RuntimeError("Chat highlights analysis failed")
        
        logging.info("Audio analysis completed")
        logging.info("Pipeline completed successfully!")
        return True

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

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
    
    # Run the pipeline
    try:
        asyncio.run(run_pipeline(args.url))
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Process failed: {str(e)}")
        raise