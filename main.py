import asyncio
import logging
import os
from pathlib import Path
from audio_downloader import TwitchVideoDownloader
from audio_transcription import TranscriptionHandler
from audio_sentiment import main as sentiment_analysis
from audio_waveform import process_audio_file
from audio_analysis import analyze_chat_highlights, plot_audio_waveform, separate_speech_and_noise, plot_combined_loudness

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

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
        logging.info(f"Audio downloaded successfully: {audio_file}")

        # Step 2: Transcribe audio
        logging.info("Step 2: Transcribing audio...")
        transcriber = TranscriptionHandler()
        await transcriber.process_audio_files()
        logging.info("Transcription completed")

        # Step 3: Sentiment analysis
        logging.info("Step 3: Performing sentiment analysis...")
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable must be set")
        
        video_id = downloader.extract_video_id(url)
        paragraphs_file = output_dir / f"audio_{video_id}_paragraphs.csv"
        await sentiment_analysis(input_file=str(paragraphs_file), api_key=api_key)
        logging.info("Sentiment analysis completed")

        # Step 4: Generate waveform
        logging.info("Step 4: Generating audio waveform...")
        waveform_data = process_audio_file()
        waveform_file = output_dir / "audio_waveform.json"
        logging.info(f"Waveform data generated: {len(waveform_data)} points")

        # Step 5: Audio analysis
        logging.info("Step 5: Performing audio analysis...")
        
        # Analyze chat highlights
        await analyze_chat_highlights(str(paragraphs_file), str(output_dir))
        
        # Plot audio visualizations
        audio_path = str(audio_file)
        plot_audio_waveform(audio_path, str(output_dir))
        
        # Separate speech and noise, then plot combined loudness
        speech_path, noise_path = separate_speech_and_noise(audio_path, str(output_dir))
        if speech_path and noise_path:
            plot_combined_loudness(audio_path, speech_path, noise_path, str(output_dir))
        
        logging.info("Audio analysis completed")
        
        logging.info("Pipeline completed successfully!")
        return True

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

async def main():
    """Main entry point"""
    try:
        # Get Twitch video URL from user
        url = input("Enter Twitch video URL: ")
        
        # Run the pipeline
        await run_pipeline(url)
        
    except KeyboardInterrupt:
        logging.info("Pipeline interrupted by user")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 