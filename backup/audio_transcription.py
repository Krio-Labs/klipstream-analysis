import os
import logging
import assemblyai as aai
import aiohttp
import csv
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
import asyncio

class TranscriptionHandler:
    def __init__(self):
        """Initialize the transcription handler"""
        self._setup_logging()
        self._setup_assemblyai()

    def _setup_logging(self):
        """Configure logging with timestamp, level, and message"""
        # Use local directory for log files
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'transcription.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _setup_assemblyai(self):
        """Setup AssemblyAI with API key"""
        # Load environment variables
        load_dotenv()

        # Get API key from environment
        api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY environment variable must be set")

        aai.settings.api_key = api_key
        logging.info("AssemblyAI API configured")

    async def process_audio_files(self, video_id, audio_file_path=None, output_dir=None):
        """
        Process audio files for a specific video ID

        Args:
            video_id (str): The ID of the video to process
            audio_file_path (str, optional): Path to the audio file. If not provided, will search in standard locations.
            output_dir (str, optional): Directory to save transcript files. If not provided, uses 'outputs'.

        Returns:
            transcript: The AssemblyAI transcript object
        """
        try:
            # Define directory paths
            raw_dir = Path("Output/Raw")
            raw_audio_dir = raw_dir / "Audio"
            raw_transcripts_dir = raw_dir / "Transcripts"
            outputs_dir = Path("outputs")

            # Determine audio file path
            if audio_file_path and os.path.exists(audio_file_path):
                audio_file = audio_file_path
            else:
                # Try different locations in order of preference
                possible_audio_paths = [
                    raw_audio_dir / f"audio_{video_id}.wav",
                    outputs_dir / f"audio_{video_id}.wav",
                    Path(f"outputs/audio_{video_id}.wav"),
                    Path(f"/tmp/outputs/audio_{video_id}.wav")
                ]

                for path in possible_audio_paths:
                    if path.exists():
                        audio_file = str(path)
                        break
                else:
                    logging.error(f"Audio file not found for video ID: {video_id}")
                    return None

            # Determine output directory
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = raw_transcripts_dir

            # Create output directory if it doesn't exist
            output_path.mkdir(exist_ok=True, parents=True)

            logging.info(f"Processing audio file: {audio_file}")
            logging.info(f"Saving transcripts to: {output_path}")

            # Configure AssemblyAI
            aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

            # Transcribe the audio file
            transcript = aai.Transcriber().transcribe(audio_file)

            # Save the transcript JSON
            transcript_json_path = output_path / f"audio_{video_id}_transcript.json"
            with open(transcript_json_path, 'w') as f:
                json.dump(transcript.json_response, f, indent=2)
            logging.info(f"Transcript JSON saved to: {transcript_json_path}")

            # Save word-level timestamps
            words_file = output_path / f"audio_{video_id}_words.csv"
            with open(words_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['word', 'start_time', 'end_time', 'confidence'])
                for word in transcript.words:
                    writer.writerow([
                        word.text,
                        word.start / 1000,  # Convert to seconds
                        word.end / 1000,
                        word.confidence
                    ])
            logging.info(f"Word timestamps saved to: {words_file}")

            # Get paragraphs using dedicated endpoint
            paragraphs_file = output_path / f"audio_{video_id}_paragraphs.csv"
            headers = {'authorization': os.getenv('ASSEMBLYAI_API_KEY')}
            paragraphs_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript.id}/paragraphs"

            async with aiohttp.ClientSession() as session:
                async with session.get(paragraphs_endpoint, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        paragraphs = data['paragraphs']

                        # Save paragraphs to CSV
                        with open(paragraphs_file, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['start_time', 'end_time', 'text'])
                            for para in paragraphs:
                                writer.writerow([
                                    para['start'] / 1000,  # Convert to seconds
                                    para['end'] / 1000,
                                    para['text']
                                ])
                        logging.info(f"Paragraphs saved to: {paragraphs_file}")
                    else:
                        logging.error(f"Failed to get paragraphs: {response.status}")
                        raise Exception(f"Failed to get paragraphs: {response.status}")

            return transcript

        except Exception as e:
            logging.error(f"Transcription error: {str(e)}")
            raise

if __name__ == "__main__":
    import asyncio
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process audio transcription')
    parser.add_argument('--video-id', type=str, required=True, help='Video ID to process')
    args = parser.parse_args()

    async def main():
        handler = TranscriptionHandler()
        await handler.process_audio_files(args.video_id)  # Pass the video_id here

    asyncio.run(main())
