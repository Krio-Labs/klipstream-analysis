import os
import logging
import assemblyai as aai
import aiohttp
import csv
from dotenv import load_dotenv
import asyncio

class TranscriptionHandler:
    def __init__(self):
        """Initialize the transcription handler"""
        self._setup_logging()
        self._setup_assemblyai()

    def _setup_logging(self):
        """Configure logging with timestamp, level, and message"""
        # Use /tmp for log files in cloud environment
        log_file = '/tmp/transcription.log'

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

    async def process_audio_files(self, video_id, input_dir="/tmp/outputs", max_retries=3, timeout=300):
        """Process audio files in the specified directory for a specific video ID"""
        try:
            # Look for specific audio file with video ID
            audio_file = os.path.join(input_dir, f"audio_{video_id}.wav")

            if not os.path.exists(audio_file):
                logging.error(f"Audio file not found: {audio_file}")
                return None

            logging.info(f"Processing {audio_file}...")

            # Configure AssemblyAI with timeout settings
            aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

            # Transcribe the audio file
            transcript = aai.Transcriber().transcribe(audio_file)

            # Save word-level timestamps
            words_file = os.path.join(input_dir, f"audio_{video_id}_words.csv")
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
            paragraphs_file = os.path.join(input_dir, f"audio_{video_id}_paragraphs.csv")
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

    async def _transcribe_audio(self, audio_file, base_name, input_dir):
        """Transcribe audio using AssemblyAI"""
        logging.info(f"Starting transcription for {base_name}")

        try:
            # Configure transcription
            transcriber = aai.Transcriber()
            config = aai.TranscriptionConfig(
                speech_model="nano",
                language_detection=True
            )

            # Get transcript
            transcript = transcriber.transcribe(audio_file, config=config)
            if transcript.error:
                raise Exception(f"Transcription failed: {transcript.error}")

            # Save transcript ID for reference
            transcript_id = transcript.id
            logging.info(f"Transcript ID: {transcript_id}")

            # Save word-level timestamps
            words_file = os.path.join(input_dir, f"{base_name}_words.csv")
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

            # Get paragraphs and sentences using API endpoint
            headers = {'authorization': aai.settings.api_key}
            base_url = "https://api.assemblyai.com/v2/transcript"

            # Save paragraphs to CSV
            paragraphs_endpoint = f"{base_url}/{transcript_id}/paragraphs"
            paragraphs_file = os.path.join(input_dir, f"{base_name}_paragraphs.csv")

            response = requests.get(paragraphs_endpoint, headers=headers)
            if response.status_code == 200:
                paragraphs = response.json()['paragraphs']
                with open(paragraphs_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['start_time', 'end_time', 'text'])
                    for para in paragraphs:
                        writer.writerow([
                            para['start'] / 1000,
                            para['end'] / 1000,
                            para['text']
                        ])
                logging.info(f"Paragraphs saved to: {paragraphs_file}")

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