import os
import logging
import assemblyai as aai
import requests
import csv
from pathlib import Path

class TranscriptionHandler:
    def __init__(self):
        """Initialize the transcription handler"""
        self._setup_logging()
        self._setup_assemblyai()

    def _setup_logging(self):
        """Configure logging with timestamp, level, and message"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('transcription.log'),
                logging.StreamHandler()
            ]
        )

    def _setup_assemblyai(self):
        """Setup AssemblyAI with API key"""
        aai.settings.api_key = "c37776f0425148949b6becd9f145550a"
        logging.info("AssemblyAI API configured")

    async def process_audio_files(self, input_dir="outputs"):
        """Process all audio files in the specified directory"""
        try:
            # Get all audio files
            audio_files = [f for f in os.listdir(input_dir) 
                         if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]

            if not audio_files:
                logging.info(f"No audio files found in {input_dir}")
                return

            for audio_file in audio_files:
                input_path = os.path.join(input_dir, audio_file)
                base_name = Path(audio_file).stem
                
                logging.info(f"Processing {audio_file}...")
                
                try:
                    transcript = await self._transcribe_audio(input_path, base_name, input_dir)
                    logging.info(f"Successfully processed {audio_file}")
                except Exception as e:
                    logging.error(f"Error processing {audio_file}: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Process failed: {str(e)}")
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
    
    async def main():
        handler = TranscriptionHandler()
        await handler.process_audio_files()

    asyncio.run(main())