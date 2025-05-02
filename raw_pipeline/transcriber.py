"""
Transcriber Module

This module handles transcribing audio files using AssemblyAI.
"""

import os
import assemblyai as aai
import csv
import json
from pathlib import Path

from utils.config import (
    RAW_AUDIO_DIR,
    RAW_TRANSCRIPTS_DIR,
    ASSEMBLYAI_API_KEY
)
from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("transcriber", "transcription.log")

class TranscriptionHandler:
    """Class for handling audio transcription"""

    def __init__(self):
        """Initialize the transcription handler"""
        self._setup_assemblyai()

    def _setup_assemblyai(self):
        """Set up AssemblyAI with API key"""
        if not ASSEMBLYAI_API_KEY:
            raise ValueError("ASSEMBLYAI_API_KEY environment variable must be set")

        aai.settings.api_key = ASSEMBLYAI_API_KEY
        logger.info("AssemblyAI API configured")

    async def process_audio_files(self, video_id, audio_file_path=None, output_dir=None):
        """
        Process audio files for a specific video ID

        Args:
            video_id (str): The ID of the video to process
            audio_file_path (str, optional): Path to the audio file. If not provided, will search in standard locations.
            output_dir (str, optional): Directory to save transcript files. If not provided, uses RAW_TRANSCRIPTS_DIR.

        Returns:
            dict: Dictionary with transcript_file, paragraphs_file, and words_file paths
        """
        try:
            # Determine audio file path
            if audio_file_path and os.path.exists(audio_file_path):
                audio_file = audio_file_path
            else:
                # Try different locations in order of preference
                possible_audio_paths = [
                    RAW_AUDIO_DIR / f"audio_{video_id}.wav",
                    Path(f"outputs/audio_{video_id}.wav"),
                    Path(f"/tmp/outputs/audio_{video_id}.wav")
                ]

                for path in possible_audio_paths:
                    if path.exists():
                        audio_file = str(path)
                        break
                else:
                    logger.error(f"Audio file not found for video ID: {video_id}")
                    return None

            # Determine output directory
            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = RAW_TRANSCRIPTS_DIR

            # Create output directory if it doesn't exist
            output_dir.mkdir(exist_ok=True, parents=True)

            # Get base name for output files
            base_name = f"audio_{video_id}"

            # Transcribe audio
            logger.info(f"Processing audio file: {audio_file}")
            logger.info(f"Saving transcripts to: {output_dir}")

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

            # Save raw transcript to file temporarily
            transcript_json_path = output_dir / f"{base_name}_transcript.json"
            with open(transcript_json_path, 'w') as f:
                json.dump(transcript.json_response, f, indent=2)
            logger.info(f"Transcript JSON saved temporarily to: {transcript_json_path}")

            # Save word-level timestamps
            words_file = output_dir / f"{base_name}_words.csv"
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
            logger.info(f"Word timestamps saved to: {words_file}")

            # Get paragraphs using dedicated endpoint
            paragraphs_file = output_dir / f"{base_name}_paragraphs.csv"
            headers = {'authorization': ASSEMBLYAI_API_KEY}
            paragraphs_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript.id}/paragraphs"

            logger.info(f"Getting paragraphs from endpoint: {paragraphs_endpoint}")

            import requests
            response = requests.get(paragraphs_endpoint, headers=headers)
            if response.status_code == 200:
                data = response.json()
                paragraphs = data.get('paragraphs', [])

                logger.info(f"Retrieved {len(paragraphs)} paragraphs from AssemblyAI")

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
                logger.info(f"Paragraphs saved to: {paragraphs_file}")
            else:
                logger.error(f"Failed to get paragraphs: {response.status_code}")
                logger.error(f"Response content: {response.text}")

                # Fallback: Try to extract paragraphs from the transcript object
                logger.info("Attempting to extract paragraphs from transcript object...")
                paragraphs = []

                # Check if transcript has paragraphs attribute
                if hasattr(transcript, 'paragraphs') and transcript.paragraphs:
                    for para in transcript.paragraphs:
                        paragraphs.append({
                            'start': para.start,
                            'end': para.end,
                            'text': para.text
                        })

                    logger.info(f"Extracted {len(paragraphs)} paragraphs from transcript object")

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
                    logger.info(f"Paragraphs saved to: {paragraphs_file}")
                else:
                    logger.error("Failed to extract paragraphs from transcript object")
                    raise Exception(f"Failed to get paragraphs: {response.status_code}")

            # Delete the temporary transcript JSON file
            try:
                # Use os.path.exists instead of Path.exists()
                transcript_path_str = str(transcript_json_path)
                logger.info(f"Attempting to delete temporary transcript JSON file: {transcript_path_str}")
                print(f"Attempting to delete temporary transcript JSON file: {transcript_path_str}")

                if os.path.exists(transcript_path_str):
                    os.remove(transcript_path_str)
                    logger.info(f"Successfully deleted temporary transcript JSON file: {transcript_path_str}")
                    print(f"Successfully deleted temporary transcript JSON file: {transcript_path_str}")
                else:
                    logger.warning(f"Transcript JSON file not found for deletion: {transcript_path_str}")
                    print(f"Transcript JSON file not found for deletion: {transcript_path_str}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary transcript JSON file: {str(e)}")
                print(f"Failed to delete temporary transcript JSON file: {str(e)}")

            return {
                "paragraphs_file": paragraphs_file,
                "words_file": words_file
            }

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
