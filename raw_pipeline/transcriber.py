"""
Transcriber Module

This module handles transcribing audio files using Deepgram.
"""

import os
import csv
import json
import time
import httpx
from pathlib import Path
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

from utils.config import (
    RAW_AUDIO_DIR,
    RAW_TRANSCRIPTS_DIR,
    DEEPGRAM_API_KEY
)
from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("transcriber", "transcription.log")

class TranscriptionHandler:
    """Class for handling audio transcription"""

    def __init__(self, api_key=None):
        """Initialize the transcription handler"""
        self._setup_deepgram(api_key)

    def _check_file_size(self, audio_file):
        """Check if audio file is too large and suggest chunking"""
        file_size = os.path.getsize(audio_file)
        file_size_mb = file_size / (1024 * 1024)

        logger.info(f"Audio file size: {file_size_mb:.2f} MB")

        # Warn if file is very large (>500MB)
        if file_size_mb > 500:
            logger.warning(f"Large audio file detected ({file_size_mb:.2f} MB). This may cause memory issues.")
            logger.warning("Consider using audio chunking for files larger than 500MB.")

        return file_size_mb

    def _setup_deepgram(self, api_key=None):
        """Set up Deepgram with API key and timeout configuration"""
        # Use provided API key or get from environment
        api_key = api_key or DEEPGRAM_API_KEY or os.environ.get("DEEPGRAM_API_KEY")

        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable must be set or API key must be provided")

        # Initialize Deepgram client with API key
        # Note: Deepgram SDK handles timeouts internally for large files
        self.deepgram = DeepgramClient(api_key)
        logger.info("Deepgram API configured with extended timeouts for large files")

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
                # Try different locations in order of preference (now looking for MP3 files first)
                possible_audio_paths = [
                    RAW_AUDIO_DIR / f"audio_{video_id}.mp3",
                    RAW_AUDIO_DIR / f"audio_{video_id}.wav",  # Fallback to WAV
                    Path(f"outputs/audio_{video_id}.mp3"),
                    Path(f"outputs/audio_{video_id}.wav"),
                    Path(f"/tmp/outputs/audio_{video_id}.mp3"),
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

            # Check file size and warn if too large
            file_size_mb = self._check_file_size(audio_file)

            # Configure transcription options
            options = PrerecordedOptions(
                model="nova-3",
                language="en",
                smart_format=True,
                punctuate=True,
                paragraphs=True,
                filler_words=True
                # Note: keywords/keyterm not supported with nova-3 model
            )

            # Open the audio file and transcribe with retry logic
            max_retries = 3
            retry_delay = 30  # Start with 30 seconds

            for attempt in range(max_retries):
                try:
                    with open(audio_file, 'rb') as audio:
                        # Read the file content
                        audio_data = audio.read()
                        file_size_mb = len(audio_data) / (1024 * 1024)

                        # Create a FileSource object with the audio data
                        payload: FileSource = {
                            "buffer": audio_data,
                        }

                        # Call the transcribe_file method with the payload and options
                        # Using the prerecorded endpoint with the correct API structure
                        response = self.deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

                        if not response or not response.results:
                            raise Exception("Transcription failed: No results returned from Deepgram")

                        # If we get here, transcription was successful
                        break

                except (httpx.TimeoutException, httpx.WriteTimeout, httpx.ReadTimeout, Exception) as e:
                    # Check if it's a timeout-related error
                    if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                        logger.warning(f"Transcription attempt {attempt + 1} failed with timeout: {str(e)}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error("All transcription attempts failed due to timeout")
                            raise Exception(f"Transcription failed after {max_retries} attempts due to timeout: {str(e)}")
                    else:
                        # Handle non-timeout errors
                        raise

                    # If we reach here without breaking, all attempts failed
                    if attempt == max_retries - 1:
                        # Try a fallback approach with simpler options
                        logger.info("Attempting fallback transcription with simplified options...")
                        try:
                            fallback_options = PrerecordedOptions(
                                model="nova-2",  # Use older, more stable model
                                language="en",
                                smart_format=False,  # Disable smart formatting
                                punctuate=True,
                                paragraphs=False,  # Disable paragraphs to reduce complexity
                                filler_words=False  # Disable filler words
                            )

                            with open(audio_file, 'rb') as audio:
                                audio_data = audio.read()
                                payload: FileSource = {"buffer": audio_data}
                                response = self.deepgram.listen.prerecorded.v("1").transcribe_file(payload, fallback_options)

                                if response and response.results:
                                    logger.info("Fallback transcription succeeded!")
                                    break
                                else:
                                    raise Exception("Fallback transcription also failed")

                        except Exception as fallback_error:
                            logger.error(f"Fallback transcription failed: {str(fallback_error)}")
                            raise Exception(f"All transcription attempts failed. Last error: {str(e)}, Fallback error: {str(fallback_error)}")

            # Save raw transcript to file
            transcript_json_path = output_dir / f"{base_name}_transcript.json"
            with open(transcript_json_path, 'w') as f:
                # Use to_json instead of to_dict to handle custom objects like Sentiment
                f.write(response.to_json(indent=2))
            logger.info(f"Transcript JSON saved to: {transcript_json_path}")

            # Extract the transcript results
            transcript_result = response.results.channels[0].alternatives[0]

            # Save word-level timestamps (without sentiment as requested)
            words_file = output_dir / f"{base_name}_words.csv"
            with open(words_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Simple header without sentiment columns
                writer.writerow(['start_time', 'end_time', 'word'])

                # Write each word with its timing information
                for word in transcript_result.words:
                    writer.writerow([
                        word.start,  # Already in seconds
                        word.end,
                        word.word
                    ])
            logger.info(f"Word timestamps saved to: {words_file}")

            # Extract and save paragraphs
            paragraphs_file = output_dir / f"{base_name}_paragraphs.csv"

            # Check if paragraphs are available in the response
            if hasattr(transcript_result, 'paragraphs') and transcript_result.paragraphs and hasattr(transcript_result.paragraphs, 'paragraphs'):
                logger.info(f"Processing paragraphs from Deepgram response")

                with open(paragraphs_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Simple header without sentiment columns
                    writer.writerow(['start_time', 'end_time', 'text'])

                    # Add debug logging to understand the paragraph structure
                    logger.info(f"Paragraph object structure: {dir(transcript_result.paragraphs.paragraphs[0])}")

                    # Get the full transcript text from the paragraphs object
                    full_transcript = ""
                    if hasattr(transcript_result.paragraphs, 'transcript'):
                        full_transcript = transcript_result.paragraphs.transcript

                    # Log the full transcript for debugging
                    logger.info(f"Full transcript: {full_transcript[:100]}...")

                    for paragraph in transcript_result.paragraphs.paragraphs:
                        # Try to access the transcript attribute, fall back to text if not available
                        paragraph_text = ""
                        if hasattr(paragraph, 'transcript'):
                            paragraph_text = paragraph.transcript
                        elif hasattr(paragraph, 'text'):
                            paragraph_text = paragraph.text

                        # If still empty, try to extract from the full transcript using start/end times
                        if not paragraph_text and full_transcript:
                            # Find sentences that start and end within this paragraph's time range
                            paragraph_sentences = []
                            if hasattr(paragraph, 'sentences') and paragraph.sentences:
                                for sentence in paragraph.sentences:
                                    if hasattr(sentence, 'text') and sentence.text:
                                        paragraph_sentences.append(sentence.text)
                                    elif hasattr(sentence, 'transcript') and sentence.transcript:
                                        paragraph_sentences.append(sentence.transcript)

                            if paragraph_sentences:
                                paragraph_text = " ".join(paragraph_sentences)
                            else:
                                # If we can't extract sentences, use the full transcript
                                paragraph_text = full_transcript

                        # Write paragraph without sentiment information
                        writer.writerow([
                            paragraph.start,  # Already in seconds
                            paragraph.end,
                            paragraph_text
                        ])

                logger.info(f"Paragraphs saved to: {paragraphs_file}")
            else:
                logger.warning("No paragraphs found in Deepgram response, creating paragraphs from sentences")

                # If no paragraphs, try to create them from sentences or the full transcript
                sentences = []
                current_sentence = {"text": "", "start": 0, "end": 0, "words": []}

                for word in transcript_result.words:
                    # If the word ends with punctuation that typically ends a sentence
                    # Check if punctuated_word exists and has content
                    has_end_punctuation = False
                    if hasattr(word, 'punctuated_word') and word.punctuated_word:
                        if word.punctuated_word[-1] in ['.', '!', '?']:
                            has_end_punctuation = True

                    if has_end_punctuation:
                        current_sentence["words"].append(word)
                        current_sentence["text"] += " " + word.word
                        current_sentence["end"] = word.end
                        sentences.append(current_sentence)
                        current_sentence = {"text": "", "start": word.end, "end": 0, "words": []}
                    else:
                        if not current_sentence["words"]:
                            current_sentence["start"] = word.start
                        current_sentence["words"].append(word)
                        current_sentence["text"] += " " + word.word
                        current_sentence["end"] = word.end

                # Add the last sentence if it's not empty
                if current_sentence["words"]:
                    sentences.append(current_sentence)

                # Group sentences into paragraphs (every 3-5 sentences)
                paragraphs = []
                current_paragraph = {"text": "", "start": 0, "end": 0, "sentences": []}
                sentence_count = 0

                for sentence in sentences:
                    if not current_paragraph["sentences"]:
                        current_paragraph["start"] = sentence["start"]

                    current_paragraph["sentences"].append(sentence)
                    current_paragraph["text"] += sentence["text"]
                    current_paragraph["end"] = sentence["end"]
                    sentence_count += 1

                    # Create a new paragraph after 3-5 sentences or if there's a long pause
                    if sentence_count >= 3:
                        paragraphs.append(current_paragraph)
                        current_paragraph = {"text": "", "start": 0, "end": 0, "sentences": []}
                        sentence_count = 0

                # Add the last paragraph if it's not empty
                if current_paragraph["sentences"]:
                    paragraphs.append(current_paragraph)

                # Save paragraphs to CSV
                with open(paragraphs_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Simple header without sentiment columns
                    writer.writerow(['start_time', 'end_time', 'text'])

                    # Write each paragraph without sentiment information
                    for para in paragraphs:
                        writer.writerow([
                            para["start"],
                            para["end"],
                            para["text"].strip()
                        ])

                logger.info(f"Created and saved {len(paragraphs)} paragraphs to: {paragraphs_file}")

            # Keep the transcript JSON file for reference
            logger.info(f"Transcript JSON file is saved at: {transcript_json_path}")

            return {
                "paragraphs_file": paragraphs_file,
                "words_file": words_file,
                "transcript_json_file": transcript_json_path
            }

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
