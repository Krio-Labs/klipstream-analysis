"""
Parakeet Transcriber Module

This module handles transcribing audio files using NVIDIA Parakeet TDT 0.6B v2 model.
This is an alternative to the Deepgram transcriber for cost optimization and offline capability.
"""

import os
import csv
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydub import AudioSegment
import librosa

from utils.config import (
    RAW_AUDIO_DIR,
    RAW_TRANSCRIPTS_DIR
)
from utils.logging_setup import setup_logger

# Try to import NeMo ASR - if not available, provide helpful error message
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    nemo_asr = None

# Set up logger
logger = setup_logger("transcriber_parakeet", "transcription_parakeet.log")

class ParakeetTranscriptionHandler:
    """Class for handling audio transcription using NVIDIA Parakeet TDT 0.6B v2"""

    def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v2", device=None):
        """Initialize the Parakeet transcription handler"""
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.model = None
        self.processor = None
        self._setup_model()

    def _get_best_device(self):
        """Determine the best available device (GPU/CPU)"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device

    def _setup_model(self):
        """Set up the Parakeet model using NeMo ASR"""
        try:
            if not NEMO_AVAILABLE:
                raise ImportError(
                    "NeMo toolkit is required for Parakeet model. "
                    "Please install it with: pip install nemo_toolkit[asr]"
                )

            logger.info(f"Loading Parakeet model: {self.model_name}")

            # Load the model using NeMo ASR
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )

            # Move model to device if GPU is available
            if self.device != "cpu" and torch.cuda.is_available():
                self.model = self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            logger.info(f"Parakeet model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {str(e)}")
            logger.error("Make sure you have installed NeMo toolkit: pip install nemo_toolkit[asr]")
            raise

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

    def _convert_audio_format(self, input_path: str, output_path: str, target_format: str = "wav") -> str:
        """Convert audio file to the required format for Parakeet model"""
        try:
            logger.info(f"Converting audio from {input_path} to {target_format.upper()} format...")

            # Load audio using pydub (handles MP3, WAV, etc.)
            audio = AudioSegment.from_file(input_path)

            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                logger.info("Converted stereo audio to mono")

            # Set sample rate to 16kHz (required for most speech models)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                logger.info(f"Resampled audio from {audio.frame_rate}Hz to 16000Hz")

            # Export to target format
            if target_format.lower() == "wav":
                audio.export(output_path, format="wav")
            elif target_format.lower() == "flac":
                audio.export(output_path, format="flac")
            else:
                raise ValueError(f"Unsupported target format: {target_format}")

            logger.info(f"Audio converted and saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to convert audio format: {str(e)}")
            raise

    def _load_and_preprocess_audio(self, audio_file_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file for the Parakeet model"""
        try:
            # Check if we need to convert the audio format
            input_path = Path(audio_file_path)
            file_extension = input_path.suffix.lower()

            # Define formats that need conversion
            formats_needing_conversion = [".mp3", ".m4a", ".aac", ".ogg"]
            compatible_formats = [".wav", ".flac"]

            # If input format needs conversion, convert to WAV
            if file_extension in formats_needing_conversion:
                # Create temporary WAV file
                temp_wav_path = input_path.parent / f"{input_path.stem}_temp_converted.wav"
                logger.info(f"Converting {file_extension} to WAV for Parakeet model compatibility")
                converted_path = self._convert_audio_format(
                    str(input_path),
                    str(temp_wav_path),
                    "wav"
                )
                audio_file_to_load = converted_path
                cleanup_temp_file = True
            elif file_extension in compatible_formats:
                logger.info(f"Audio format {file_extension} is compatible with Parakeet model")
                audio_file_to_load = audio_file_path
                cleanup_temp_file = False
            else:
                logger.warning(f"Unknown audio format {file_extension}, attempting to load directly")
                audio_file_to_load = audio_file_path
                cleanup_temp_file = False

            # Load audio using librosa (now guaranteed to be in a compatible format)
            audio, sample_rate = librosa.load(audio_file_to_load, sr=16000, mono=True)

            # Clean up temporary file if created
            if cleanup_temp_file and Path(audio_file_to_load).exists():
                Path(audio_file_to_load).unlink()
                logger.info("Cleaned up temporary converted audio file")

            logger.info(f"Loaded audio: {len(audio)/sample_rate:.2f} seconds at {sample_rate} Hz")

            return audio, sample_rate

        except Exception as e:
            logger.error(f"Failed to load audio file {audio_file_path}: {str(e)}")
            raise

    def _chunk_audio(self, audio: np.ndarray, sample_rate: int, chunk_length_s: int = 30) -> List[np.ndarray]:
        """Split audio into chunks for processing"""
        chunk_length_samples = chunk_length_s * sample_rate
        chunks = []
        
        for i in range(0, len(audio), chunk_length_samples):
            chunk = audio[i:i + chunk_length_samples]
            chunks.append(chunk)
        
        logger.info(f"Split audio into {len(chunks)} chunks of {chunk_length_s}s each")
        return chunks

    def _validate_audio_format(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Validate that audio format is compatible with Parakeet model"""
        try:
            # Check sample rate
            if sample_rate != 16000:
                logger.warning(f"Audio sample rate is {sample_rate}Hz, expected 16000Hz")
                return False

            # Check audio length
            if len(audio) == 0:
                logger.error("Audio array is empty")
                return False

            # Check for NaN or infinite values
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                logger.error("Audio contains NaN or infinite values")
                return False

            # Check audio range (should be roughly between -1 and 1)
            audio_max = np.max(np.abs(audio))
            if audio_max > 10:  # Allow some headroom for different formats
                logger.warning(f"Audio values seem unusually large (max: {audio_max})")

            logger.info(f"Audio validation passed: {len(audio)} samples at {sample_rate}Hz")
            return True

        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            return False

    def _transcribe_audio_file(self, audio_file_path: str) -> Dict:
        """Transcribe an audio file using NeMo ASR model with chunking for long files"""
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")

            # Check audio duration first
            try:
                audio_info = AudioSegment.from_file(audio_file_path)
                duration_seconds = len(audio_info) / 1000.0
                logger.info(f"Audio duration: {duration_seconds:.1f} seconds ({duration_seconds/3600:.2f} hours)")
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {e}")
                duration_seconds = 0

            # For very long files (>30 minutes), use chunking approach
            if duration_seconds > 1800:  # 30 minutes
                logger.info("Long audio detected, using chunking approach...")
                return self._transcribe_long_audio_chunked(audio_file_path, duration_seconds)

            # Use NeMo's transcribe method with timestamps for shorter files
            try:
                output = self.model.transcribe(
                    audio=[audio_file_path],
                    timestamps=True
                )
                logger.info(f"Output received: {type(output)}, length: {len(output) if output else 0}")
            except Exception as transcribe_error:
                logger.error(f"Error during transcription: {transcribe_error}")
                # Try chunking approach as fallback
                if duration_seconds > 0:
                    logger.info("Falling back to chunking approach...")
                    return self._transcribe_long_audio_chunked(audio_file_path, duration_seconds)
                return {"text": "", "words": [], "segments": []}

            if not output or len(output) == 0:
                logger.error("No transcription output received")
                return {"text": "", "words": [], "segments": []}

            # Extract the transcription result
            result = output[0]
            logger.info(f"Result type: {type(result)}")

            # Get the full text
            full_text = result.text if hasattr(result, 'text') else ""
            logger.info(f"Full text: '{full_text}'")

            # Get word-level timestamps
            words = []
            if hasattr(result, 'timestamp') and result.timestamp:
                word_timestamps = result.timestamp.get('word', [])
                logger.info(f"Found {len(word_timestamps)} word timestamps")
                for word_info in word_timestamps:
                    words.append({
                        "word": word_info.get('word', ''),
                        "start": word_info.get('start', 0.0),
                        "end": word_info.get('end', 0.0)
                    })

            # Get segment-level timestamps
            segments = []
            if hasattr(result, 'timestamp') and result.timestamp:
                segment_timestamps = result.timestamp.get('segment', [])
                logger.info(f"Found {len(segment_timestamps)} segment timestamps")
                for segment_info in segment_timestamps:
                    segments.append({
                        "text": segment_info.get('segment', ''),
                        "start": segment_info.get('start', 0.0),
                        "end": segment_info.get('end', 0.0)
                    })

            # If no segments but we have words, create segments from words
            if not segments and words:
                logger.info("Creating segments from word timestamps...")
                # Group words into segments (every ~10 words or ~10 seconds)
                current_segment = {"words": [], "start": 0.0, "end": 0.0}
                segment_word_limit = 10
                segment_duration_limit = 10.0

                for word in words:
                    if not current_segment["words"]:
                        current_segment["start"] = word["start"]

                    current_segment["words"].append(word)
                    current_segment["end"] = word["end"]

                    # Check if we should end this segment
                    segment_duration = current_segment["end"] - current_segment["start"]
                    if (len(current_segment["words"]) >= segment_word_limit or
                        segment_duration >= segment_duration_limit):

                        # Create segment text
                        segment_text = " ".join([w["word"] for w in current_segment["words"]])
                        segments.append({
                            "text": segment_text,
                            "start": current_segment["start"],
                            "end": current_segment["end"]
                        })

                        # Start new segment
                        current_segment = {"words": [], "start": 0.0, "end": 0.0}

                # Add final segment if it has words
                if current_segment["words"]:
                    segment_text = " ".join([w["word"] for w in current_segment["words"]])
                    segments.append({
                        "text": segment_text,
                        "start": current_segment["start"],
                        "end": current_segment["end"]
                    })

            logger.info(f"Transcription completed: {len(words)} words, {len(segments)} segments")

            return {
                "text": full_text,
                "words": words,
                "segments": segments
            }

        except Exception as e:
            logger.error(f"Failed to transcribe audio file: {str(e)}")
            return {"text": "", "words": [], "segments": []}

    def _transcribe_long_audio_chunked(self, audio_file_path: str, duration_seconds: float) -> Dict:
        """Transcribe long audio files by splitting into chunks with GPU optimization"""
        try:
            logger.info(f"Transcribing long audio file using chunking approach...")

            # Load audio
            audio = AudioSegment.from_file(audio_file_path)

            # Convert to mono and 16kHz for consistency
            if audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)

            # Adaptive chunk parameters based on available resources
            if torch.cuda.is_available():
                # GPU available - use larger chunks for efficiency
                chunk_duration_ms = 10 * 60 * 1000  # 10 minutes per chunk
                overlap_ms = 10 * 1000  # 10 seconds overlap
                logger.info("Using GPU-optimized chunking (10-minute chunks)")
            else:
                # CPU only - use smaller chunks to avoid memory issues
                chunk_duration_ms = 5 * 60 * 1000  # 5 minutes per chunk
                overlap_ms = 5 * 1000  # 5 seconds overlap
                logger.info("Using CPU-optimized chunking (5-minute chunks)")

            chunks = []
            chunk_start = 0
            chunk_index = 0

            logger.info(f"Splitting {duration_seconds/60:.1f} minute audio into {chunk_duration_ms/60000:.1f} minute chunks...")

            while chunk_start < len(audio):
                chunk_end = min(chunk_start + chunk_duration_ms, len(audio))
                chunk = audio[chunk_start:chunk_end]

                # Save chunk to temporary file
                temp_chunk_path = f"/tmp/chunk_{chunk_index}.wav"
                chunk.export(temp_chunk_path, format="wav")

                chunks.append({
                    "path": temp_chunk_path,
                    "start_time": chunk_start / 1000.0,  # Convert to seconds
                    "end_time": chunk_end / 1000.0,
                    "index": chunk_index
                })

                chunk_start += chunk_duration_ms - overlap_ms  # Move forward with overlap
                chunk_index += 1

            logger.info(f"Created {len(chunks)} chunks for processing")

            # Transcribe each chunk (with GPU batch processing if available)
            all_words = []
            all_segments = []
            full_text_parts = []

            # GPU batch processing for better efficiency
            if torch.cuda.is_available() and len(chunks) > 1:
                logger.info("Using GPU batch processing for chunks...")
                batch_size = min(4, len(chunks))  # Process up to 4 chunks at once

                for batch_start in range(0, len(chunks), batch_size):
                    batch_end = min(batch_start + batch_size, len(chunks))
                    batch_chunks = chunks[batch_start:batch_end]

                    logger.info(f"Processing batch {batch_start//batch_size + 1} ({len(batch_chunks)} chunks)")

                    # Prepare batch paths
                    batch_paths = [chunk["path"] for chunk in batch_chunks]

                    try:
                        # Batch transcribe
                        batch_outputs = self.model.transcribe(
                            audio=batch_paths,
                            timestamps=True
                        )

                        # Process batch results
                        for i, (chunk_info, output) in enumerate(zip(batch_chunks, batch_outputs)):
                            self._process_chunk_result(output, chunk_info, all_words, all_segments, full_text_parts)

                    except Exception as batch_error:
                        logger.error(f"Batch processing failed: {batch_error}, falling back to individual processing")
                        # Fall back to individual processing for this batch
                        for chunk_info in batch_chunks:
                            self._process_single_chunk(chunk_info, all_words, all_segments, full_text_parts)

                    finally:
                        # Clean up batch chunk files
                        for chunk_info in batch_chunks:
                            try:
                                os.remove(chunk_info["path"])
                            except:
                                pass
            else:
                # Sequential processing for CPU or single chunk
                for i, chunk_info in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} ({chunk_info['start_time']:.1f}s - {chunk_info['end_time']:.1f}s)")
                    self._process_single_chunk(chunk_info, all_words, all_segments, full_text_parts)

                    # Clean up chunk file
                    try:
                        os.remove(chunk_info["path"])
                    except:
                        pass

                    if output and len(output) > 0:
                        result = output[0]
                        chunk_text = result.text if hasattr(result, 'text') else ""

                        if chunk_text:
                            full_text_parts.append(chunk_text)

                            # Process timestamps with offset
                            if hasattr(result, 'timestamp') and result.timestamp:
                                # Word timestamps
                                word_timestamps = result.timestamp.get('word', [])
                                for word_info in word_timestamps:
                                    all_words.append({
                                        "word": word_info.get('word', ''),
                                        "start": word_info.get('start', 0.0) + chunk_info['start_time'],
                                        "end": word_info.get('end', 0.0) + chunk_info['start_time']
                                    })

                                # Segment timestamps
                                segment_timestamps = result.timestamp.get('segment', [])
                                for segment_info in segment_timestamps:
                                    all_segments.append({
                                        "text": segment_info.get('segment', ''),
                                        "start": segment_info.get('start', 0.0) + chunk_info['start_time'],
                                        "end": segment_info.get('end', 0.0) + chunk_info['start_time']
                                    })

                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {i+1}: {chunk_error}")
                    continue

                finally:
                    # Clean up temporary chunk file
                    try:
                        os.remove(chunk_info["path"])
                    except:
                        pass

            # Combine results
            full_text = " ".join(full_text_parts)

            logger.info(f"Chunked transcription completed: {len(all_words)} words, {len(all_segments)} segments")

            return {
                "text": full_text,
                "words": all_words,
                "segments": all_segments
            }

        except Exception as e:
            logger.error(f"Failed to transcribe long audio file: {str(e)}")
            return {"text": "", "words": [], "segments": []}

    def _process_chunk_result(self, output, chunk_info: Dict, all_words: List, all_segments: List, full_text_parts: List):
        """Process the result from a single chunk transcription"""
        if output and hasattr(output, 'text'):
            chunk_text = output.text
            if chunk_text:
                full_text_parts.append(chunk_text)

                # Process timestamps with offset
                if hasattr(output, 'timestamp') and output.timestamp:
                    # Word timestamps
                    word_timestamps = output.timestamp.get('word', [])
                    for word_info in word_timestamps:
                        all_words.append({
                            "word": word_info.get('word', ''),
                            "start": word_info.get('start', 0.0) + chunk_info['start_time'],
                            "end": word_info.get('end', 0.0) + chunk_info['start_time']
                        })

                    # Segment timestamps
                    segment_timestamps = output.timestamp.get('segment', [])
                    for segment_info in segment_timestamps:
                        all_segments.append({
                            "text": segment_info.get('segment', ''),
                            "start": segment_info.get('start', 0.0) + chunk_info['start_time'],
                            "end": segment_info.get('end', 0.0) + chunk_info['start_time']
                        })

    def _process_single_chunk(self, chunk_info: Dict, all_words: List, all_segments: List, full_text_parts: List):
        """Process a single chunk individually"""
        try:
            # Transcribe chunk
            output = self.model.transcribe(
                audio=[chunk_info["path"]],
                timestamps=True
            )

            if output and len(output) > 0:
                self._process_chunk_result(output[0], chunk_info, all_words, all_segments, full_text_parts)

        except Exception as chunk_error:
            logger.error(f"Error processing chunk: {chunk_error}")

    def _create_word_timestamps(self, transcription_text: str, start_time: float, duration: float) -> List[Dict]:
        """Create word-level timestamps (simplified approach)"""
        words = transcription_text.split()
        if not words:
            return []
        
        # Simple approach: distribute words evenly across the duration
        word_duration = duration / len(words)
        word_timestamps = []
        
        for i, word in enumerate(words):
            word_start = start_time + (i * word_duration)
            word_end = word_start + word_duration
            
            word_timestamps.append({
                "word": word,
                "start": word_start,
                "end": word_end
            })
        
        return word_timestamps

    def _create_paragraphs(self, word_timestamps: List[Dict], paragraph_length_s: float = 60.0) -> List[Dict]:
        """Create paragraph-level segments from word timestamps"""
        if not word_timestamps:
            return []
        
        paragraphs = []
        current_paragraph = {
            "start": word_timestamps[0]["start"],
            "end": word_timestamps[0]["end"],
            "words": []
        }
        
        for word_data in word_timestamps:
            # If we've exceeded the paragraph length, start a new paragraph
            if word_data["start"] - current_paragraph["start"] >= paragraph_length_s:
                # Finalize current paragraph
                current_paragraph["text"] = " ".join([w["word"] for w in current_paragraph["words"]])
                paragraphs.append(current_paragraph)
                
                # Start new paragraph
                current_paragraph = {
                    "start": word_data["start"],
                    "end": word_data["end"],
                    "words": []
                }
            
            current_paragraph["words"].append(word_data)
            current_paragraph["end"] = word_data["end"]
        
        # Add the last paragraph
        if current_paragraph["words"]:
            current_paragraph["text"] = " ".join([w["word"] for w in current_paragraph["words"]])
            paragraphs.append(current_paragraph)
        
        return paragraphs

    async def process_audio_files(self, video_id: str, audio_file_path: Optional[str] = None, output_dir: Optional[str] = None) -> Optional[Dict]:
        """
        Process audio files for a specific video ID using Parakeet model
        
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
                # Try different locations in order of preference (MP3 files first)
                possible_audio_paths = [
                    RAW_AUDIO_DIR / f"audio_{video_id}.mp3",
                    RAW_AUDIO_DIR / f"audio_{video_id}.wav",
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

            # Transcribe the audio file using NeMo
            logger.info("Starting transcription with NeMo ASR...")
            transcription_result = self._transcribe_audio_file(audio_file)

            if not transcription_result["text"]:
                logger.error("Transcription failed - no text output")
                return None

            full_transcription = transcription_result["text"]
            all_word_timestamps = transcription_result["words"]

            # Get audio duration for metadata
            try:
                audio_info = AudioSegment.from_file(audio_file)
                audio_duration = len(audio_info) / 1000.0  # Convert to seconds
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {e}")
                audio_duration = 0.0

            # If we don't have word timestamps, create them from the full text
            if not all_word_timestamps and full_transcription:
                logger.info("Creating word timestamps from full transcription...")
                if audio_duration > 0:
                    all_word_timestamps = self._create_word_timestamps(
                        full_transcription, 0.0, audio_duration
                    )
                else:
                    all_word_timestamps = []

            # Create paragraphs from word timestamps or segments
            if transcription_result["segments"]:
                # Use segments from NeMo if available
                paragraphs = []
                for segment in transcription_result["segments"]:
                    paragraphs.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    })
            else:
                # Create paragraphs from word timestamps
                paragraphs = self._create_paragraphs(all_word_timestamps)

            # Save raw transcript to JSON file (mimicking Deepgram format)
            transcript_json_path = output_dir / f"{base_name}_transcript.json"
            transcript_data = {
                "results": {
                    "channels": [{
                        "alternatives": [{
                            "transcript": full_transcription,
                            "words": [
                                {
                                    "word": w["word"],
                                    "start": w["start"],
                                    "end": w["end"]
                                } for w in all_word_timestamps
                            ]
                        }]
                    }]
                },
                "metadata": {
                    "model": self.model_name,
                    "duration": audio_duration,
                    "channels": 1
                }
            }
            
            with open(transcript_json_path, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            logger.info(f"Transcript JSON saved to: {transcript_json_path}")

            # Save word-level timestamps
            words_file = output_dir / f"{base_name}_words.csv"
            with open(words_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['start_time', 'end_time', 'word'])
                
                for word_data in all_word_timestamps:
                    writer.writerow([
                        word_data["start"],
                        word_data["end"],
                        word_data["word"]
                    ])
            logger.info(f"Word timestamps saved to: {words_file}")

            # Save paragraphs
            paragraphs_file = output_dir / f"{base_name}_paragraphs.csv"
            with open(paragraphs_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['start_time', 'end_time', 'text'])
                
                for para in paragraphs:
                    writer.writerow([
                        para["start"],
                        para["end"],
                        para["text"]
                    ])
            logger.info(f"Paragraphs saved to: {paragraphs_file}")

            return {
                "paragraphs_file": paragraphs_file,
                "words_file": words_file,
                "transcript_json_file": transcript_json_path
            }

        except Exception as e:
            logger.error(f"Parakeet transcription error: {str(e)}")
            raise
