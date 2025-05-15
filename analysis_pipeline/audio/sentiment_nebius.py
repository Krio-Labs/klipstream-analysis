"""
Audio Sentiment Analysis Module with Nebius API Integration

This module analyzes emotions and sentiment in audio transcriptions using Nebius-hosted LLMs.
It uses meta-llama/Meta-Llama-3.1-8B-Instruct-fast as the primary model with google/gemma-2-9b-it
and meta-llama/Llama-3.2-3B-Instruct as fallbacks. It also incorporates audio intensity and
relative loudness (energy Z-score) to improve highlight detection.
Optimized for performance with asynchronous processing and selective analysis.
"""

import pandas as pd
import time
import os
import sys
import json
import numpy as np
import asyncio
from pathlib import Path
from collections import deque
from functools import lru_cache
from dotenv import load_dotenv
from tqdm import tqdm

# Try to import optional dependencies
try:
    import psutil
except ImportError:
    os.system(f"{sys.executable} -m pip install psutil")
    import psutil

try:
    import librosa
except ImportError:
    os.system(f"{sys.executable} -m pip install librosa")
    import librosa

try:
    from openai import OpenAI
except ImportError:
    os.system(f"{sys.executable} -m pip install openai")
    from openai import OpenAI

try:
    import aiohttp
except ImportError:
    os.system(f"{sys.executable} -m pip install aiohttp")
    import aiohttp

from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("audio_sentiment_nebius", "audio_sentiment_nebius.log")

# Model configurations
MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",  # Using the fast variant
    "gemma": "google/gemma-2-9b-it",
    "llama_small": "meta-llama/Llama-3.2-3B-Instruct"  # Additional fallback option
}

# Default model to use
DEFAULT_MODEL = "llama"
FALLBACK_MODEL = "gemma"

# System prompt for sentiment analysis
SYSTEM_PROMPT = """You are a *strict* sentiment & emotion analysis engine, specialized for Twitch gaming stream transcripts.

IMPORTANT CONTEXT: In gaming streams, phrases like "bad financial decisions" and spending in-game currency (like "1,000,000 credits") are POSITIVE events that generate excitement. Swear words like "fucking" often express excitement, not anger.

When given a 60-second transcript, you must:

1. **Calibrate your baseline** so that most segments score low (<0.3) on emotion categories unless there's clear, vivid evidence.
2. Only award high emotion scores (>0.7) when the language *strongly* signals that emotion.
3. **Decide `highlight_score`** solely on whether this segment has genuine "viral potential".

**Ranges:**
- `sentiment_score`: –1.0 (very negative) to +1.0 (very positive)
- all emotions & `highlight_score`: 0.0 to 1.0

**Example analysis for a gaming stream:**
For text: "Today is a big day, bro. We are spending 1,000,000 credits on the marketplace. Look at the fucking title."
sentiment_score: 0.7 (positive because spending in-game currency is exciting)
excitement: 0.6 (excited about spending credits)
funny: 0.2 (some humor but not hilarious)
happiness: 0.5 (happy about the "big day")
anger: 0.0 (no anger despite swearing)
sadness: 0.0 (no sadness)
neutral: 0.1 (mostly emotional)
highlight_score: 0.5 (somewhat memorable moment)

**Output ONLY** the following lines (no extra text, no labels beyond these keys, do not provide any explanation):

sentiment_score:
excitement:
funny:
happiness:
anger:
sadness:
neutral:
highlight_score:"""

class RollingAudioStats:
    """
    Maintains a rolling window of audio energy values to calculate
    mean, standard deviation, and Z-scores for relative loudness.
    Uses highly optimized implementation for better performance.
    """
    def __init__(self, window_size=600):  # 600 seconds = 10 minutes
        self.window_size = window_size
        # Set a maximum size to prevent unbounded growth
        self.energy_values = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)

        # Running statistics for efficient calculation
        self.running_sum = 0.0
        self.running_sum_sq = 0.0
        self.count = 0

        # Cache for mean and std to avoid recalculation
        self.cached_mean = 0.0
        self.cached_std = 1.0
        self.cache_valid = False

        # Pre-allocate arrays for vectorized operations
        self.values_array = np.zeros(1000)
        self.array_size = 0

    def add_value(self, timestamp, energy):
        """Add a new energy value with timestamp using highly optimized algorithm"""
        # Add new value
        self.energy_values.append(energy)
        self.timestamps.append(timestamp)
        self.running_sum += energy
        self.running_sum_sq += energy * energy
        self.count += 1

        # Remove values older than window_size efficiently
        current_time = timestamp
        cutoff_time = current_time - self.window_size

        # Use efficient deque popleft() operation with batch removal
        # This is faster than removing one at a time
        to_remove = 0
        for ts in self.timestamps:
            if ts < cutoff_time:
                to_remove += 1
            else:
                break

        if to_remove > 0:
            # Remove values in batch
            removed_sum = 0
            removed_sum_sq = 0
            for _ in range(to_remove):
                old_energy = self.energy_values.popleft()
                self.timestamps.popleft()
                removed_sum += old_energy
                removed_sum_sq += old_energy * old_energy

            # Update running statistics
            self.running_sum -= removed_sum
            self.running_sum_sq -= removed_sum_sq
            self.count -= to_remove

        # Invalidate cache since values have changed
        self.cache_valid = False

    def get_stats(self):
        """Calculate mean and standard deviation using optimized algorithm"""
        # Return cached values if valid
        if self.cache_valid:
            return self.cached_mean, self.cached_std

        # Need at least a few samples
        if self.count < 5:
            return 0.0, 1.0

        # For small number of values, use running statistics
        if self.count < 100:
            # Calculate mean and variance from running sums
            mean = self.running_sum / self.count

            # Calculate variance: E[X²] - E[X]²
            variance = (self.running_sum_sq / self.count) - (mean * mean)
        else:
            # For larger datasets, use numpy's vectorized operations
            # Convert to numpy array once for vectorized operations
            values_array = np.array(list(self.energy_values))

            # Calculate statistics using numpy
            mean = np.mean(values_array)
            variance = np.var(values_array)

        # Handle numerical instability
        if variance < 1e-10:
            std = 1.0
        else:
            std = np.sqrt(variance)

        # Cache results
        self.cached_mean = mean
        self.cached_std = std
        self.cache_valid = True

        return mean, std

    def get_z_score(self, energy):
        """Calculate z-score for a given energy value"""
        mean, std = self.get_stats()
        return (energy - mean) / std

def setup_client():
    """
    Set up the OpenAI client for Nebius Studio

    Returns:
        OpenAI: Configured OpenAI client
    """
    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("NEBIUS_API_KEY environment variable must be set")

    # Create client
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key
    )

    return client

def get_optimal_workers():
    """Determine the optimal number of workers based on system resources"""
    # Get CPU count and available memory
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()  # Physical cores, fallback to logical
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB

    # Heuristic: 1 worker per 2GB of available memory, up to CPU count
    memory_based_workers = int(available_memory / 2)
    optimal_workers = min(max(1, memory_based_workers), cpu_count)

    # For API calls, we don't want to overwhelm the API
    api_call_workers = min(optimal_workers, 8)

    return api_call_workers

def get_optimal_batch_size(segment_count):
    """Determine the optimal batch size based on segment count"""
    if segment_count < 50:
        return 5  # Small datasets
    elif segment_count < 200:
        return 10  # Medium datasets
    elif segment_count < 500:
        return 20  # Large datasets
    else:
        return 30  # Very large datasets (800+ segments)

def extract_sentiment_from_text(response_text):
    """Extract sentiment values from response text"""
    # Default sentiment values
    ratings = {
        "sentiment_score": 0.0,
        "excitement": 0.0,
        "funny": 0.0,
        "happiness": 0.0,
        "anger": 0.0,
        "sadness": 0.0,
        "neutral": 0.0,
        "highlight_score": 0.0
    }

    # Try to extract values from the response line by line
    lines = response_text.split('\n')

    # Look for lines with sentiment_score, excitement, etc.
    for line in lines:
        line = line.strip()

        # Check for sentiment_score
        if line.startswith('sentiment_score:'):
            try:
                value = line.split(':')[1].strip()
                ratings["sentiment_score"] = float(value)
            except (ValueError, IndexError):
                pass

        # Check for excitement
        elif line.startswith('excitement:'):
            try:
                value = line.split(':')[1].strip()
                ratings["excitement"] = float(value)
            except (ValueError, IndexError):
                pass

        # Check for funny
        elif line.startswith('funny:'):
            try:
                value = line.split(':')[1].strip()
                ratings["funny"] = float(value)
            except (ValueError, IndexError):
                pass

        # Check for happiness
        elif line.startswith('happiness:'):
            try:
                value = line.split(':')[1].strip()
                ratings["happiness"] = float(value)
            except (ValueError, IndexError):
                pass

        # Check for anger
        elif line.startswith('anger:'):
            try:
                value = line.split(':')[1].strip()
                ratings["anger"] = float(value)
            except (ValueError, IndexError):
                pass

        # Check for sadness
        elif line.startswith('sadness:'):
            try:
                value = line.split(':')[1].strip()
                ratings["sadness"] = float(value)
            except (ValueError, IndexError):
                pass

        # Check for neutral
        elif line.startswith('neutral:'):
            try:
                value = line.split(':')[1].strip()
                ratings["neutral"] = float(value)
            except (ValueError, IndexError):
                pass

        # Check for highlight_score
        elif line.startswith('highlight_score:'):
            try:
                value = line.split(':')[1].strip()
                ratings["highlight_score"] = float(value)
            except (ValueError, IndexError):
                pass

    # Validate ratings to ensure they're in the correct range
    if ratings["sentiment_score"] < -1.0 or ratings["sentiment_score"] > 1.0:
        ratings["sentiment_score"] = max(-1.0, min(1.0, ratings["sentiment_score"]))

    for key in ["excitement", "funny", "happiness", "anger", "sadness", "neutral", "highlight_score"]:
        if ratings[key] < 0.0 or ratings[key] > 1.0:
            ratings[key] = max(0.0, min(1.0, ratings[key]))

    return ratings

# Create a simple cache for sentiment analysis
sentiment_cache = {}

async def analyze_sentiment_async(session, text, api_key, model_key=DEFAULT_MODEL, timeout=30, speech_rate=None, abs_intensity=None, rel_intensity=None):
    """
    Analyze sentiment using Nebius API asynchronously

    Args:
        session (aiohttp.ClientSession): HTTP session for making requests
        text (str): The text to analyze
        api_key (str): Nebius API key
        model_key (str): Model key to use (llama or gemma)
        timeout (int): Request timeout in seconds
        speech_rate (float, optional): Speech rate in words per second
        abs_intensity (float, optional): Absolute loudness (0-1 scale)
        rel_intensity (float, optional): Relative loudness (0-1 scale)

    Returns:
        dict: Dictionary containing sentiment scores
    """
    # Handle empty or very short texts
    if not text or not text.strip() or len(text.split()) < 2:
        return {
            "sentiment_score": 0.0,
            "excitement": 0.0,
            "funny": 0.0,
            "happiness": 0.0,
            "anger": 0.0,
            "sadness": 0.0,
            "neutral": 1.0,
            "highlight_score": 0.0,
            "inference_time": 0.0
        }

    # Create a cache key that includes audio metrics if provided
    if speech_rate is not None and abs_intensity is not None and rel_intensity is not None:
        # Include rounded audio metrics in the cache key
        cache_key = f"{model_key}_{round(speech_rate, 1)}_{round(abs_intensity, 1)}_{round(rel_intensity, 1)}_{' '.join(text.lower().split())[:100]}"
    else:
        cache_key = f"{model_key}_{' '.join(text.lower().split())[:100]}"  # First 100 chars after normalization

    # Check cache
    if cache_key in sentiment_cache:
        logger.info(f"[{hash(text) % 10000:04d}] Cache hit for text: {text[:50]}...")
        return sentiment_cache[cache_key]

    start_time = time.time()
    request_id = f"{model_key}-{hash(text) % 10000:04d}"

    try:
        model_name = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
        logger.debug(f"[{request_id}] Starting request to {model_name}")

        # Create user prompt with audio metrics if available
        if speech_rate is not None and abs_intensity is not None and rel_intensity is not None:
            # Format the metrics to 2 decimal places for readability
            user_prompt = f"""Analyze the following transcript segment from a Twitch stream and provide scores for sentiment and emotions.

Speech Rate: {speech_rate:.2f} words per second
Absolute Loudness: {abs_intensity:.2f}
Relative Loudness: {rel_intensity:.2f}

Transcript:
{text}"""
        else:
            # Use the original prompt without audio metrics
            user_prompt = f"Analyze the following transcript segment from a Twitch stream and provide scores for sentiment and emotions.\n\n{text}"

        payload = {
            "model": model_name,
            "max_tokens": 512,
            "temperature": 0.6,
            "top_p": 0.9,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Set a timeout for the request
        try:
            # Use timeout parameter in session.post instead of asyncio.timeout
            logger.info(f"[{request_id}] Sending request to API for text: {text[:50]}...")
            async with session.post(
                "https://api.studio.nebius.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout  # Use timeout parameter here
            ) as response:
                logger.info(f"[{request_id}] Received response with status {response.status}")

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")

                logger.debug(f"[{request_id}] Parsing JSON response...")
                result = await response.json()
                response_text = result["choices"][0]["message"]["content"]

                # Extract sentiment values
                logger.debug(f"[{request_id}] Extracting sentiment values...")
                sentiment = extract_sentiment_from_text(response_text)
                sentiment["inference_time"] = time.time() - start_time
                sentiment["model"] = model_name

                # Cache the result
                sentiment_cache[cache_key] = sentiment

                logger.debug(f"[{request_id}] Successfully processed request in {sentiment['inference_time']:.2f}s")
                return sentiment
        except asyncio.TimeoutError:
            logger.warning(f"[{request_id}] Request timed out after {timeout} seconds")
            raise Exception(f"Request timed out after {timeout} seconds")
        except aiohttp.ClientError as e:
            logger.warning(f"[{request_id}] Client error: {str(e)}")
            raise Exception(f"Client error: {str(e)}")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Error after {elapsed:.2f}s: {str(e)}")

        # Try fallback model if primary model fails
        if model_key == DEFAULT_MODEL:
            logger.info(f"[{request_id}] Trying fallback model {FALLBACK_MODEL}...")
            try:
                # Create a new session for the fallback request
                async with aiohttp.ClientSession() as fallback_session:
                    fallback_result = await analyze_sentiment_async(
                        fallback_session, text, api_key, FALLBACK_MODEL, timeout
                    )
                    return fallback_result
            except Exception as fallback_error:
                logger.error(f"[{request_id}] Fallback model also failed: {str(fallback_error)}")

                # Try small Llama model as a second fallback
                logger.info(f"[{request_id}] Trying small Llama model as second fallback...")
                try:
                    # Create a new session for the second fallback request
                    async with aiohttp.ClientSession() as small_model_session:
                        small_model_result = await analyze_sentiment_async(
                            small_model_session, text, api_key, "llama_small", timeout
                        )
                        return small_model_result
                except Exception as small_model_error:
                    logger.error(f"[{request_id}] Small Llama model also failed: {str(small_model_error)}")

        # Return default sentiment with inference time
        logger.warning(f"[{request_id}] All models failed. Using default sentiment scores.")
        return {
            "sentiment_score": 0.5,  # Neutral sentiment
            "excitement": 0.2,
            "funny": 0.1,
            "happiness": 0.2,
            "anger": 0.1,
            "sadness": 0.1,
            "neutral": 0.7,
            "highlight_score": 0.2,
            "error": str(e),
            "inference_time": elapsed,
            "model": "default_fallback"
        }

def analyze_sentiment_sync(text, client, model_key=DEFAULT_MODEL, timeout=30, speech_rate=None, abs_intensity=None, rel_intensity=None):
    """
    Synchronous wrapper for analyze_sentiment_async

    Args:
        text (str): The text to analyze
        client (OpenAI): OpenAI client for Nebius
        model_key (str): Model key to use (llama or gemma)
        timeout (int): Request timeout in seconds
        speech_rate (float, optional): Speech rate in words per second
        abs_intensity (float, optional): Absolute loudness (0-1 scale)
        rel_intensity (float, optional): Relative loudness (0-1 scale)

    Returns:
        dict: Dictionary containing sentiment scores
    """
    # Handle empty or very short texts
    if not text or not text.strip() or len(text.split()) < 2:
        return {
            "sentiment_score": 0.0,
            "excitement": 0.0,
            "funny": 0.0,
            "happiness": 0.0,
            "anger": 0.0,
            "sadness": 0.0,
            "neutral": 1.0,
            "highlight_score": 0.0,
            "inference_time": 0.0
        }

    # Create a cache key that includes audio metrics if provided
    if speech_rate is not None and abs_intensity is not None and rel_intensity is not None:
        # Include rounded audio metrics in the cache key
        cache_key = f"{model_key}_{round(speech_rate, 1)}_{round(abs_intensity, 1)}_{round(rel_intensity, 1)}_{' '.join(text.lower().split())[:100]}"
    else:
        cache_key = f"{model_key}_{' '.join(text.lower().split())[:100]}"  # First 100 chars after normalization

    # Check cache
    if cache_key in sentiment_cache:
        return sentiment_cache[cache_key]

    start_time = time.time()

    try:
        model_name = MODELS.get(model_key, MODELS[DEFAULT_MODEL])

        # Create user prompt with audio metrics if available
        if speech_rate is not None and abs_intensity is not None and rel_intensity is not None:
            # Format the metrics to 2 decimal places for readability
            user_prompt = f"""Analyze the following transcript segment from a Twitch stream and provide scores for sentiment and emotions.

Speech Rate: {speech_rate:.2f} words per second
Absolute Loudness: {abs_intensity:.2f}
Relative Loudness: {rel_intensity:.2f}

Transcript:
{text}"""
        else:
            # Use the original prompt without audio metrics
            user_prompt = f"Analyze this Twitch gaming stream transcript:\n\n{text}"

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=512,
            temperature=0.6,
            top_p=0.9,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract the response content
        response_text = response.choices[0].message.content

        # Extract sentiment values
        sentiment = extract_sentiment_from_text(response_text)
        sentiment["inference_time"] = time.time() - start_time
        sentiment["model"] = model_name

        # Cache the result
        sentiment_cache[cache_key] = sentiment

        return sentiment

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error in sentiment analysis: {str(e)}")

        # Try fallback model if primary model fails
        if model_key == DEFAULT_MODEL:
            logger.info(f"Trying fallback model {FALLBACK_MODEL}...")
            try:
                fallback_result = analyze_sentiment_sync(
                    text, client, FALLBACK_MODEL, timeout,
                    speech_rate=speech_rate,
                    abs_intensity=abs_intensity,
                    rel_intensity=rel_intensity
                )
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {str(fallback_error)}")

        # Return default sentiment with inference time
        return {
            "sentiment_score": 0.0,
            "excitement": 0.0,
            "funny": 0.0,
            "happiness": 0.0,
            "anger": 0.0,
            "sadness": 0.0,
            "neutral": 1.0,
            "highlight_score": 0.0,
            "error": str(e),
            "inference_time": elapsed,
            "model": MODELS.get(model_key, MODELS[DEFAULT_MODEL])
        }

def extract_audio_features(audio_path):
    """
    Extract audio features for the entire audio file once with optimized parameters.

    Args:
        audio_path (str): Path to the audio file

    Returns:
        tuple: (audio_data, sample_rate, rms_energy, spectral_centroid, timestamps)
    """
    try:
        # Check if audio file exists
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return None, None, None, None, None

        # Load audio file with mono channel if stereo
        logger.info(f"Loading audio file: {audio_path}")
        y, sr = librosa.load(audio_path, mono=True)

        # Use slightly larger frame and hop sizes for faster processing
        # but still maintain good resolution
        frame_length = int(sr * 0.04)  # 40ms frames (was 30ms)
        hop_length = int(sr * 0.015)   # 15ms hop (was 10ms)

        # Calculate RMS energy for the entire file
        logger.info("Calculating RMS energy for entire audio file")
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Calculate spectral centroid for the entire file
        logger.info("Calculating spectral centroid for entire audio file")
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr,
            n_fft=frame_length,
            hop_length=hop_length
        )[0]

        # Calculate timestamps for each frame
        timestamps = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        return y, sr, rms, spectral_centroid, timestamps
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        return None, None, None, None, None

def calculate_speech_rate(text, start_time, end_time):
    """
    Calculate speech rate (words per second) for a given text segment.

    Args:
        text (str): The text to analyze
        start_time (float): Start time in seconds
        end_time (float): End time in seconds

    Returns:
        float: Words per second
    """
    try:
        # Handle edge cases
        if not isinstance(text, str):
            return 0.0

        if not text or not text.strip():
            return 0.0

        # Ensure start_time and end_time are floats
        start_time = float(start_time) if start_time is not None else 0.0
        end_time = float(end_time) if end_time is not None else 0.0

        if end_time <= start_time:
            return 0.0

        # Count words (simple split by whitespace)
        word_count = len(text.split())

        # Calculate duration in seconds
        duration = end_time - start_time

        # Calculate words per second
        words_per_second = word_count / duration if duration > 0 else 0.0

        return words_per_second
    except Exception as e:
        logger.error(f"Error calculating speech rate: {str(e)}")
        return 0.0

def extract_audio_intensity(audio_data, start_time, end_time, rolling_stats=None):
    """
    Extract audio intensity (loudness) for a specific time range with z-score normalization.

    Args:
        audio_data (tuple): Tuple containing (audio_array, sample_rate, rms, spectral_centroid, timestamps)
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        rolling_stats (RollingAudioStats): Rolling statistics object for Z-score calculation

    Returns:
        tuple: (absolute_intensity, relative_intensity_z_score)
    """
    try:
        # Check if audio_data is None
        if audio_data is None:
            return 0.0, 0.0

        # Unpack audio data
        y, sr, rms_full, spectral_centroid_full, timestamps = audio_data

        # If audio data is not available, return default values
        if y is None or sr is None:
            return 0.0, 0.0

        # Find indices in the timestamps array that correspond to our time range
        start_idx = np.searchsorted(timestamps, start_time)
        end_idx = np.searchsorted(timestamps, end_time)

        # Ensure we have at least one frame
        if start_idx >= end_idx or start_idx >= len(timestamps):
            return 0.0, 0.0

        # Extract the RMS and spectral centroid values for this time range
        rms = rms_full[start_idx:end_idx]
        spectral_centroid = spectral_centroid_full[start_idx:end_idx]

        # Calculate raw energy (not normalized)
        raw_energy = np.mean(rms) if len(rms) > 0 else 0.0

        # Normalize features for absolute intensity
        if len(rms) > 0:
            # Use global min/max for more consistent normalization
            rms_min = np.min(rms_full)
            rms_max = np.max(rms_full)
            rms_norm = (rms - rms_min) / (rms_max - rms_min + 1e-10)
            rms_mean = np.mean(rms_norm)
        else:
            rms_mean = 0.0

        if len(spectral_centroid) > 0:
            # Use global min/max for more consistent normalization
            sc_min = np.min(spectral_centroid_full)
            sc_max = np.max(spectral_centroid_full)
            centroid_norm = (spectral_centroid - sc_min) / (sc_max - sc_min + 1e-10)
            centroid_mean = np.mean(centroid_norm)
        else:
            centroid_mean = 0.0

        # Combine features for absolute intensity (weighted)
        absolute_intensity = (rms_mean * 0.7) + (centroid_mean * 0.3)

        # Calculate relative intensity (z-score) if rolling stats provided
        relative_intensity = 0.0
        if rolling_stats is not None:
            # Add this value to rolling stats
            mid_time = (start_time + end_time) / 2
            rolling_stats.add_value(mid_time, raw_energy)

            # Get z-score
            z_score = rolling_stats.get_z_score(raw_energy)

            # Cap extreme values
            z_score = max(-3.0, min(3.0, z_score))

            # Convert to 0-1 range for easier integration
            # Z-score of -3 maps to 0.0, Z-score of +3 maps to 1.0
            relative_intensity = (z_score + 3.0) / 6.0

        return float(absolute_intensity), float(relative_intensity)
    except Exception as e:
        logger.error(f"Error extracting audio intensity: {str(e)}")
        return 0.0, 0.0

async def process_segment_async(segment, audio_data, rolling_stats, api_key, model_key=DEFAULT_MODEL, timeout=30):
    """
    Process a single segment asynchronously

    Args:
        segment (dict): Segment data with text, start_time, end_time
        audio_data (tuple): Audio data tuple
        rolling_stats (RollingAudioStats): Rolling statistics object
        api_key (str): Nebius API key
        model_key (str): Model key to use
        timeout (int): Request timeout in seconds

    Returns:
        dict: Processed segment with sentiment and audio features
    """
    try:
        # Extract segment data
        text = segment.get('text', '')
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', 0)

        # Skip empty segments
        if not isinstance(text, str) or not text.strip():
            return None

        # Create a new session for this request
        async with aiohttp.ClientSession() as session:
            # Calculate speech rate
            speech_rate = calculate_speech_rate(text, start_time, end_time)

            # Extract audio intensity
            abs_intensity, rel_intensity = extract_audio_intensity(
                audio_data, start_time, end_time, rolling_stats
            )

            # Analyze sentiment with audio metrics
            sentiment = await analyze_sentiment_async(
                session, text, api_key, model_key, timeout,
                speech_rate=speech_rate,
                abs_intensity=abs_intensity,
                rel_intensity=rel_intensity
            )

            # Calculate highlight score
            # Updated weights: Model score (50%), Speech rate (15%),
            # Relative loudness (15%), Absolute loudness (10%), Emotional intensity (10%)

            # Normalize speech rate (assuming normal speech is 2-3 words per second)
            # Cap at 6 wps to avoid extreme values
            speech_rate_norm = min(speech_rate / 6.0, 1.0)

            # Get average emotion score (excluding neutral)
            emotion_score = (
                sentiment.get('excitement', 0.0) +
                sentiment.get('funny', 0.0) +
                sentiment.get('happiness', 0.0) +
                sentiment.get('anger', 0.0) +
                sentiment.get('sadness', 0.0)
            ) / 5.0

            # Calculate base highlight score (without model score)
            base_highlight_score = (
                (speech_rate_norm * 0.15) +
                (rel_intensity * 0.15) +
                (abs_intensity * 0.10) +
                (emotion_score * 0.10)
            )

            # Scale the base score to account for the components' weights
            # This ensures the base score still contributes meaningfully
            # when combined with the model score
            base_highlight_score = base_highlight_score * (1.0 / 0.5)

            # If model already provided a highlight score, blend them
            if 'highlight_score' in sentiment and sentiment['highlight_score'] > 0:
                model_highlight = sentiment['highlight_score']
                # Blend 50% model score, 50% calculated score
                highlight_score = (model_highlight * 0.5) + (base_highlight_score * 0.5)
            else:
                # If no model score, use the base score
                highlight_score = base_highlight_score

            # Create result dictionary
            result = {
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'sentiment_score': sentiment.get('sentiment_score', 0.0),
                'excitement': sentiment.get('excitement', 0.0),
                'funny': sentiment.get('funny', 0.0),
                'happiness': sentiment.get('happiness', 0.0),
                'anger': sentiment.get('anger', 0.0),
                'sadness': sentiment.get('sadness', 0.0),
                'neutral': sentiment.get('neutral', 0.0),
                'highlight_score': highlight_score,
                'speech_rate': speech_rate,
                'absolute_intensity': abs_intensity,
                'relative_intensity': rel_intensity,
                'model': sentiment.get('model', MODELS[model_key]),
                'inference_time': sentiment.get('inference_time', 0.0)
            }

            return result
    except Exception as e:
        logger.error(f"Error processing segment: {str(e)}")
        return None

async def process_batch_async(batch, audio_data, rolling_stats, api_key, model_key=DEFAULT_MODEL, timeout=30):
    """
    Process a batch of segments asynchronously

    Args:
        batch (list): List of segment dictionaries
        audio_data (tuple): Audio data tuple
        rolling_stats (RollingAudioStats): Rolling statistics object
        api_key (str): Nebius API key
        model_key (str): Model key to use
        timeout (int): Request timeout in seconds

    Returns:
        list: List of processed segments
    """
    tasks = []
    for segment in batch:
        task = process_segment_async(
            segment, audio_data, rolling_stats, api_key, model_key, timeout
        )
        tasks.append(task)

    # Process all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Filter out None results
    return [r for r in results if r is not None]

async def analyze_audio_sentiment_async(
    video_id,
    input_file=None,
    output_dir=None,
    audio_file=None,
    use_nebius=True,
    model_key=DEFAULT_MODEL,
    max_concurrent=None,
    timeout=30
):
    """
    Analyze audio sentiment asynchronously using Nebius API

    Args:
        video_id (str): Video ID
        input_file (str): Path to input segments CSV file
        output_dir (str): Path to output directory
        audio_file (str): Path to audio file
        use_nebius (bool): Whether to use Nebius API (True) or local models (False)
        model_key (str): Model key to use (llama or gemma)
        max_concurrent (int): Maximum number of concurrent requests
        timeout (int): Request timeout in seconds

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set up paths
        if output_dir is None:
            output_dir = 'output/Analysis/Audio'
        os.makedirs(output_dir, exist_ok=True)

        # Determine input file path
        if input_file is None:
            input_file = f'output/Raw/Transcripts/audio_{video_id}_segments.csv'

        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False

        # Determine audio file path
        if audio_file is None:
            # Try multiple possible paths with different case variations
            possible_paths = [
                f'output/Raw/audio/audio_{video_id}.wav',  # lowercase 'audio'
                f'output/Raw/Audio/audio_{video_id}.wav',  # uppercase 'Audio'
                f'output/Raw/audio_{video_id}.wav',        # directly in Raw
                f'output/Raw/audio_{video_id}.mp3'         # mp3 format
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    audio_file = path
                    logger.info(f"Found audio file at: {audio_file}")
                    break

            if audio_file is None:
                logger.warning(f"Audio file not found in any of the expected locations")
                audio_file = None

        # Load segments
        logger.info(f"Loading segments from {input_file}")
        df = pd.read_csv(input_file)
        total_segments = len(df)
        logger.info(f"Loaded {total_segments} segments")

        # Extract audio features if audio file is available
        audio_data = None
        if audio_file:
            logger.info(f"Extracting audio features from {audio_file}")
            audio_data = extract_audio_features(audio_file)

        # Create rolling stats object for relative loudness calculation
        rolling_stats = RollingAudioStats(window_size=600)  # 10-minute window

        # Determine optimal concurrency if not specified
        if max_concurrent is None:
            max_concurrent = get_optimal_workers()
        logger.info(f"Using {max_concurrent} concurrent workers")

        # Determine optimal batch size
        batch_size = get_optimal_batch_size(total_segments)
        logger.info(f"Using batch size of {batch_size}")

        # Prepare segments for processing
        segments = []
        for _, row in df.iterrows():
            segments.append({
                'text': row['text'],
                'start_time': row['start_time'],
                'end_time': row['end_time']
            })

        # Split segments into batches
        batches = []
        for i in range(0, len(segments), batch_size):
            batches.append(segments[i:i+batch_size])

        # Get API key
        load_dotenv()
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key and use_nebius:
            logger.error("NEBIUS_API_KEY environment variable must be set")
            return False

        # Process batches with progress tracking
        logger.info(f"Processing {len(batches)} batches with {max_concurrent} concurrent workers")
        results = []

        with tqdm(total=total_segments, desc=f"Processing segments", unit="segment") as progress:
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_batch_with_semaphore(batch_idx):
                async with semaphore:
                    batch = batches[batch_idx]
                    batch_results = await process_batch_async(
                        batch, audio_data, rolling_stats, api_key, model_key, timeout
                    )
                    progress.update(len(batch))
                    return batch_results

            # Create tasks for all batches
            tasks = [process_batch_with_semaphore(i) for i in range(len(batches))]

            # Process all batches
            batch_results = await asyncio.gather(*tasks)

            # Combine results
            for batch_result in batch_results:
                results.extend(batch_result)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        output_file = f'{output_dir}/audio_{video_id}_sentiment.csv'

        # Rearrange columns to have start_time and end_time first, scores in the middle, and text at the end
        columns = list(results_df.columns)

        # Define the desired column order
        time_columns = ['start_time', 'end_time']
        text_columns = ['text']

        # Get all other columns (scores, metrics, etc.)
        score_columns = [col for col in columns if col not in time_columns + text_columns]

        # Create the new column order
        new_column_order = time_columns + score_columns + text_columns

        # Reorder the columns
        results_df = results_df[new_column_order]

        logger.info(f"Saving results to {output_file} with columns rearranged for better readability")
        results_df.to_csv(output_file, index=False)

        # Also save as JSON for easier inspection
        json_file = output_file.replace('.csv', '.json')
        with open(json_file, 'w') as f:
            json.dump(results_df.to_dict(orient='records'), f, indent=2)

        logger.info(f"Analysis complete. Processed {len(results)} segments.")
        return True

    except Exception as e:
        logger.error(f"Error in analyze_audio_sentiment_async: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def analyze_audio_sentiment(
    video_id,
    input_file=None,
    output_dir=None,
    audio_file=None,
    use_nebius=True,
    model_key=DEFAULT_MODEL,
    max_concurrent=None,
    timeout=30
):
    """
    Wrapper function to run the async sentiment analysis

    Args:
        video_id (str): Video ID
        input_file (str): Path to input segments CSV file
        output_dir (str): Path to output directory
        audio_file (str): Path to audio file
        use_nebius (bool): Whether to use Nebius API (True) or local models (False)
        model_key (str): Model key to use (llama or gemma)
        max_concurrent (int): Maximum number of concurrent requests
        timeout (int): Request timeout in seconds

    Returns:
        bool: True if successful, False otherwise
    """
    return asyncio.run(analyze_audio_sentiment_async(
        video_id=video_id,
        input_file=input_file,
        output_dir=output_dir,
        audio_file=audio_file,
        use_nebius=use_nebius,
        model_key=model_key,
        max_concurrent=max_concurrent,
        timeout=timeout
    ))

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze audio sentiment using Nebius API")
    parser.add_argument("--video-id", required=True, help="Video ID to process")
    parser.add_argument("--input-file", help="Path to input segments CSV file")
    parser.add_argument("--output-dir", help="Path to output directory")
    parser.add_argument("--audio-file", help="Path to audio file")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()),
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--concurrency", type=int, help="Maximum number of concurrent requests")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds (default: 30)")
    parser.add_argument("--no-nebius", action="store_true", help="Don't use Nebius API (use local models)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)
        # Add a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Print configuration
    logger.info(f"System: {os.name}")
    logger.info(f"Python: {sys.version.split()[0]}")

    # Check if aiohttp is installed
    try:
        logger.info(f"aiohttp version: {aiohttp.__version__}")
    except (ImportError, AttributeError):
        logger.info("aiohttp version information not available")

    logger.info("Configuration:")
    logger.info(f"  Video ID: {args.video_id}")
    logger.info(f"  Input file: {args.input_file or 'auto'}")
    logger.info(f"  Output directory: {args.output_dir or 'auto'}")
    logger.info(f"  Audio file: {args.audio_file or 'auto'}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Max concurrency: {args.concurrency or 'auto'}")
    logger.info(f"  Request timeout: {args.timeout}s")
    logger.info(f"  Use Nebius API: {not args.no_nebius}")
    logger.info(f"  Debug mode: {'Enabled' if args.debug else 'Disabled'}")

    # Run the analysis
    success = analyze_audio_sentiment(
        video_id=args.video_id,
        input_file=args.input_file,
        output_dir=args.output_dir,
        audio_file=args.audio_file,
        use_nebius=not args.no_nebius,
        model_key=args.model,
        max_concurrent=args.concurrency,
        timeout=args.timeout
    )

    # Exit with appropriate status code
    sys.exit(0 if success else 1)
