"""
Audio Sentiment Analysis Module

This module analyzes emotions and sentiment in audio transcriptions.
It uses modernbert-large-go-emotions for emotion detection and RoBERTa for sentiment analysis.
It also incorporates audio intensity and relative loudness (energy Z-score) to improve highlight detection.
Optimized for performance with parallel processing and selective analysis.
"""

import pandas as pd
import time
import os
import json
import numpy as np
from pathlib import Path
from collections import deque
import concurrent.futures
from functools import lru_cache
from dotenv import load_dotenv
import librosa
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils.logging_setup import setup_logger


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

# Set up logger
logger = setup_logger("audio_sentiment", "audio_sentiment.log")

# Mapping of Go Emotions to our standard categories
# The 28 emotion categories from cirimus/modernbert-large-go-emotions:
# admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire,
# disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief,
# joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
EMOTION_MAPPING = {
    # Map to 'excitement'
    'excitement': 'excitement',
    'surprise': 'excitement',
    'curiosity': 'excitement',
    'realization': 'excitement',
    'optimism': 'excitement',
    'pride': 'excitement',
    'desire': 'excitement',

    # Map to 'funny'
    'amusement': 'funny',

    # Map to 'happiness'
    'joy': 'happiness',
    'admiration': 'happiness',
    'approval': 'happiness',
    'caring': 'happiness',
    'gratitude': 'happiness',
    'love': 'happiness',
    'relief': 'happiness',

    # Map to 'anger'
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',
    'disgust': 'anger',

    # Map to 'sadness'
    'sadness': 'sadness',
    'disappointment': 'sadness',
    'embarrassment': 'sadness',
    'fear': 'sadness',
    'grief': 'sadness',
    'nervousness': 'sadness',
    'remorse': 'sadness',

    # Keep neutral as is
    'neutral': 'neutral',

    # Add confusion to excitement as it can be engaging
    'confusion': 'excitement'
}

# Global variables for models
sentiment_model = None
sentiment_tokenizer = None
emotion_model = None
emotion_tokenizer = None

# Initialize the models
def optimize_model_memory():
    """Configure PyTorch for memory-efficient inference"""
    # Use memory-efficient operations where possible
    torch.set_grad_enabled(False)  # Ensure gradients are disabled

    if torch.cuda.is_available():
        # Set to deterministic mode for potentially better memory usage
        torch.backends.cudnn.deterministic = True

        # Benchmark mode can select the most efficient algorithms
        # but may use more memory
        torch.backends.cudnn.benchmark = False

def initialize_models():
    """Initialize the sentiment and emotion models with memory optimizations"""
    global sentiment_model, sentiment_tokenizer, emotion_model, emotion_tokenizer

    try:
        logger.info("Initializing models...")

        # Apply memory optimizations
        optimize_model_memory()

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load the sentiment model (RoBERTa)
        logger.info("Loading RoBERTa sentiment model...")
        sentiment_model_name = "siebert/sentiment-roberta-large-english"

        # Load sentiment tokenizer and model
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        sentiment_model.to(device)
        sentiment_model.eval()  # Set to evaluation mode

        # Load the emotion model (ModernBERT)
        logger.info("Loading ModernBERT emotion model...")
        emotion_model_name = "cirimus/modernbert-large-go-emotions"

        # Load emotion tokenizer and model
        emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
        emotion_model.to(device)
        emotion_model.eval()  # Set to evaluation mode

        # Clear any cached memory after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Successfully initialized all models")
        return True
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        return False

# Create a simple cache for emotion and sentiment analysis
emotion_cache = {}
sentiment_cache = {}

def analyze_emotions(text, model=None):
    """
    Analyze emotions in the given text using ModernBERT Go Emotions model.

    Args:
        text (str): The text to analyze
        model: Not used, kept for compatibility

    Returns:
        dict: A dictionary containing detected emotions
    """
    global emotion_model, emotion_tokenizer

    # Handle empty or very short texts
    if not text or not text.strip() or len(text.split()) < 2:
        return {'llm_response': {'emotions': ['neutral']}, 'confidence_score': 0.7}

    if emotion_model is None or emotion_tokenizer is None:
        logger.warning("ModernBERT emotion model not available, returning neutral emotion")
        return {'llm_response': {'emotions': ['neutral']}}

    # Check cache first (use a simplified version of the text as key)
    # This helps with minor variations in whitespace, etc.
    cache_key = ' '.join(text.lower().split())[:100]  # First 100 chars after normalization
    if cache_key in emotion_cache:
        return emotion_cache[cache_key]

    try:
        # Truncate text if it's too long (BERT has a token limit)
        if len(text) > 512:
            text = text[:512]

        # Prepare for inference
        inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Move inputs to the same device as the model
        device = next(emotion_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = emotion_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get the top 3 emotions
        top_scores, top_indices = torch.topk(scores, 3, dim=1)

        # The 28 emotion categories from cirimus/modernbert-large-go-emotions
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
            'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        # Get the top emotions and their scores
        top_emotions = [emotion_labels[idx] for idx in top_indices[0].tolist()]
        top_scores = top_scores[0].tolist()

        # Create response
        response = {
            'llm_response': {
                'emotions': top_emotions
            },
            'confidence_score': top_scores[0]  # Confidence of the top emotion
        }

        # Cache the result
        emotion_cache[cache_key] = response

        return response
    except Exception as e:
        logger.error(f"Error analyzing emotions with ModernBERT: {str(e)}")
        return {'llm_response': {'emotions': ['neutral']}}

def analyze_sentiment(text, model=None):
    """
    Analyze sentiment in the given text using RoBERTa sentiment model.
    Uses caching to avoid repeated analysis of the same text.

    Args:
        text (str): The text to analyze
        model: Not used, kept for compatibility

    Returns:
        dict: A dictionary containing detected sentiment
    """
    global sentiment_model, sentiment_tokenizer

    # Handle empty or very short texts
    if not text or not text.strip() or len(text.split()) < 2:
        return {'llm_response': {'sentiment': ['neutral']}, 'confidence_score': 0.7}

    if sentiment_model is None or sentiment_tokenizer is None:
        logger.warning("RoBERTa sentiment model not available, returning neutral sentiment")
        return {'llm_response': {'sentiment': ['neutral']}}

    # Check cache first (use a simplified version of the text as key)
    cache_key = ' '.join(text.lower().split())[:100]  # First 100 chars after normalization
    if cache_key in sentiment_cache:
        return sentiment_cache[cache_key]

    try:
        # Truncate text if it's too long (RoBERTa has a token limit)
        if len(text) > 512:
            text = text[:512]

        # Prepare for inference
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Move inputs to the same device as the model
        device = next(sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get prediction
        # RoBERTa model outputs: [negative, neutral, positive]
        prediction_idx = scores.argmax().item()
        confidence = scores[0][prediction_idx].item()

        # Map to sentiment labels
        sentiment_labels = ['negative', 'neutral', 'positive']
        sentiment = sentiment_labels[prediction_idx]

        # Create response in the same format as llmware for compatibility
        response = {
            'llm_response': {
                'sentiment': [sentiment]
            },
            'confidence_score': confidence
        }

        # Cache the result
        sentiment_cache[cache_key] = response

        return response
    except Exception as e:
        logger.error(f"Error analyzing sentiment with RoBERTa: {str(e)}")
        return {'llm_response': {'sentiment': ['neutral']}}

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

def generate_scores_with_precomputed_results(text, start_time, end_time, audio_intensities, emotion_result, sentiment_result, speech_rate=None):
    """
    Generate scores using pre-computed emotion and sentiment results

    Args:
        text (str): The text to analyze
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        audio_intensities (tuple): Tuple of (absolute_intensity, relative_intensity_z_score)
        emotion_result (dict): Pre-computed emotion analysis result
        sentiment_result (dict): Pre-computed sentiment analysis result
        speech_rate (float, optional): Normalized speech rate (words per second relative to average)

    Returns:
        dict: Dictionary containing all scores
    """
    # Extract absolute and relative audio intensities
    absolute_intensity, relative_intensity = audio_intensities

    # Initialize scores dictionary with default values
    scores = {
        'excitement': 0.0,
        'funny': 0.0,
        'happiness': 0.0,
        'anger': 0.0,
        'sadness': 0.0,
        'neutral': 1.0,  # Default to neutral
        'sentiment_score': 0.0,
        'highlight_score': 0.0,
        'audio_intensity': absolute_intensity,  # Store the absolute audio intensity
        'audio_intensity_relative': relative_intensity,  # Store the relative audio intensity (z-score)
        'text': text,
        'start_time': start_time,
        'end_time': end_time
    }

    try:
        # Process emotion results
        if emotion_result and 'emotions' in emotion_result:
            top_emotions = emotion_result['emotions']
            emotion_confidence = emotion_result.get('confidence', 0.8)

            # Process primary emotion
            if top_emotions:
                primary_emotion = top_emotions[0]

                # Map to our standard emotion categories
                if primary_emotion not in EMOTION_MAPPING:
                    # Default to neutral if not in mapping
                    mapped_emotion = 'neutral'
                else:
                    mapped_emotion = EMOTION_MAPPING[primary_emotion]

                # Set the corresponding emotion score
                scores[mapped_emotion] = emotion_confidence

                # Process secondary emotions if available
                if len(top_emotions) > 1:
                    # Process second emotion with reduced weight
                    secondary_emotion = top_emotions[1]
                    secondary_mapped = EMOTION_MAPPING.get(secondary_emotion, 'neutral')
                    secondary_confidence = emotion_confidence * 0.5  # Half the confidence of primary

                    # Only add if it's a different category than the primary
                    if secondary_mapped != mapped_emotion:
                        scores[secondary_mapped] += secondary_confidence

                    # Process third emotion with further reduced weight if available
                    if len(top_emotions) > 2:
                        tertiary_emotion = top_emotions[2]
                        tertiary_mapped = EMOTION_MAPPING.get(tertiary_emotion, 'neutral')
                        tertiary_confidence = emotion_confidence * 0.25  # Quarter the confidence of primary

                        # Only add if it's a different category than the primary and secondary
                        if tertiary_mapped != mapped_emotion and tertiary_mapped != secondary_mapped:
                            scores[tertiary_mapped] += tertiary_confidence

                # Adjust neutral score to be minimal unless no other emotions are detected
                # Sum all non-neutral emotion scores
                emotion_sum = sum(scores[e] for e in ['excitement', 'funny', 'happiness', 'anger', 'sadness'])

                # If we have significant emotion detected, reduce neutral score
                if emotion_sum > 0.3:  # If we have at least 30% confidence in any emotion
                    scores['neutral'] = max(0.0, 0.1)  # Keep a small amount of neutral
                else:
                    # Otherwise, set neutral to complement the sum of other emotions
                    scores['neutral'] = max(0.0, 1.0 - emotion_sum)

        # Process sentiment results
        if sentiment_result and 'sentiment' in sentiment_result:
            sentiment = sentiment_result['sentiment']
            sentiment_confidence = sentiment_result.get('confidence', 0.7)

            # Convert sentiment to score
            if sentiment == 'positive':
                scores['sentiment_score'] = sentiment_confidence
            elif sentiment == 'negative':
                scores['sentiment_score'] = -sentiment_confidence
            # neutral remains 0.0

        # Calculate highlight score based on emotions, sentiment, and both absolute and relative audio intensity
        # Higher scores for excitement, funny, and happiness
        highlight_base = 0.0
        mapped_emotion = next((e for e in ['excitement', 'funny', 'happiness', 'anger', 'sadness'] if scores[e] > 0), 'neutral')

        if mapped_emotion in ['excitement', 'funny', 'happiness']:
            highlight_base = 0.5  # Reduced to make room for audio intensity components
        elif mapped_emotion in ['anger', 'sadness']:
            highlight_base = 0.3  # Reduced to make room for audio intensity components

        # Calculate emotion intensity (how strong the emotion is)
        emotion_intensity = scores[mapped_emotion] if mapped_emotion != 'neutral' else 0.0

        # Calculate sentiment intensity (how strong the sentiment is)
        sentiment_intensity = abs(scores['sentiment_score'])

        # Get audio intensity values
        absolute_intensity = scores['audio_intensity']
        relative_intensity = scores['audio_intensity_relative']

        # Get speech rate factor (default to 1.0 if not provided)
        speech_rate_factor = 0.0
        if speech_rate is not None:
            # Convert normalized speech rate to a 0-1 score
            # A normalized rate of 1.0 (average) becomes 0.5
            # Rates below average get lower scores, rates above average get higher scores
            speech_rate_factor = min(1.0, max(0.0, (speech_rate - 0.5) * 0.5 + 0.5))

        # Combine all factors for highlight score with rebalanced weights
        scores['highlight_score'] = (
            highlight_base * 0.30 +                # Base score from emotion type (30% weight, was 35%)
            emotion_intensity * 0.20 +             # Emotion intensity (20% weight, was 15%)
            sentiment_intensity * 0.10 +           # Sentiment intensity (10% weight, unchanged)
            absolute_intensity * 0.05 +            # Absolute audio intensity (5% weight, was 10%)
            relative_intensity * 0.20 +            # Relative audio intensity (20% weight, was 15%)
            speech_rate_factor * 0.15              # Speech rate factor (15% weight, unchanged)
        )

        # Enhanced synergy bonus calculation
        # Boost score for segments with high emotion and either high relative audio intensity or high speech rate
        if emotion_intensity > 0.5 and (relative_intensity > 0.7 or speech_rate_factor > 0.7):
            synergy_factor = max(relative_intensity, speech_rate_factor if speech_rate is not None else 0)
            emotion_synergy = min(emotion_intensity, synergy_factor) * 0.2

            # Add extra bonus when ALL three factors align (emotion, audio, and speech rate)
            # This rewards moments where everything comes together
            if emotion_intensity > 0.6 and relative_intensity > 0.7 and speech_rate_factor > 0.7:
                emotion_synergy += 0.1  # Additional fixed bonus for multi-factor alignment

            scores['highlight_score'] += emotion_synergy

        # Cap at 1.0
        scores['highlight_score'] = min(1.0, scores['highlight_score'])

        # Ensure the sum of emotion scores is approximately 1.0
        emotion_sum = scores['excitement'] + scores['funny'] + scores['happiness'] + scores['anger'] + scores['sadness'] + scores['neutral']
        if emotion_sum > 0:
            scale_factor = 1.0 / emotion_sum
            scores['excitement'] = round(scores['excitement'] * scale_factor, 3)
            scores['funny'] = round(scores['funny'] * scale_factor, 3)
            scores['happiness'] = round(scores['happiness'] * scale_factor, 3)
            scores['anger'] = round(scores['anger'] * scale_factor, 3)
            scores['sadness'] = round(scores['sadness'] * scale_factor, 3)
            scores['neutral'] = round(scores['neutral'] * scale_factor, 3)

        # Round all scores to 3 decimal places
        for key in scores:
            if isinstance(scores[key], float):
                scores[key] = round(scores[key], 3)

        return scores
    except Exception as e:
        logger.error(f"Error generating scores with precomputed results: {str(e)}")
        return scores

def generate_scores(text, start_time, end_time, model=None, audio_intensities=(0.0, 0.0)):
    """
    Generate emotion and sentiment scores using the emotion and sentiment models

    Args:
        text (str): The text to analyze
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        model: The initialized emotion model (not used for sentiment anymore)
        audio_intensities (tuple): Tuple of (absolute_intensity, relative_intensity_z_score)

    Returns:
        dict: Dictionary containing all scores
    """
    # Extract absolute and relative audio intensities
    absolute_intensity, relative_intensity = audio_intensities

    # Initialize scores dictionary with default values
    scores = {
        'excitement': 0.0,
        'funny': 0.0,
        'happiness': 0.0,
        'anger': 0.0,
        'sadness': 0.0,
        'neutral': 1.0,  # Default to neutral
        'sentiment_score': 0.0,
        'highlight_score': 0.0,
        'audio_intensity': absolute_intensity,  # Store the absolute audio intensity
        'audio_intensity_relative': relative_intensity,  # Store the relative audio intensity (z-score)
        'text': text,
        'start_time': start_time,
        'end_time': end_time
    }

    try:
        # Skip very short texts to improve performance
        if len(text.split()) < 3:
            return scores

        # Analyze emotions (using global model if none provided)
        emotion_result = analyze_emotions(text, model)

        # Process emotions result
        if isinstance(emotion_result, dict) and 'llm_response' in emotion_result and 'emotions' in emotion_result['llm_response']:
            # Get the emotions list
            emotions = emotion_result['llm_response']['emotions']

            # If emotions detected
            if emotions:
                # Get the primary emotion (first in the list)
                primary_emotion = emotions[0].lower()

                # Map to our standard emotion categories
                # For unknown emotions, try to intelligently map them instead of defaulting to neutral
                if primary_emotion not in EMOTION_MAPPING:
                    # Check if the emotion contains certain keywords to help categorize it
                    if any(word in primary_emotion for word in ['joy', 'happ', 'content', 'satisf', 'pleas', 'delight', 'glad']):
                        mapped_emotion = 'happiness'
                    elif any(word in primary_emotion for word in ['excit', 'thrill', 'surpris', 'amaz', 'interest', 'energe', 'enthu', 'eager', 'inspir', 'motiv']):
                        mapped_emotion = 'excitement'
                    elif any(word in primary_emotion for word in ['fun', 'amus', 'humor', 'laugh', 'joke', 'silly', 'playful']):
                        mapped_emotion = 'funny'
                    elif any(word in primary_emotion for word in ['ang', 'mad', 'furi', 'annoy', 'irritat', 'frustrat', 'rage', 'hate', 'resent', 'bitter']):
                        mapped_emotion = 'anger'
                    elif any(word in primary_emotion for word in ['sad', 'depress', 'disappoint', 'upset', 'hurt', 'fear', 'anxi', 'terr', 'scare', 'worry', 'stress', 'grief', 'mourn', 'regret']):
                        mapped_emotion = 'sadness'
                    else:
                        # If we can't categorize it, try to make a best guess based on positive/negative connotation
                        positive_words = ['good', 'great', 'nice', 'love', 'like', 'posit', 'hope']
                        negative_words = ['bad', 'awful', 'terribl', 'horri', 'negat', 'dislike']

                        if any(word in primary_emotion for word in positive_words):
                            mapped_emotion = 'happiness'  # Default positive to happiness
                        elif any(word in primary_emotion for word in negative_words):
                            mapped_emotion = 'sadness'    # Default negative to sadness
                        else:
                            # If still can't categorize, default to excitement as it's more engaging than neutral
                            mapped_emotion = 'excitement'

                    logger.debug(f"Unknown emotion detected: {primary_emotion} - mapping to {mapped_emotion}")
                else:
                    mapped_emotion = EMOTION_MAPPING[primary_emotion]

                # Set confidence score (default to 0.8 if not provided)
                emotion_confidence = 0.8
                if 'confidence_score' in emotion_result:
                    emotion_confidence = float(emotion_result['confidence_score'])

                # Set the corresponding emotion score
                scores[mapped_emotion] = emotion_confidence

                # Process secondary emotions if available (up to 2 more)
                if len(emotions) > 1:
                    # Process second emotion with reduced weight
                    secondary_emotion = emotions[1].lower()
                    secondary_mapped = EMOTION_MAPPING.get(secondary_emotion, 'neutral')
                    secondary_confidence = emotion_confidence * 0.5  # Half the confidence of primary

                    # Only add if it's a different category than the primary
                    if secondary_mapped != mapped_emotion:
                        scores[secondary_mapped] += secondary_confidence

                    # Process third emotion with further reduced weight if available
                    if len(emotions) > 2:
                        tertiary_emotion = emotions[2].lower()
                        tertiary_mapped = EMOTION_MAPPING.get(tertiary_emotion, 'neutral')
                        tertiary_confidence = emotion_confidence * 0.25  # Quarter the confidence of primary

                        # Only add if it's a different category than the primary and secondary
                        if tertiary_mapped != mapped_emotion and tertiary_mapped != secondary_mapped:
                            scores[tertiary_mapped] += tertiary_confidence

                # Adjust neutral score to be minimal unless no other emotions are detected
                # Sum all non-neutral emotion scores
                emotion_sum = sum(scores[e] for e in ['excitement', 'funny', 'happiness', 'anger', 'sadness'])

                # If we have significant emotion detected, reduce neutral score
                if emotion_sum > 0.3:  # If we have at least 30% confidence in any emotion
                    scores['neutral'] = max(0.0, 0.1)  # Keep a small amount of neutral
                else:
                    # Otherwise, set neutral to complement the sum of other emotions
                    scores['neutral'] = max(0.0, 1.0 - emotion_sum)

        # Analyze sentiment using RoBERTa (no need to pass model)
        sentiment_result = analyze_sentiment(text)

        # Process sentiment result
        if isinstance(sentiment_result, dict) and 'llm_response' in sentiment_result and 'sentiment' in sentiment_result['llm_response']:
            # Get the sentiment
            sentiment = sentiment_result['llm_response']['sentiment']

            # If sentiment detected
            if sentiment:
                # Get the primary sentiment (first in the list)
                primary_sentiment = sentiment[0].lower()

                # Set confidence score (default to 0.7 if not provided)
                sentiment_confidence = 0.7
                if 'confidence_score' in sentiment_result:
                    sentiment_confidence = float(sentiment_result['confidence_score'])

                # Convert sentiment to score
                if primary_sentiment == 'positive':
                    scores['sentiment_score'] = sentiment_confidence
                elif primary_sentiment == 'negative':
                    scores['sentiment_score'] = -sentiment_confidence
                # neutral remains 0.0

        # Calculate highlight score based on emotions, sentiment, and both absolute and relative audio intensity
        # Higher scores for excitement, funny, and happiness
        highlight_base = 0.0
        mapped_emotion = next((e for e in ['excitement', 'funny', 'happiness', 'anger', 'sadness'] if scores[e] > 0), 'neutral')

        if mapped_emotion in ['excitement', 'funny', 'happiness']:
            highlight_base = 0.5  # Reduced to make room for audio intensity components
        elif mapped_emotion in ['anger', 'sadness']:
            highlight_base = 0.3  # Reduced to make room for audio intensity components

        # Calculate emotion intensity (how strong the emotion is)
        emotion_intensity = scores[mapped_emotion] if mapped_emotion != 'neutral' else 0.0

        # Calculate sentiment intensity (how strong the sentiment is)
        sentiment_intensity = abs(scores['sentiment_score'])

        # Get audio intensity values
        absolute_intensity = scores['audio_intensity']
        relative_intensity = scores['audio_intensity_relative']

        # Combine all factors for highlight score with both absolute and relative audio intensity
        scores['highlight_score'] = (
            highlight_base * 0.4 +                # Base score from emotion type (40% weight)
            emotion_intensity * 0.2 +             # Emotion intensity (20% weight)
            sentiment_intensity * 0.1 +           # Sentiment intensity (10% weight)
            absolute_intensity * 0.1 +            # Absolute audio intensity (10% weight)
            relative_intensity * 0.2              # Relative audio intensity (20% weight)
        )

        # Boost score for segments with both high emotion and high relative audio intensity
        # This helps identify moments where emotional content aligns with volume changes
        if emotion_intensity > 0.5 and relative_intensity > 0.7:
            emotion_audio_synergy = min(emotion_intensity, relative_intensity) * 0.2
            scores['highlight_score'] += emotion_audio_synergy

        # Cap at 1.0
        scores['highlight_score'] = min(1.0, scores['highlight_score'])

        # Ensure the sum of emotion scores is approximately 1.0
        emotion_sum = scores['excitement'] + scores['funny'] + scores['happiness'] + scores['anger'] + scores['sadness'] + scores['neutral']
        if emotion_sum > 0:
            scale_factor = 1.0 / emotion_sum
            scores['excitement'] = round(scores['excitement'] * scale_factor, 3)
            scores['funny'] = round(scores['funny'] * scale_factor, 3)
            scores['happiness'] = round(scores['happiness'] * scale_factor, 3)
            scores['anger'] = round(scores['anger'] * scale_factor, 3)
            scores['sadness'] = round(scores['sadness'] * scale_factor, 3)
            scores['neutral'] = round(scores['neutral'] * scale_factor, 3)

        # Round all scores to 3 decimal places
        for key in scores:
            if isinstance(scores[key], float):
                scores[key] = round(scores[key], 3)

        return scores
    except Exception as e:
        logger.error(f"Error generating scores: {str(e)}")
        return scores

def analyze_audio_sentiment(video_id, input_file=None, output_dir=None, audio_file=None):
    """Main function to process audio sentiment analysis for a specific video

    Args:
        video_id (str): The ID of the video to process
        input_file (str, optional): Path to the input segments CSV file
        output_dir (str, optional): Directory to save output files
        audio_file (str, optional): Path to the audio file

    Returns:
        bool: True if analysis was successful, False otherwise
    """
    # Load environment variables
    load_dotenv()

    try:
        # Define output directory
        if output_dir is None:
            output_dir = Path('Output/Analysis/Audio')
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Determine input file path
        if input_file is None:
            if video_id:
                input_file = Path(f'Output/Raw/Transcripts/audio_{video_id}_segments.csv')
            else:
                # Default to first audio file found if no video ID provided
                audio_files = list(Path('Output/Raw/Transcripts').glob('audio_*_segments.csv'))
                if not audio_files:
                    raise FileNotFoundError(f"No audio segment files found in Output/Raw/Transcripts directory")
                input_file = str(audio_files[0])
                # Extract video ID from filename
                video_id = Path(input_file).stem.split('_')[1]
        else:
            input_file = Path(input_file)

        logger.info(f"Processing video ID: {video_id}")
        logger.info(f"Reading input file: {input_file}")

        # Read the existing CSV file
        df = pd.read_csv(input_file)

        # Initialize the models
        logger.info("Initializing llmware models...")
        model = initialize_models()

        if model is None:
            logger.warning("Failed to initialize models, using default values")

        # Determine audio file path
        if audio_file is None:
            # Try different possible locations
            possible_audio_paths = [
                Path(f'Output/Raw/Audio/audio_{video_id}.wav'),
                Path(f'outputs/audio_{video_id}.wav'),
                Path(f'/tmp/outputs/audio_{video_id}.wav')
            ]

            for path in possible_audio_paths:
                if path.exists():
                    audio_file = path
                    logger.info(f"Found audio file: {audio_file}")
                    break
            else:
                logger.warning("Audio file not found, proceeding without audio intensity")
                audio_file = None

        # Extract audio features once for the entire file
        audio_data = None
        if audio_file:
            logger.info("Extracting audio features for the entire file (this may take a moment)...")
            audio_data = extract_audio_features(audio_file)
            logger.info("Audio feature extraction complete")

        # Initialize rolling audio stats for Z-score calculation
        rolling_stats = RollingAudioStats(window_size=600)  # 10-minute window

        # Define a function to process a batch of segments
        def process_segment_batch(batch_rows, audio_data, model, rolling_stats, batch_size=16):
            try:
                # Extract text, start_time, and end_time from each row
                texts = []
                start_times = []
                end_times = []
                audio_intensities_list = []
                speech_rates_list = []

                # Calculate average speech rate for the entire batch first
                total_words = 0
                total_duration = 0

                for _, row in batch_rows.iterrows():
                    text = row['text']
                    start_time = row['start_time']
                    end_time = row['end_time']

                    # Count words for average calculation
                    if text and text.strip():
                        word_count = len(text.split())
                        duration = end_time - start_time
                        if duration > 0:
                            total_words += word_count
                            total_duration += duration

                # Calculate average speech rate for normalization
                avg_speech_rate = total_words / total_duration if total_duration > 0 else 1.0

                # Now process each segment
                for _, row in batch_rows.iterrows():
                    texts.append(row['text'])
                    start_times.append(row['start_time'])
                    end_times.append(row['end_time'])

                    # Extract audio intensity if audio data is available
                    audio_intensities = (0.0, 0.0)  # Default: (absolute_intensity, relative_intensity)
                    if audio_data is not None:
                        audio_intensities = extract_audio_intensity(
                            audio_data,
                            row['start_time'],
                            row['end_time'],
                            rolling_stats
                        )
                    audio_intensities_list.append(audio_intensities)

                    # Calculate speech rate and normalize
                    speech_rate = calculate_speech_rate(row['text'], row['start_time'], row['end_time'])
                    normalized_speech_rate = speech_rate / avg_speech_rate if avg_speech_rate > 0 else 0.0

                    # Cap extreme values
                    normalized_speech_rate = min(3.0, normalized_speech_rate)

                    speech_rates_list.append(normalized_speech_rate)

                # Batch process emotions
                emotion_results = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]

                    # Skip empty texts
                    batch_texts = [t if t and t.strip() else "neutral" for t in batch_texts]

                    # Tokenize all texts in one batch
                    inputs = emotion_tokenizer(batch_texts, padding=True, truncation=True,
                                              max_length=512, return_tensors="pt")

                    # Move inputs to the same device as the model
                    device = next(emotion_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Run emotion inference on the entire batch
                    with torch.no_grad():
                        outputs = emotion_model(**inputs)
                        emotion_scores = torch.nn.functional.softmax(outputs.logits, dim=1)

                    # Process emotion results for this batch
                    for j in range(len(batch_texts)):
                        # Get the top 3 emotions
                        top_scores, top_indices = torch.topk(emotion_scores[j], 3)

                        # The 28 emotion categories from cirimus/modernbert-large-go-emotions
                        emotion_labels = [
                            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                            'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
                        ]

                        top_emotions = [emotion_labels[idx] for idx in top_indices.tolist()]
                        emotion_results.append({
                            'emotions': top_emotions,
                            'confidence': top_scores[0].item()
                        })

                # Batch process sentiment
                sentiment_results = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]

                    # Skip empty texts
                    batch_texts = [t if t and t.strip() else "neutral" for t in batch_texts]

                    # Tokenize all texts in one batch
                    inputs = sentiment_tokenizer(batch_texts, padding=True, truncation=True,
                                               max_length=512, return_tensors="pt")

                    # Move inputs to the same device as the model
                    device = next(sentiment_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Run sentiment inference on the entire batch
                    with torch.no_grad():
                        outputs = sentiment_model(**inputs)
                        sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=1)

                    # Process sentiment results for this batch
                    for j in range(len(batch_texts)):
                        # Get prediction
                        prediction_idx = sentiment_scores[j].argmax().item()
                        confidence = sentiment_scores[j][prediction_idx].item()

                        # Map to sentiment labels
                        sentiment_labels = ['negative', 'neutral', 'positive']
                        sentiment = sentiment_labels[prediction_idx]

                        sentiment_results.append({
                            'sentiment': sentiment,
                            'confidence': confidence
                        })

                # Generate final scores for each segment
                batch_results = []
                for i in range(len(texts)):
                    # Generate scores with pre-computed results, including speech rate
                    scores = generate_scores_with_precomputed_results(
                        texts[i],
                        start_times[i],
                        end_times[i],
                        audio_intensities_list[i],
                        emotion_results[i],
                        sentiment_results[i],
                        speech_rates_list[i]  # Pass the normalized speech rate
                    )
                    batch_results.append(scores)

                return batch_results

            except Exception as e:
                logger.error(f"Error processing segment batch: {str(e)}")
                # Return default scores for each segment in the batch
                default_results = []

                # Create a default result for each row in the batch
                for _, row in batch_rows.iterrows():
                    default_result = {
                        'excitement': 0.0,
                        'funny': 0.0,
                        'happiness': 0.0,
                        'anger': 0.0,
                        'sadness': 0.0,
                        'neutral': 1.0,
                        'sentiment_score': 0.0,
                        'highlight_score': 0.0,
                        'audio_intensity': 0.0,
                        'audio_intensity_relative': 0.0,
                        'text': str(row.get('text', "")),
                        'start_time': float(row.get('start_time', 0.0)),
                        'end_time': float(row.get('end_time', 0.0))
                    }
                    default_results.append(default_result)

                return default_results

        # Process all segments without filtering by length
        filtered_df = df.copy()

        logger.info(f"Processing all {len(filtered_df)} segments")

        # Generate scores for segments using batch processing
        logger.info("Generating emotion, sentiment, and audio intensity scores for segments")
        results = []
        total_rows = len(filtered_df)

        # Determine batch size and number of workers
        batch_size = 16  # Process 16 segments in a single model inference
        max_workers = min(os.cpu_count() or 4, 4)  # Use at most 4 workers for batch processing
        logger.info(f"Using batch size {batch_size} with {max_workers} parallel workers")

        # Split dataframe into batches
        batch_dfs = []
        for i in range(0, len(filtered_df), batch_size):
            end_idx = min(i + batch_size, len(filtered_df))
            batch_dfs.append(filtered_df.iloc[i:end_idx])

        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(process_segment_batch, batch_df, audio_data, model, rolling_stats, batch_size): i
                for i, batch_df in enumerate(batch_dfs)
            }

            # Process results as they complete
            completed_batches = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_results = future.result()
                results.extend(batch_results)

                completed_batches += 1
                completed_segments = min(completed_batches * batch_size, total_rows)

                # Log progress periodically
                if completed_batches % 5 == 0 or completed_batches == len(batch_dfs):
                    logger.info(f"Processed {completed_segments}/{total_rows} segments ({completed_segments/total_rows*100:.1f}%)")

            # Add a small delay to avoid overwhelming the system
            time.sleep(0.2)

        logger.info(f"Processing complete. Generated scores for {len(results)}/{total_rows} segments")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Add new columns to original DataFrame
        emotion_columns = [col for col in results_df.columns if col not in ['start_time', 'end_time', 'text']]

        # Create a mapping from text to results for efficient lookup
        result_map = {r['text']: r for r in results}

        # Update the original dataframe with results
        for i, row in df.iterrows():
            text = row['text']
            if text in result_map:
                for col in emotion_columns:
                    df.at[i, col] = result_map[text][col]
            else:
                # Use default values for rows without results
                for col in emotion_columns:
                    if col == 'neutral':
                        df.at[i, col] = 1.0
                    elif col in ['audio_intensity', 'audio_intensity_relative']:
                        df.at[i, col] = 0.0
                    else:
                        df.at[i, col] = 0.0

        # Save to output file
        output_file = output_dir / f"audio_{video_id}_sentiment.csv"
        logger.info(f"Saving results to: {output_file}")
        df.to_csv(output_file, index=False)
        logger.info("Save complete")

        return True

    except Exception as e:
        logger.error(f"Error in sentiment_analysis: {str(e)}")
        return False
