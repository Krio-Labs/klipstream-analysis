"""
Integration Module for Chat and Audio Analysis

This module integrates chat analysis data with audio sentiment analysis to create
a comprehensive analysis that leverages both audience reactions and streamer content.
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from google.oauth2 import service_account
from pathlib import Path

from utils.config import ANALYSIS_BUCKET, GCP_SERVICE_ACCOUNT_PATH, USE_GCS
from utils.file_manager import FileManager
from utils.convex_client_updated import ConvexManager
# Video processing status constants
STATUS_QUEUED = "Queued"
STATUS_DOWNLOADING = "Downloading"
STATUS_FETCHING_CHAT = "Fetching chat"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_ANALYZING = "Analyzing"
STATUS_FINDING_HIGHLIGHTS = "Finding highlights"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

# Configure logging
logger = logging.getLogger(__name__)

def extract_weighted_chat_metrics(chat_df, start_time, end_time):
    """
    Extract time-weighted chat metrics for a specific time range

    Args:
        chat_df (DataFrame): DataFrame containing chat analysis
        start_time (float): Start time of the segment in seconds
        end_time (float): End time of the segment in seconds

    Returns:
        dict: Dictionary of weighted chat metrics
    """
    # Initialize metrics
    metrics = {
        'chat_volume': 0.0,
        'chat_sentiment': 0.0,
        'chat_highlight_score': 0.0,
        'chat_excitement': 0.0,
        'chat_funny': 0.0,
        'chat_happiness': 0.0,
        'chat_anger': 0.0,
        'chat_sadness': 0.0,
        'chat_neutral': 0.0
    }

    # Define mapping from chat analysis column names to our expected names
    column_mapping = {
        'message_count': 'chat_volume',
        'avg_sentiment': 'chat_sentiment',
        'avg_highlight': 'chat_highlight_score',
        'avg_excitement': 'chat_excitement',
        'avg_funny': 'chat_funny',
        'avg_happiness': 'chat_happiness',
        'avg_anger': 'chat_anger',
        'avg_sadness': 'chat_sadness',
        'avg_neutral': 'chat_neutral'
    }

    # If no chat data, return zeros
    if chat_df.empty:
        return metrics

    # Calculate weighted average based on overlap duration
    total_weight = 0

    for _, row in chat_df.iterrows():
        # Calculate overlap duration
        overlap_start = max(start_time, row['start_time'])
        overlap_end = min(end_time, row['end_time'])
        overlap_duration = max(0, overlap_end - overlap_start)

        # Skip if no overlap
        if overlap_duration <= 0:
            continue

        # Use overlap duration as weight
        weight = overlap_duration
        total_weight += weight

        # Weight each metric using the column mapping
        for chat_col, metric in column_mapping.items():
            if chat_col in row:
                metrics[metric] += row[chat_col] * weight

    # Calculate weighted averages
    if total_weight > 0:
        for metric in metrics:
            metrics[metric] /= total_weight

    return metrics

def fuse_emotions(audio_emotions, chat_emotions, audio_weight=0.6, chat_weight=0.4):
    """
    Create a weighted fusion of emotions from audio and chat analysis

    Args:
        audio_emotions (dict): Dictionary of emotion scores from audio analysis
        chat_emotions (dict): Dictionary of emotion scores from chat analysis
        audio_weight (float): Weight to give audio emotions (0-1)
        chat_weight (float): Weight to give chat emotions (0-1)

    Returns:
        dict: Dictionary of fused emotion scores
    """
    # Ensure weights sum to 1
    total_weight = audio_weight + chat_weight
    audio_weight = audio_weight / total_weight
    chat_weight = chat_weight / total_weight

    # Initialize fused emotions with all possible categories
    emotion_categories = ['excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral']
    fused_emotions = {emotion: 0.0 for emotion in emotion_categories}

    # Add weighted audio emotions
    for emotion, score in audio_emotions.items():
        if emotion in fused_emotions:
            fused_emotions[emotion] += score * audio_weight

    # Add weighted chat emotions
    for emotion, score in chat_emotions.items():
        if emotion in fused_emotions:
            fused_emotions[emotion] += score * chat_weight

    # Normalize to ensure sum is approximately 1.0
    emotion_sum = sum(fused_emotions.values())
    if emotion_sum > 0:
        for emotion in fused_emotions:
            fused_emotions[emotion] /= emotion_sum

    return fused_emotions

def fuse_sentiment(audio_sentiment, audio_confidence, chat_sentiment, chat_confidence):
    """
    Fuse sentiment scores from audio and chat, weighted by confidence

    Args:
        audio_sentiment (float): Sentiment score from audio (-1 to 1)
        audio_confidence (float): Confidence in audio sentiment (0-1)
        chat_sentiment (float): Sentiment score from chat (-1 to 1)
        chat_confidence (float): Confidence in chat sentiment (0-1)

    Returns:
        float: Fused sentiment score (-1 to 1)
    """
    # If either source has zero confidence, use the other
    if audio_confidence <= 0:
        return chat_sentiment
    if chat_confidence <= 0:
        return audio_sentiment

    # Calculate weighted average based on confidence
    total_confidence = audio_confidence + chat_confidence
    audio_weight = audio_confidence / total_confidence
    chat_weight = chat_confidence / total_confidence

    # Fuse sentiment scores
    fused_sentiment = (audio_sentiment * audio_weight) + (chat_sentiment * chat_weight)

    return fused_sentiment

def analyze_emotional_coherence(audio_emotions, chat_emotions):
    """
    Analyze how well streamer emotions align with audience emotions

    Args:
        audio_emotions (dict): Dictionary of emotion scores from audio
        chat_emotions (dict): Dictionary of emotion scores from chat

    Returns:
        dict: Coherence metrics
    """
    # Calculate dominant emotions
    audio_dominant = max(audio_emotions.items(), key=lambda x: x[1])[0] if audio_emotions else 'neutral'
    chat_dominant = max(chat_emotions.items(), key=lambda x: x[1])[0] if chat_emotions else 'neutral'

    # Calculate alignment score (how well emotions match)
    alignment = 0.0
    for emotion in audio_emotions:
        if emotion in chat_emotions:
            # Higher score when both have similar values for the same emotion
            alignment += 1.0 - abs(audio_emotions[emotion] - chat_emotions[emotion])

    # Normalize alignment score
    alignment = alignment / len(audio_emotions) if audio_emotions else 0.0

    # Determine emotional relationship
    if audio_dominant == chat_dominant:
        relationship = "synchronized"  # Same dominant emotion
    elif (audio_dominant in ['excitement', 'happiness', 'funny'] and
          chat_dominant in ['excitement', 'happiness', 'funny']):
        relationship = "positive_aligned"  # Different but both positive
    elif (audio_dominant in ['anger', 'sadness'] and
          chat_dominant in ['anger', 'sadness']):
        relationship = "negative_aligned"  # Different but both negative
    elif (audio_dominant in ['excitement', 'happiness', 'funny'] and
          chat_dominant in ['anger', 'sadness']):
        relationship = "audience_negative"  # Streamer positive, audience negative
    elif (audio_dominant in ['anger', 'sadness'] and
          chat_dominant in ['excitement', 'happiness', 'funny']):
        relationship = "audience_positive"  # Streamer negative, audience positive
    else:
        relationship = "mixed"  # Other combinations

    return {
        "audio_dominant": audio_dominant,
        "chat_dominant": chat_dominant,
        "alignment_score": alignment,
        "relationship": relationship
    }

def calculate_enhanced_highlight_score(row, coherence_data):
    """
    Calculate enhanced highlight score incorporating emotional coherence

    Args:
        row (Series): Row from merged dataframe
        coherence_data (dict): Emotional coherence analysis

    Returns:
        float: Enhanced highlight score
    """
    # Get base components
    base_highlight = row['highlight_score']  # Audio-based highlight score
    chat_highlight = row['chat_highlight_score']  # Chat-based highlight score

    # Get coherence components
    alignment = coherence_data['alignment_score']
    relationship = coherence_data['relationship']

    # Base combined score (weighted average)
    combined_score = base_highlight * 0.6 + chat_highlight * 0.4

    # Apply relationship-specific adjustments
    if relationship == "synchronized":
        # Perfect alignment between streamer and audience emotions
        # This is likely a very genuine moment
        coherence_bonus = 0.15
    elif relationship == "positive_aligned":
        # Both positive but different emotions
        # Still a good highlight candidate
        coherence_bonus = 0.1
    elif relationship == "audience_positive":
        # Audience positive despite streamer negative
        # Could be entertaining (audience enjoying streamer's frustration)
        coherence_bonus = 0.08
    elif relationship == "audience_negative":
        # Audience negative despite streamer positive
        # Could be controversial or divisive content
        coherence_bonus = 0.05
    else:
        coherence_bonus = 0.0

    # Apply alignment-based adjustment
    alignment_factor = alignment * 0.1

    # Calculate final score
    enhanced_score = combined_score + coherence_bonus + alignment_factor

    # Cap at 1.0
    enhanced_score = min(1.0, enhanced_score)

    return enhanced_score

def determine_optimal_weights(audio_df, chat_df):
    """
    Determine optimal weights for audio vs chat based on stream characteristics

    Args:
        audio_df (DataFrame): Audio analysis data
        chat_df (DataFrame): Chat analysis data

    Returns:
        tuple: (audio_weight, chat_weight)
    """
    # Calculate average chat activity (messages per minute)
    total_duration = audio_df['end_time'].max() - audio_df['start_time'].min()

    # Check if 'message_count' exists in chat_df
    if 'message_count' in chat_df.columns:
        total_messages = chat_df['message_count'].sum()
    else:
        # Fallback if message_count doesn't exist
        total_messages = len(chat_df)

    chat_activity = total_messages / (total_duration / 60) if total_duration > 0 else 0

    # Calculate audio emotion intensity
    emotion_cols = ['excitement', 'funny', 'happiness', 'anger', 'sadness']
    audio_emotion_cols = [col for col in emotion_cols if col in audio_df.columns]
    audio_emotion_intensity = audio_df[audio_emotion_cols].max(axis=1).mean() if audio_emotion_cols else 0.5

    # Calculate chat emotion intensity
    chat_emotion_cols = [col for col in emotion_cols if col in chat_df.columns]
    chat_emotion_intensity = chat_df[chat_emotion_cols].max(axis=1).mean() if chat_emotion_cols else 0.5

    # Adjust weights based on relative strengths
    # High chat activity → more weight to chat
    # High audio emotion → more weight to audio
    base_audio_weight = 0.6
    base_chat_weight = 0.4

    # Adjust for chat activity (more chat → more chat weight)
    chat_activity_factor = min(1.0, max(0.0, (chat_activity - 10) / 100))  # Normalize to 0-1

    # Adjust for relative emotion intensity
    if audio_emotion_intensity > 0 and chat_emotion_intensity > 0:
        emotion_ratio = audio_emotion_intensity / chat_emotion_intensity
        emotion_factor = min(1.0, max(0.0, (emotion_ratio - 0.5) / 2))  # Normalize to 0-1
    else:
        emotion_factor = 0.5

    # Calculate final weights
    audio_weight = base_audio_weight + (chat_activity_factor * 0.2) - (emotion_factor * 0.2)
    chat_weight = base_chat_weight - (chat_activity_factor * 0.2) + (emotion_factor * 0.2)

    # Ensure weights are in valid range
    audio_weight = min(0.8, max(0.4, audio_weight))
    chat_weight = min(0.6, max(0.2, chat_weight))

    # Normalize to sum to 1
    total = audio_weight + chat_weight
    audio_weight /= total
    chat_weight /= total

    return audio_weight, chat_weight

def integrate_chat_and_audio_analysis(audio_df, chat_df):
    """
    Comprehensive integration of chat and audio analysis

    Args:
        audio_df (DataFrame): Audio analysis dataframe
        chat_df (DataFrame): Chat analysis dataframe

    Returns:
        DataFrame: Integrated analysis dataframe
    """
    logger.info("Starting integration of chat and audio analysis")

    # Create a copy of audio dataframe for integration
    integrated_df = audio_df.copy()

    # Add columns for chat metrics
    chat_columns = ['chat_volume', 'chat_sentiment', 'chat_highlight_score',
                   'chat_excitement', 'chat_funny', 'chat_happiness',
                   'chat_anger', 'chat_sadness', 'chat_neutral']

    for col in chat_columns:
        integrated_df[col] = 0.0

    # Add columns for fused metrics
    integrated_df['fused_sentiment'] = 0.0
    for emotion in ['excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral']:
        integrated_df[f'fused_{emotion}'] = 0.0

    # Add columns for coherence analysis
    integrated_df['emotional_alignment'] = 0.0
    integrated_df['emotional_relationship'] = ''
    integrated_df['enhanced_highlight_score'] = 0.0

    # Determine optimal weights for this stream
    audio_weight, chat_weight = determine_optimal_weights(audio_df, chat_df)
    logger.info(f"Determined optimal weights: audio={audio_weight:.2f}, chat={chat_weight:.2f}")

    # Process each segment
    total_rows = len(integrated_df)
    logger.info(f"Processing {total_rows} segments")

    for i, row in integrated_df.iterrows():
        # Log progress periodically
        if i % 100 == 0 or i == total_rows - 1:
            logger.info(f"Processing segment {i+1}/{total_rows} ({(i+1)/total_rows*100:.1f}%)")

        # Get time range for this segment
        start_time = row['start_time']
        end_time = row['end_time']

        # Find overlapping chat segments
        overlapping_chat = chat_df[
            ((chat_df['start_time'] >= start_time) & (chat_df['start_time'] <= end_time)) |
            ((chat_df['end_time'] >= start_time) & (chat_df['end_time'] <= end_time)) |
            ((chat_df['start_time'] <= start_time) & (chat_df['end_time'] >= end_time))
        ]

        if not overlapping_chat.empty:
            # Extract chat metrics with time-weighted averaging
            chat_metrics = extract_weighted_chat_metrics(overlapping_chat, start_time, end_time)

            # Update chat columns
            for col in chat_columns:
                if col in chat_metrics:
                    integrated_df.at[i, col] = chat_metrics[col]

            # Extract audio emotions
            audio_emotions = {
                'excitement': row['excitement'],
                'funny': row['funny'],
                'happiness': row['happiness'],
                'anger': row['anger'],
                'sadness': row['sadness'],
                'neutral': row['neutral']
            }

            # Extract chat emotions
            chat_emotions = {
                'excitement': chat_metrics.get('chat_excitement', 0.0),
                'funny': chat_metrics.get('chat_funny', 0.0),
                'happiness': chat_metrics.get('chat_happiness', 0.0),
                'anger': chat_metrics.get('chat_anger', 0.0),
                'sadness': chat_metrics.get('chat_sadness', 0.0),
                'neutral': chat_metrics.get('chat_neutral', 0.0)
            }

            # Fuse emotions
            fused_emotions = fuse_emotions(audio_emotions, chat_emotions, audio_weight, chat_weight)

            # Update fused emotion columns
            for emotion, score in fused_emotions.items():
                integrated_df.at[i, f'fused_{emotion}'] = score

            # Fuse sentiment
            audio_sentiment = row['sentiment_score']
            chat_sentiment = chat_metrics.get('chat_sentiment', 0.0)
            # Estimate confidence from emotion intensity
            audio_confidence = max(audio_emotions.values())
            chat_confidence = max(chat_emotions.values())

            fused_sentiment = fuse_sentiment(audio_sentiment, audio_confidence,
                                           chat_sentiment, chat_confidence)
            integrated_df.at[i, 'fused_sentiment'] = fused_sentiment

            # Analyze emotional coherence
            coherence_data = analyze_emotional_coherence(audio_emotions, chat_emotions)
            integrated_df.at[i, 'emotional_alignment'] = coherence_data['alignment_score']
            integrated_df.at[i, 'emotional_relationship'] = coherence_data['relationship']

            # Calculate enhanced highlight score
            enhanced_score = calculate_enhanced_highlight_score(
                integrated_df.iloc[i], coherence_data
            )
            integrated_df.at[i, 'enhanced_highlight_score'] = enhanced_score

    logger.info("Integration complete")
    return integrated_df

def plot_emotional_coherence(integrated_df, output_dir, video_id):
    """
    Plot the emotional coherence between streamer and audience over time

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        output_dir (str): Directory to save the plot
        video_id (str): Video ID for filename
    """
    logger.info("Generating emotional coherence plot")

    plt.figure(figsize=(15, 8))

    # Plot alignment score
    plt.subplot(2, 1, 1)
    plt.plot(integrated_df['start_time'], integrated_df['emotional_alignment'], 'b-')
    plt.title('Emotional Alignment Between Streamer and Audience')
    plt.ylabel('Alignment Score')
    plt.grid(True)

    # Plot relationship categories
    plt.subplot(2, 1, 2)
    # Convert relationship categories to numeric values for plotting
    relationship_map = {
        'synchronized': 5,
        'positive_aligned': 4,
        'audience_positive': 3,
        'mixed': 2,
        'audience_negative': 1,
        'negative_aligned': 0
    }

    # Handle missing values
    relationship_values = integrated_df['emotional_relationship'].map(
        lambda x: relationship_map.get(x, 2)  # Default to 'mixed' if not found
    )

    plt.scatter(integrated_df['start_time'], relationship_values, c=relationship_values, cmap='viridis')
    plt.yticks(list(relationship_map.values()), list(relationship_map.keys()))
    plt.title('Emotional Relationship Categories')
    plt.xlabel('Stream Time (seconds)')
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"{video_id}_emotional_coherence.png")
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Emotional coherence plot saved to {plot_path}")

def plot_highlight_comparison(integrated_df, output_dir, video_id):
    """
    Plot a comparison of different highlight detection methods

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        output_dir (str): Directory to save the plot
        video_id (str): Video ID for filename
    """
    logger.info("Generating highlight comparison plot")

    plt.figure(figsize=(15, 10))

    # Plot audio-only highlights
    plt.subplot(3, 1, 1)
    plt.plot(integrated_df['start_time'], integrated_df['highlight_score'], 'b-')
    plt.title('Audio-Only Highlight Score')
    plt.ylabel('Score')
    plt.grid(True)

    # Plot chat-only highlights
    plt.subplot(3, 1, 2)
    plt.plot(integrated_df['start_time'], integrated_df['chat_highlight_score'], 'g-')
    plt.title('Chat-Only Highlight Score')
    plt.ylabel('Score')
    plt.grid(True)

    # Plot enhanced highlights
    plt.subplot(3, 1, 3)
    plt.plot(integrated_df['start_time'], integrated_df['enhanced_highlight_score'], 'r-')
    plt.title('Enhanced Highlight Score (Combined with Emotional Coherence)')
    plt.xlabel('Stream Time (seconds)')
    plt.ylabel('Score')
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"{video_id}_highlight_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Highlight comparison plot saved to {plot_path}")

def apply_smoothing(x, y, window_size=15, method='moving_average'):
    """
    Apply smoothing to a time series

    Args:
        x (array): Time values
        y (array): Data values to smooth
        window_size (int): Size of the smoothing window
        method (str): Smoothing method ('moving_average', 'exponential', or 'savgol')

    Returns:
        tuple: (smoothed_x, smoothed_y)
    """
    import numpy as np
    from scipy import signal

    # Handle empty or very small data
    if len(y) < window_size:
        return x, y

    if method == 'moving_average':
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        smoothed_y = np.convolve(y, kernel, mode='same')

        # Fix the edges (they get distorted by the convolution)
        half_window = window_size // 2
        smoothed_y[:half_window] = y[:half_window]
        smoothed_y[-half_window:] = y[-half_window:]

        return x, smoothed_y

    elif method == 'exponential':
        # Exponential moving average
        alpha = 2 / (window_size + 1)  # Smoothing factor
        smoothed_y = np.zeros_like(y)
        smoothed_y[0] = y[0]

        for i in range(1, len(y)):
            smoothed_y[i] = alpha * y[i] + (1 - alpha) * smoothed_y[i-1]

        return x, smoothed_y

    elif method == 'savgol':
        # Savitzky-Golay filter (polynomial smoothing)
        # Window size must be odd
        if window_size % 2 == 0:
            window_size += 1

        # Polynomial order must be less than window size
        poly_order = min(3, window_size - 1)

        try:
            smoothed_y = signal.savgol_filter(y, window_size, poly_order)
            return x, smoothed_y
        except Exception:
            # Fall back to moving average if savgol fails
            return apply_smoothing(x, y, window_size, 'moving_average')

    else:
        # Default to no smoothing
        return x, y

def plot_emotion_comparison(integrated_df, emotion, output_dir, video_id, smooth=True, window_size=15):
    """
    Plot comparison of streamer, audience, and fused emotion for a specific emotion

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        emotion (str): Emotion to plot (e.g., 'excitement', 'funny')
        output_dir (str): Directory to save the plot
        video_id (str): Video ID for filename
        smooth (bool): Whether to apply smoothing
        window_size (int): Size of the smoothing window
    """
    plt.figure(figsize=(15, 8))

    # Define column names for this emotion
    streamer_col = emotion
    audience_col = f'chat_{emotion}'
    fused_col = f'fused_{emotion}'

    # Get time values
    x = integrated_df['start_time'].values

    # Get and potentially smooth the emotion values
    streamer_y = integrated_df[streamer_col].values
    audience_y = integrated_df[audience_col].values
    fused_y = integrated_df[fused_col].values

    if smooth:
        # Apply smoothing
        _, streamer_y = apply_smoothing(x, streamer_y, window_size, 'savgol')
        _, audience_y = apply_smoothing(x, audience_y, window_size, 'savgol')
        _, fused_y = apply_smoothing(x, fused_y, window_size, 'savgol')

    # Plot each source
    plt.plot(x, streamer_y, 'b-',
             label=f'Streamer {emotion.capitalize()}', linewidth=2)
    plt.plot(x, audience_y, 'g-',
             label=f'Audience {emotion.capitalize()}', linewidth=2)
    plt.plot(x, fused_y, 'r-',
             label=f'Fused {emotion.capitalize()}', linewidth=2.5)

    # Add title and labels
    if smooth:
        plt.title(f'{emotion.capitalize()} Emotion Throughout Stream (Smoothed)')
    else:
        plt.title(f'{emotion.capitalize()} Emotion Throughout Stream')

    plt.xlabel('Stream Time (seconds)')
    plt.ylabel('Emotion Intensity')
    plt.legend()
    plt.grid(True)

    # Add a bit of padding to y-axis
    plt.ylim(-0.05, 1.05)

    # Format x-axis as minutes:seconds
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"{video_id}_{emotion}_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path

def plot_all_emotions(integrated_df, output_dir, video_id, smooth=True, window_size=15):
    """
    Plot separate comparisons for each emotion

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        output_dir (str): Directory to save the plots
        video_id (str): Video ID for filename
        smooth (bool): Whether to apply smoothing
        window_size (int): Size of the smoothing window
    """
    logger.info("Generating separate emotion comparison plots")

    # Create a directory for emotion plots
    emotions_dir = os.path.join(output_dir, "emotions")
    os.makedirs(emotions_dir, exist_ok=True)

    # List of emotions to plot
    emotions = ['excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral']

    # Plot each emotion separately
    plot_paths = []
    for emotion in emotions:
        try:
            plot_path = plot_emotion_comparison(
                integrated_df, emotion, emotions_dir, video_id,
                smooth=smooth, window_size=window_size
            )
            plot_paths.append(plot_path)
            logger.info(f"{emotion.capitalize()} comparison plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error generating {emotion} comparison plot: {str(e)}")

    # Create a summary plot with small versions of all emotions
    create_emotion_summary(
        integrated_df, emotions, output_dir, video_id,
        smooth=smooth, window_size=window_size
    )

    return plot_paths

def create_emotion_summary(integrated_df, emotions, output_dir, video_id, smooth=True, window_size=15):
    """
    Create a summary plot with emotion comparisons, audio loudness, speech rate, chat message count, and sentiment analysis

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        emotions (list): List of emotions to include (neutral will be ignored)
        output_dir (str): Directory to save the plot
        video_id (str): Video ID for filename
        smooth (bool): Whether to apply smoothing
        window_size (int): Size of the smoothing window
    """
    # Filter out 'neutral' from emotions list
    emotions = [e for e in emotions if e != 'neutral']

    # Calculate rows and columns for subplot grid
    n_emotions = len(emotions)
    n_technical = 5  # Audio loudness, sentiment analysis, highlight score, speech rate, chat message count
    n_total = n_emotions + n_technical
    n_cols = 2
    n_rows = (n_total + n_cols - 1) // n_cols  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Flatten axes array for easier indexing
    axes = axes.flatten() if n_rows > 1 else [axes[0], axes[1]]

    # Plot each emotion
    for i, emotion in enumerate(emotions):
        ax = axes[i]

        # Define column names for this emotion
        streamer_col = emotion
        audience_col = f'chat_{emotion}'
        fused_col = f'fused_{emotion}'

        # Get time values
        x = integrated_df['start_time'].values

        # Get and potentially smooth the emotion values
        streamer_y = integrated_df[streamer_col].values
        audience_y = integrated_df[audience_col].values
        fused_y = integrated_df[fused_col].values

        if smooth:
            # Apply smoothing
            _, streamer_y = apply_smoothing(x, streamer_y, window_size, 'savgol')
            _, audience_y = apply_smoothing(x, audience_y, window_size, 'savgol')
            _, fused_y = apply_smoothing(x, fused_y, window_size, 'savgol')

        # Plot each source
        ax.plot(x, streamer_y, 'b-',
                label='Streamer', linewidth=1.5)
        ax.plot(x, audience_y, 'g-',
                label='Audience', linewidth=1.5)
        ax.plot(x, fused_y, 'r-',
                label='Fused', linewidth=2)

        # Set title and labels
        ax.set_title(f'{emotion.capitalize()}')
        ax.set_xlabel('Time (min:sec)')
        ax.set_ylabel('Intensity')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True)

        # Add a bit of padding to y-axis
        ax.set_ylim(-0.05, 1.05)

        # Format x-axis as minutes:seconds
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))

    # Plot audio loudness
    ax_loudness = axes[n_emotions]

    # Try to get audio file path from file manager first, then fallback to hardcoded paths
    try:
        from utils.file_manager import FileManager
        file_manager = FileManager()
        audio_file_path = file_manager.get_local_path("audio")
    except:
        # Fallback to multiple possible paths
        possible_audio_paths = [
            f"/tmp/output/Raw/Audio/audio_{video_id}.wav",
            f"/tmp/output/Raw/audio/audio_{video_id}.wav",
            f"output/Raw/Audio/audio_{video_id}.wav",
            f"Output/Raw/Audio/audio_{video_id}.wav"
        ]
        audio_file_path = None
        for path in possible_audio_paths:
            if os.path.exists(path):
                audio_file_path = path
                break
        if audio_file_path is None:
            audio_file_path = possible_audio_paths[0]  # Use first as default

    waveform_times, waveform_amplitudes = extract_audio_waveform(audio_file_path, num_points=2000)

    if len(waveform_times) > 0 and len(waveform_amplitudes) > 0:
        # Convert to absolute values to get loudness
        loudness = np.abs(waveform_amplitudes)

        # Apply smoothing with rolling window
        window_size_loudness = 50  # Adjust based on desired smoothness
        if len(loudness) > window_size_loudness:
            # Use pandas rolling window for smoothing
            loudness_series = pd.Series(loudness)
            smoothed_loudness = loudness_series.rolling(window=window_size_loudness, center=True).mean()
            # Fill NaN values at the edges
            smoothed_loudness = smoothed_loudness.bfill().ffill()
        else:
            smoothed_loudness = loudness

        # Plot the raw loudness as a light fill
        ax_loudness.fill_between(waveform_times, 0, loudness, alpha=0.3, color='lightblue', label='Raw Loudness')

        # Plot the smoothed loudness as a solid line
        ax_loudness.plot(waveform_times, smoothed_loudness, 'b-', linewidth=1.5, label='Smoothed Loudness')

        # Calculate and plot the average loudness
        avg_loudness = np.mean(loudness)
        ax_loudness.axhline(y=avg_loudness, color='r', linestyle='--',
                    label=f'Avg: {avg_loudness:.2f}')

        ax_loudness.set_title('Audio Loudness')
        ax_loudness.set_xlabel('Time (min:sec)')
        ax_loudness.set_ylabel('Loudness')
        ax_loudness.legend(loc='upper right', fontsize='small')
        ax_loudness.grid(True)
        ax_loudness.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))
    else:
        ax_loudness.text(0.5, 0.5, 'Audio loudness data not available',
                         horizontalalignment='center', verticalalignment='center')
        ax_loudness.set_title('Audio Loudness')

    # Plot sentiment analysis
    ax_sentiment = axes[n_emotions + 1]
    if 'sentiment_score' in integrated_df.columns and 'fused_sentiment' in integrated_df.columns:
        x = integrated_df['start_time'].values
        audio_sentiment = integrated_df['sentiment_score'].values
        chat_sentiment = integrated_df['chat_sentiment'].values if 'chat_sentiment' in integrated_df.columns else np.zeros_like(audio_sentiment)
        fused_sentiment = integrated_df['fused_sentiment'].values

        if smooth:
            # Apply smoothing
            _, audio_sentiment = apply_smoothing(x, audio_sentiment, window_size, 'savgol')
            _, chat_sentiment = apply_smoothing(x, chat_sentiment, window_size, 'savgol')
            _, fused_sentiment = apply_smoothing(x, fused_sentiment, window_size, 'savgol')

        # Plot sentiment scores
        ax_sentiment.plot(x, audio_sentiment, 'b-', label='Streamer', linewidth=1.5)
        ax_sentiment.plot(x, chat_sentiment, 'g-', label='Audience', linewidth=1.5)
        ax_sentiment.plot(x, fused_sentiment, 'r-', label='Fused', linewidth=2)

        # Add a horizontal line at zero
        ax_sentiment.axhline(y=0, color='k', linestyle='-', alpha=0.2)

        ax_sentiment.set_title('Sentiment Analysis')
        ax_sentiment.set_xlabel('Time (min:sec)')
        ax_sentiment.set_ylabel('Sentiment (-1 to 1)')
        ax_sentiment.legend(loc='upper right', fontsize='small')
        ax_sentiment.grid(True)
        ax_sentiment.set_ylim(-1.05, 1.05)
        ax_sentiment.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))
    else:
        ax_sentiment.text(0.5, 0.5, 'Sentiment data not available',
                         horizontalalignment='center', verticalalignment='center')
        ax_sentiment.set_title('Sentiment Analysis')



    # Plot highlight scores
    ax_highlight = axes[n_emotions + 2]
    if 'highlight_score' in integrated_df.columns and 'chat_highlight_score' in integrated_df.columns and 'enhanced_highlight_score' in integrated_df.columns:
        x = integrated_df['start_time'].values
        audio_highlight = integrated_df['highlight_score'].values
        chat_highlight = integrated_df['chat_highlight_score'].values
        enhanced_highlight = integrated_df['enhanced_highlight_score'].values

        if smooth:
            # Apply smoothing
            _, audio_highlight = apply_smoothing(x, audio_highlight, window_size, 'savgol')
            _, chat_highlight = apply_smoothing(x, chat_highlight, window_size, 'savgol')
            _, enhanced_highlight = apply_smoothing(x, enhanced_highlight, window_size, 'savgol')

        # Plot highlight scores
        ax_highlight.plot(x, audio_highlight, 'b-', label='Audio', linewidth=1.5)
        ax_highlight.plot(x, chat_highlight, 'g-', label='Chat', linewidth=1.5)
        ax_highlight.plot(x, enhanced_highlight, 'r-', label='Enhanced', linewidth=2)

        # Calculate threshold for "good" highlights (e.g., top 15% of scores)
        highlight_threshold = np.percentile(enhanced_highlight, 85)

        # Mark the threshold line
        ax_highlight.axhline(y=highlight_threshold, color='gray', linestyle='--',
                           label=f'Threshold (85th)')

        ax_highlight.set_title('Highlight Scores')
        ax_highlight.set_xlabel('Time (min:sec)')
        ax_highlight.set_ylabel('Score')
        ax_highlight.legend(loc='upper right', fontsize='small')
        ax_highlight.grid(True)
        ax_highlight.set_ylim(-0.05, 1.05)
        ax_highlight.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))
    else:
        ax_highlight.text(0.5, 0.5, 'Highlight score data not available',
                         horizontalalignment='center', verticalalignment='center')
        ax_highlight.set_title('Highlight Scores')

    # Plot speech rate
    ax_speech_rate = axes[n_emotions + 3]
    if 'speech_rate' in integrated_df.columns:
        x = integrated_df['start_time'].values
        speech_rate = integrated_df['speech_rate'].values
        if smooth:
            _, speech_rate = apply_smoothing(x, speech_rate, window_size, 'savgol')

        # Calculate average speech rate for reference
        avg_speech_rate = np.mean(speech_rate)

        ax_speech_rate.plot(x, speech_rate, 'b-', linewidth=1.5)
        ax_speech_rate.axhline(y=avg_speech_rate, color='r', linestyle='--',
                              label=f'Avg: {avg_speech_rate:.2f} words/sec')
        ax_speech_rate.set_title('Speech Rate')
        ax_speech_rate.set_xlabel('Time (min:sec)')
        ax_speech_rate.set_ylabel('Words per Second')
        ax_speech_rate.legend(loc='upper right', fontsize='small')
        ax_speech_rate.grid(True)
        ax_speech_rate.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))
    else:
        ax_speech_rate.text(0.5, 0.5, 'Speech rate data not available',
                           horizontalalignment='center', verticalalignment='center')
        ax_speech_rate.set_title('Speech Rate')

    # Plot chat message count
    ax_chat_count = axes[n_emotions + 4]
    if 'chat_volume' in integrated_df.columns:
        message_count = integrated_df['chat_volume'].values
        if smooth:
            _, message_count = apply_smoothing(x, message_count, window_size, 'savgol')

        ax_chat_count.plot(x, message_count, 'g-', linewidth=1.5)
        ax_chat_count.set_title('Chat Activity')
        ax_chat_count.set_xlabel('Time (min:sec)')
        ax_chat_count.set_ylabel('Message Count')
        ax_chat_count.grid(True)
        ax_chat_count.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))
    else:
        ax_chat_count.text(0.5, 0.5, 'Chat message count not available',
                          horizontalalignment='center', verticalalignment='center')
        ax_chat_count.set_title('Chat Activity')

    # Hide any unused subplots
    for i in range(n_emotions + n_technical, len(axes)):
        axes[i].set_visible(False)

    # Add a title to the entire figure
    if smooth:
        fig.suptitle('Stream Analysis Summary (Smoothed)', fontsize=16)
    else:
        fig.suptitle('Stream Analysis Summary', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle

    # Save the plot
    plot_path = os.path.join(output_dir, f"{video_id}_emotion_summary.png")
    plt.savefig(plot_path, dpi=150)  # Higher DPI for better quality
    plt.close()

    logger.info(f"Emotion summary plot saved to {plot_path}")
    return plot_path

def plot_fused_emotions(integrated_df, output_dir, video_id, smooth=True, window_size=15):
    """
    Plot the fused emotions over time and create separate emotion comparison plots

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        output_dir (str): Directory to save the plot
        video_id (str): Video ID for filename
        smooth (bool): Whether to apply smoothing
        window_size (int): Size of the smoothing window
    """
    # Create separate emotion comparison plots
    plot_all_emotions(integrated_df, output_dir, video_id, smooth=smooth, window_size=window_size)

    # Also create the original combined plot for reference
    logger.info("Generating combined fused emotions plot")

    plt.figure(figsize=(15, 8))

    # Extract emotion columns
    emotion_cols = ['fused_excitement', 'fused_funny', 'fused_happiness',
                   'fused_anger', 'fused_sadness', 'fused_neutral']

    # Get time values
    x = integrated_df['start_time'].values

    # Plot each emotion
    for col in emotion_cols:
        emotion_name = col.replace('fused_', '')
        y = integrated_df[col].values

        if smooth:
            # Apply smoothing
            _, y_smooth = apply_smoothing(x, y, window_size, 'savgol')
            plt.plot(x, y_smooth, label=emotion_name, linewidth=2)
        else:
            plt.plot(x, y, label=emotion_name, linewidth=1.5)

    # Add title and labels
    if smooth:
        plt.title('All Fused Emotions Throughout Stream (Smoothed)')
    else:
        plt.title('All Fused Emotions Throughout Stream')

    plt.xlabel('Stream Time (min:sec)')
    plt.ylabel('Emotion Intensity')
    plt.legend()
    plt.grid(True)

    # Format x-axis as minutes:seconds
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"{video_id}_fused_emotions_combined.png")
    plt.savefig(plot_path, dpi=150)  # Higher DPI for better quality
    plt.close()

    logger.info(f"Combined fused emotions plot saved to {plot_path}")

def create_editors_highlight_view(integrated_df, output_dir, video_id, num_highlights=15, window_size=15):
    """
    Create a visualization specifically designed for video editors to identify highlight moments

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        output_dir (str): Directory to save the plot
        video_id (str): Video ID for filename
        num_highlights (int): Number of top highlights to mark
        window_size (int): Size of the smoothing window
    """
    logger.info("Generating editor's highlight view")

    # Create a directory for editor views
    editor_dir = os.path.join(output_dir, "editor_view")
    os.makedirs(editor_dir, exist_ok=True)

    # Get top highlights
    top_highlights = integrated_df.nlargest(num_highlights, 'enhanced_highlight_score')

    # Create the main figure
    plt.figure(figsize=(20, 12))

    # Get time values and highlight scores
    x = integrated_df['start_time'].values
    highlight_scores = integrated_df['enhanced_highlight_score'].values

    # Apply smoothing to highlight scores
    _, smoothed_scores = apply_smoothing(x, highlight_scores, window_size, 'savgol')

    # Plot the smoothed highlight scores
    plt.plot(x, smoothed_scores, 'b-', linewidth=2.5, label='Enhanced Highlight Score')

    # Calculate threshold for "good" highlights (e.g., top 15% of scores)
    highlight_threshold = np.percentile(smoothed_scores, 85)

    # Shade regions above the threshold
    plt.fill_between(x, highlight_threshold, smoothed_scores,
                    where=(smoothed_scores > highlight_threshold),
                    color='lightblue', alpha=0.5, label='Highlight Regions')

    # Mark the threshold line
    plt.axhline(y=highlight_threshold, color='gray', linestyle='--',
               label=f'Highlight Threshold (85th percentile)')

    # Mark top highlights with vertical lines and annotations
    for i, highlight in enumerate(top_highlights.itertuples()):
        # Add vertical line at highlight
        plt.axvline(x=highlight.start_time, color='red', linestyle='-', alpha=0.7)

        # Format timestamp as MM:SS
        minutes = int(highlight.start_time // 60)
        seconds = int(highlight.start_time % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"

        # Add annotation with timestamp and score
        score_text = f"{highlight.enhanced_highlight_score:.2f}"
        plt.annotate(f"#{i+1}: {timestamp}\nScore: {score_text}",
                    xy=(highlight.start_time, highlight.enhanced_highlight_score),
                    xytext=(10, (-1)**i * 20),  # Alternate above/below to avoid overlap
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    # Add dominant emotions at top highlights
    for i, highlight in enumerate(top_highlights.itertuples()):
        # Find dominant emotion
        emotions = {
            'excitement': highlight.fused_excitement if hasattr(highlight, 'fused_excitement') else 0,
            'funny': highlight.fused_funny if hasattr(highlight, 'fused_funny') else 0,
            'happiness': highlight.fused_happiness if hasattr(highlight, 'fused_happiness') else 0,
            'anger': highlight.fused_anger if hasattr(highlight, 'fused_anger') else 0,
            'sadness': highlight.fused_sadness if hasattr(highlight, 'fused_sadness') else 0
        }
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]

        # Add emotion annotation
        y_pos = highlight.enhanced_highlight_score - 0.05
        plt.annotate(f"{dominant_emotion.capitalize()}",
                    xy=(highlight.start_time, y_pos),
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # Add stream timeline markers (every 5 minutes)
    max_time = integrated_df['end_time'].max()
    for minute in range(0, int(max_time/60) + 1, 5):
        time_sec = minute * 60
        if time_sec <= max_time:
            plt.axvline(x=time_sec, color='gray', linestyle=':', alpha=0.5)
            plt.text(time_sec, -0.05, f"{minute:02d}:00",
                    ha='center', va='top', fontsize=9, alpha=0.7)

    # Set title and labels
    plt.title(f"Editor's Highlight View - {video_id}", fontsize=16)
    plt.xlabel('Stream Time (minutes:seconds)', fontsize=12)
    plt.ylabel('Highlight Score', fontsize=12)

    # Format x-axis as minutes:seconds
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60):02d}:{int(x%60):02d}'))

    # Set y-axis limits with some padding
    plt.ylim(-0.1, 1.1)

    # Add legend
    plt.legend(loc='upper right')

    # Add grid for easier reading
    plt.grid(True, alpha=0.3)

    # Add text box with instructions
    instruction_text = (
        "Editor's Guide:\n"
        "• Red vertical lines mark top highlight moments\n"
        "• Blue shaded regions are potential highlight areas\n"
        "• Annotations show timestamp and dominant emotion\n"
        "• Higher scores indicate stronger highlight potential"
    )
    plt.figtext(0.02, 0.02, instruction_text, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(editor_dir, f"{video_id}_editors_view.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Editor's highlight view saved to {plot_path}")

    # Create a CSV file with detailed highlight information for editors
    create_editors_highlight_csv(top_highlights, editor_dir, video_id)

    # Create an interactive HTML version
    create_interactive_editor_view(integrated_df, top_highlights, editor_dir, video_id)

    return plot_path

def hex_to_rgb(hex_color):
    """
    Convert hex color to RGB values

    Args:
        hex_color (str): Hex color code (e.g., '#FF0000' or 'red')

    Returns:
        tuple: (r, g, b) values from 0-255
    """
    # Handle named colors
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'gray': (128, 128, 128)
    }

    if hex_color in color_map:
        return color_map[hex_color]

    # Handle hex colors
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    # Convert hex to RGB
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)

    # Default fallback
    return (128, 128, 128)

def extract_audio_waveform(audio_file_path, num_points=1000):
    """
    Extract audio waveform data from an audio file

    Args:
        audio_file_path (str): Path to the audio file
        num_points (int): Number of points to extract for visualization

    Returns:
        tuple: (times, amplitudes) arrays for plotting
    """
    import numpy as np

    try:
        import librosa

        # Load audio file
        logger.info(f"Loading audio file for waveform extraction: {audio_file_path}")
        y, sr = librosa.load(audio_file_path, sr=None)

        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)

        # Resample to desired number of points
        if len(y) > num_points:
            # Create time array
            times = np.linspace(0, duration, num_points)

            # Resample amplitude data
            indices = np.floor(np.linspace(0, len(y) - 1, num_points)).astype(int)
            amplitudes = y[indices]

            # Normalize amplitudes to range [-1, 1]
            if np.max(np.abs(amplitudes)) > 0:
                amplitudes = amplitudes / np.max(np.abs(amplitudes))
        else:
            # If audio is shorter than desired points, use as is
            times = np.linspace(0, duration, len(y))
            amplitudes = y

            # Normalize amplitudes
            if np.max(np.abs(amplitudes)) > 0:
                amplitudes = amplitudes / np.max(np.abs(amplitudes))

        logger.info(f"Extracted waveform with {len(times)} points")
        return times, amplitudes

    except ImportError:
        logger.warning("librosa not installed. Cannot extract audio waveform.")
        # Create a simple sine wave as a placeholder
        times = np.linspace(0, 60, num_points)  # 1 minute of audio
        amplitudes = np.zeros(num_points)  # Flat line
        return times, amplitudes
    except Exception as e:
        logger.error(f"Error extracting audio waveform: {str(e)}")
        times = np.linspace(0, 60, num_points)  # 1 minute of audio
        amplitudes = np.zeros(num_points)  # Flat line
        return times, amplitudes

def create_interactive_editor_view(integrated_df, top_highlights, output_dir, video_id):
    """
    Create an interactive HTML visualization for video editors

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        top_highlights (DataFrame): Top highlights dataframe
        output_dir (str): Directory to save the HTML
        video_id (str): Video ID for filename
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        logger.info("Generating interactive editor's view")

        # Get time values and highlight scores
        x = integrated_df['start_time'].values
        highlight_scores = integrated_df['enhanced_highlight_score'].values

        # Apply smoothing to highlight scores
        window_size = 15
        _, smoothed_scores = apply_smoothing(x, highlight_scores, window_size, 'savgol')

        # Calculate threshold for "good" highlights (e.g., top 15% of scores)
        highlight_threshold = np.percentile(smoothed_scores, 85)

        # Extract audio waveform - try multiple possible paths
        possible_audio_paths = [
            f"/tmp/output/Raw/Audio/audio_{video_id}.wav",  # /tmp prefix with uppercase 'Audio'
            f"/tmp/output/Raw/audio/audio_{video_id}.wav",  # /tmp prefix with lowercase 'audio'
            f"output/Raw/Audio/audio_{video_id}.wav",       # relative path with uppercase 'Audio'
            f"output/Raw/audio/audio_{video_id}.wav",       # relative path with lowercase 'audio'
            f"Output/Raw/Audio/audio_{video_id}.wav",       # legacy uppercase 'Output'
            f"Output/Raw/audio/audio_{video_id}.wav",       # legacy lowercase 'audio'
            f"Output/Raw/audio_{video_id}.mp3"              # mp3 format
        ]

        audio_file_path = None
        for path in possible_audio_paths:
            if os.path.exists(path):
                audio_file_path = path
                logger.info(f"Found audio file for waveform extraction at: {audio_file_path}")
                break

        if audio_file_path is None:
            logger.warning(f"Audio file not found in any of the expected locations for waveform extraction")
            audio_file_path = possible_audio_paths[0]  # Use the first path as default even if it doesn't exist

        waveform_times, waveform_amplitudes = extract_audio_waveform(audio_file_path)

        # Create figure with subplots - one for highlight score, one for waveform
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Highlight Score", "Audio Waveform")
        )

        # Add highlight score trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=smoothed_scores,
                mode='lines',
                name='Highlight Score',
                line=dict(color='blue', width=3),
                hovertemplate='Time: %{text}<br>Score: %{y:.2f}<extra></extra>',
                text=[f"{int(t//60):02d}:{int(t%60):02d}" for t in x]
            ),
            row=1, col=1
        )

        # Add threshold line
        fig.add_trace(
            go.Scatter(
                x=[x[0], x[-1]],
                y=[highlight_threshold, highlight_threshold],
                mode='lines',
                name='Highlight Threshold',
                line=dict(color='gray', width=2, dash='dash')
            ),
            row=1, col=1
        )

        # Add audio waveform if available
        if len(waveform_times) > 0 and len(waveform_amplitudes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=waveform_times,
                    y=waveform_amplitudes,
                    mode='lines',
                    name='Audio Waveform',
                    line=dict(color='green', width=1),
                    hoverinfo='skip'
                ),
                row=2, col=1
            )

        # Add markers for top highlights
        highlight_times = []
        highlight_scores = []
        highlight_texts = []
        highlight_emotions = []
        highlight_colors = {
            'excitement': 'red',
            'funny': 'orange',
            'happiness': 'green',
            'anger': 'purple',
            'sadness': 'blue',
            'neutral': 'gray'
        }
        colors = []

        # Ensure we have at least one highlight for each emotion
        emotion_highlights = {emotion: [] for emotion in highlight_colors.keys()}

        for i, highlight in enumerate(top_highlights.itertuples()):
            # Format timestamp
            minutes = int(highlight.start_time // 60)
            seconds = int(highlight.start_time % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            # Get text content (truncated if too long)
            text = getattr(highlight, 'text', '')
            if text and len(text) > 50:
                text = text[:47] + "..."

            # Find dominant emotion
            emotions = {
                'excitement': highlight.fused_excitement if hasattr(highlight, 'fused_excitement') else 0,
                'funny': highlight.fused_funny if hasattr(highlight, 'fused_funny') else 0,
                'happiness': highlight.fused_happiness if hasattr(highlight, 'fused_happiness') else 0,
                'anger': highlight.fused_anger if hasattr(highlight, 'fused_anger') else 0,
                'sadness': highlight.fused_sadness if hasattr(highlight, 'fused_sadness') else 0
            }
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]

            # Add to emotion-specific list
            emotion_highlights[dominant_emotion].append((highlight, i))

            # Add to lists
            highlight_times.append(highlight.start_time)
            highlight_scores.append(highlight.enhanced_highlight_score)
            highlight_texts.append(f"#{i+1}: {timestamp}<br>{text}")
            highlight_emotions.append(dominant_emotion)
            colors.append(highlight_colors.get(dominant_emotion, 'gray'))

        # Check if we're missing any emotions and find the best candidate for each
        for emotion, highlights in emotion_highlights.items():
            if not highlights:
                # Find the best candidate for this emotion in the full dataset
                emotion_col = f'fused_{emotion}'
                if emotion_col in integrated_df.columns:
                    # Get top 3 segments for this emotion
                    top_emotion_segments = integrated_df.nlargest(3, emotion_col)

                    if not top_emotion_segments.empty:
                        best_segment = top_emotion_segments.iloc[0]

                        # Format timestamp
                        minutes = int(best_segment['start_time'] // 60)
                        seconds = int(best_segment['start_time'] % 60)
                        timestamp = f"{minutes:02d}:{seconds:02d}"

                        # Get text content
                        text = best_segment['text'] if 'text' in best_segment else ''
                        if text and len(text) > 50:
                            text = text[:47] + "..."

                        # Add to lists
                        highlight_times.append(best_segment['start_time'])
                        highlight_scores.append(best_segment['enhanced_highlight_score'])
                        highlight_texts.append(f"Best {emotion.capitalize()}: {timestamp}<br>{text}")
                        highlight_emotions.append(emotion)
                        colors.append(highlight_colors.get(emotion, 'gray'))

                        logger.info(f"Added best {emotion} highlight at {timestamp}")

        # Add highlight markers to the highlight score plot
        fig.add_trace(
            go.Scatter(
                x=highlight_times,
                y=highlight_scores,
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors,
                    line=dict(width=2, color='black')
                ),
                name='Top Highlights',
                text=highlight_texts,
                hovertemplate='%{text}<br>Emotion: %{customdata}<br>Score: %{y:.2f}<extra></extra>',
                customdata=highlight_emotions
            ),
            row=1, col=1
        )

        # Add vertical lines at highlight points that extend to the waveform
        for i, time in enumerate(highlight_times):
            fig.add_shape(
                type="line",
                x0=time, x1=time,
                y0=0, y1=1,
                yref="paper",
                line=dict(
                    color=colors[i],
                    width=2,
                    dash="dot",
                ),
                opacity=0.7,
                layer="below"
            )

        # Update layout
        fig.update_layout(
            title=f"Interactive Editor's Highlight View - {video_id}",
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            height=800,  # Increased height for better waveform visibility
            template="plotly_white"
        )

        # Update xaxis for both subplots
        fig.update_xaxes(
            title="Stream Time",
            tickformat="%M:%S",
            tickmode='array',
            tickvals=[i*60 for i in range(int(x[-1]/60)+1) if i % 5 == 0],  # Every 5 minutes
            ticktext=[f"{i:02d}:00" for i in range(int(x[-1]/60)+1) if i % 5 == 0],
            row=2, col=1  # Only add title to bottom subplot
        )

        # Update yaxis for highlight score plot
        fig.update_yaxes(
            title="Highlight Score",
            range=[-0.05, 1.05],
            row=1, col=1
        )

        # Update yaxis for waveform plot
        fig.update_yaxes(
            title="Audio Amplitude",
            range=[-1.05, 1.05],
            row=2, col=1
        )

        # Add one-minute segment markers
        max_time = x[-1]
        for minute in range(0, int(max_time/60) + 1):
            time_sec = minute * 60
            if time_sec <= max_time:
                # Add vertical line for each minute
                fig.add_shape(
                    type="line",
                    x0=time_sec, x1=time_sec,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(
                        color="rgba(150,150,150,0.5)",
                        width=1,
                        dash="dot",
                    ),
                    layer="below"
                )

                # Add minute label every 5 minutes
                if minute % 5 == 0:
                    fig.add_annotation(
                        x=time_sec,
                        y=0,
                        yref="paper",
                        text=f"{minute:02d}:00",
                        showarrow=False,
                        font=dict(size=10),
                        opacity=0.7
                    )

        # Add shapes for highlight regions
        for i, score in enumerate(smoothed_scores):
            if score > highlight_threshold:
                # Find continuous regions above threshold
                start_idx = i
                while i < len(smoothed_scores) and smoothed_scores[i] > highlight_threshold:
                    i += 1
                end_idx = i - 1

                # Only add if region is significant (at least 3 seconds)
                if x[end_idx] - x[start_idx] >= 3:
                    fig.add_shape(
                        type="rect",
                        x0=x[start_idx],
                        x1=x[end_idx],
                        y0=highlight_threshold,
                        y1=max(smoothed_scores[start_idx:end_idx+1]),
                        fillcolor="lightblue",
                        opacity=0.3,
                        layer="below",
                        row=1, col=1
                    )

        # Add one-minute segment analysis
        # Group data into one-minute segments
        if len(x) > 0:
            max_time = int(np.ceil(x[-1] / 60)) * 60
            minute_segments = []

            for minute in range(0, int(max_time/60)):
                start_time = minute * 60
                end_time = (minute + 1) * 60

                # Find all data points in this minute
                segment_indices = np.where((x >= start_time) & (x < end_time))[0]

                if len(segment_indices) > 0:
                    # Calculate average highlight score for this minute
                    avg_score = np.mean(smoothed_scores[segment_indices])

                    # Find dominant emotion for this minute
                    emotion_cols = ['fused_excitement', 'fused_funny', 'fused_happiness',
                                   'fused_anger', 'fused_sadness', 'fused_neutral']

                    emotion_values = {}
                    for col in emotion_cols:
                        if col in integrated_df.columns:
                            emotion_values[col.replace('fused_', '')] = integrated_df.iloc[segment_indices][col].mean()

                    dominant_emotion = max(emotion_values.items(), key=lambda x: x[1])[0] if emotion_values else 'neutral'

                    # Add to minute segments
                    minute_segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'avg_score': avg_score,
                        'dominant_emotion': dominant_emotion,
                        'color': highlight_colors.get(dominant_emotion, 'gray')
                    })

            # Add minute segment markers
            for segment in minute_segments:
                # Only add marker if score is above threshold
                if segment['avg_score'] > highlight_threshold * 0.8:  # Slightly lower threshold for minute segments
                    # Add colored background for this minute
                    fig.add_shape(
                        type="rect",
                        x0=segment['start_time'],
                        x1=segment['end_time'],
                        y0=-1,
                        y1=1,
                        fillcolor=f"rgba({','.join(map(str, hex_to_rgb(segment['color'])))},.15)",
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        row=2, col=1
                    )

                    # Add minute label with emotion
                    minute = int(segment['start_time'] / 60)
                    fig.add_annotation(
                        x=(segment['start_time'] + segment['end_time']) / 2,
                        y=-0.8,
                        text=f"{minute:02d}:00-{minute:02d}:59<br>{segment['dominant_emotion'].capitalize()}",
                        showarrow=False,
                        font=dict(size=8, color=segment['color']),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor=segment['color'],
                        borderwidth=1,
                        borderpad=2,
                        row=2, col=1
                    )

        # Add annotations for instructions
        fig.add_annotation(
            text="Click on a highlight marker to see details",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12)
        )

        # Create HTML with embedded JavaScript for Twitch timestamp links
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Editor's Highlight View - {video_id}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1 {{ color: #333; }}
                .highlight-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .highlight-table th, .highlight-table td {{
                    padding: 8px; text-align: left; border-bottom: 1px solid #ddd;
                }}
                .highlight-table tr:hover {{ background-color: #f5f5f5; }}
                .highlight-table th {{ background-color: #4CAF50; color: white; }}
                .timestamp {{ color: blue; cursor: pointer; text-decoration: underline; }}
                .emotion-excitement {{ color: red; font-weight: bold; }}
                .emotion-funny {{ color: orange; font-weight: bold; }}
                .emotion-happiness {{ color: green; font-weight: bold; }}
                .emotion-anger {{ color: purple; font-weight: bold; }}
                .emotion-sadness {{ color: blue; font-weight: bold; }}
                .emotion-neutral {{ color: gray; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Editor's Highlight View - {video_id}</h1>
                <div id="plot"></div>

                <h2>Top Highlights</h2>
                <table class="highlight-table">
                    <tr>
                        <th>Rank</th>
                        <th>Timestamp</th>
                        <th>Duration</th>
                        <th>Score</th>
                        <th>Emotion</th>
                        <th>Content</th>
                    </tr>
        """

        # Add table rows for each highlight
        for i, highlight in enumerate(top_highlights.itertuples()):
            # Format timestamps
            start_minutes = int(highlight.start_time // 60)
            start_seconds = int(highlight.start_time % 60)
            start_timestamp = f"{start_minutes:02d}:{start_seconds:02d}"

            end_minutes = int(highlight.end_time // 60)
            end_seconds = int(highlight.end_time % 60)
            end_timestamp = f"{end_minutes:02d}:{end_seconds:02d}"

            # Calculate duration
            duration = highlight.end_time - highlight.start_time
            duration_text = f"{int(duration // 60):02d}:{int(duration % 60):02d}"

            # Get text content
            text = getattr(highlight, 'text', '')

            # Find dominant emotion
            emotions = {
                'excitement': highlight.fused_excitement if hasattr(highlight, 'fused_excitement') else 0,
                'funny': highlight.fused_funny if hasattr(highlight, 'fused_funny') else 0,
                'happiness': highlight.fused_happiness if hasattr(highlight, 'fused_happiness') else 0,
                'anger': highlight.fused_anger if hasattr(highlight, 'fused_anger') else 0,
                'sadness': highlight.fused_sadness if hasattr(highlight, 'fused_sadness') else 0
            }
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]

            # Add table row
            html_content += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td><span class="timestamp" onclick="openTwitchTimestamp({int(highlight.start_time)})">{start_timestamp} - {end_timestamp}</span></td>
                        <td>{duration_text}</td>
                        <td>{highlight.enhanced_highlight_score:.2f}</td>
                        <td class="emotion-{dominant_emotion}">{dominant_emotion.capitalize()}</td>
                        <td>{text}</td>
                    </tr>
            """

        # Complete the HTML
        html_content += f"""
                </table>
            </div>

            <script>
                var plotData = {fig.to_json()};
                Plotly.newPlot('plot', plotData.data, plotData.layout);

                function openTwitchTimestamp(seconds) {{
                    var url = "https://www.twitch.tv/videos/{video_id}?t=" + seconds + "s";
                    window.open(url, '_blank');
                }}

                // Add click handler to plot points
                document.getElementById('plot').on('plotly_click', function(data) {{
                    if (data.points[0].curveNumber === 2) {{ // Top Highlights trace
                        var pointIndex = data.points[0].pointIndex;
                        var timestamp = data.points[0].x;
                        openTwitchTimestamp(Math.floor(timestamp));
                    }}
                }});
            </script>
        </body>
        </html>
        """

        # Save the HTML file
        html_path = os.path.join(output_dir, f"{video_id}_editors_view.html")
        with open(html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Interactive editor's view saved to {html_path}")
        return html_path

    except ImportError:
        logger.warning("Plotly not installed. Skipping interactive editor view.")
        return None

def create_editors_highlight_csv(top_highlights, output_dir, video_id):
    """
    Create a CSV file with detailed highlight information for video editors

    Args:
        top_highlights (DataFrame): Top highlights dataframe
        output_dir (str): Directory to save the CSV
        video_id (str): Video ID for filename
    """
    # Check if top_highlights is empty
    if top_highlights.empty:
        logger.warning("No highlights found for editor's CSV")
        # Create a simple CSV with headers but no data
        editors_df = pd.DataFrame(columns=[
            'rank', 'start_time', 'end_time', 'duration',
            'start_timestamp', 'end_timestamp', 'highlight_score',
            'content', 'dominant_emotion', 'audience_reaction',
            'streamer_emotion_intensity'
        ])
        csv_path = os.path.join(output_dir, f"{video_id}_editors_highlights.csv")
        editors_df.to_csv(csv_path, index=False)
        logger.info(f"Empty editor's highlight CSV saved to {csv_path}")
        return csv_path

    # Create a dataframe with the information editors need
    editors_df = pd.DataFrame()

    # Reset index to make sure we can iterate properly
    top_highlights = top_highlights.reset_index(drop=True)

    # Add columns for timestamp, duration, score, and text
    editors_df['rank'] = range(1, len(top_highlights) + 1)
    editors_df['start_time'] = top_highlights['start_time'].values
    editors_df['end_time'] = top_highlights['end_time'].values
    editors_df['duration'] = top_highlights['end_time'].values - top_highlights['start_time'].values

    # Format timestamps as MM:SS, handling NaN values
    def format_timestamp(seconds):
        if pd.isna(seconds):
            return "00:00"
        return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

    editors_df['start_timestamp'] = editors_df['start_time'].apply(format_timestamp)
    editors_df['end_timestamp'] = editors_df['end_time'].apply(format_timestamp)

    # Add highlight score
    editors_df['highlight_score'] = top_highlights['enhanced_highlight_score'].values

    # Add text content
    editors_df['content'] = top_highlights['text'].values

    # Add dominant emotion
    def get_dominant_emotion(row_idx):
        row = top_highlights.iloc[row_idx]
        emotions = {
            'excitement': row['fused_excitement'] if 'fused_excitement' in row else 0,
            'funny': row['fused_funny'] if 'fused_funny' in row else 0,
            'happiness': row['fused_happiness'] if 'fused_happiness' in row else 0,
            'anger': row['fused_anger'] if 'fused_anger' in row else 0,
            'sadness': row['fused_sadness'] if 'fused_sadness' in row else 0
        }
        if all(v == 0 for v in emotions.values()):
            return "neutral"
        return max(emotions.items(), key=lambda x: x[1])[0]

    editors_df['dominant_emotion'] = [get_dominant_emotion(i) for i in range(len(top_highlights))]

    # Add audience reaction strength (from chat)
    editors_df['audience_reaction'] = top_highlights['chat_highlight_score'].values

    # Add streamer emotion intensity
    def get_streamer_emotion_intensity(row_idx):
        row = top_highlights.iloc[row_idx]
        emotions = {
            'excitement': row['excitement'] if 'excitement' in row else 0,
            'funny': row['funny'] if 'funny' in row else 0,
            'happiness': row['happiness'] if 'happiness' in row else 0,
            'anger': row['anger'] if 'anger' in row else 0,
            'sadness': row['sadness'] if 'sadness' in row else 0
        }
        if all(v == 0 for v in emotions.values()):
            return 0
        return max(emotions.values())

    editors_df['streamer_emotion_intensity'] = [get_streamer_emotion_intensity(i) for i in range(len(top_highlights))]

    # Move content column to the end for better readability
    content_column = editors_df.pop('content')
    editors_df['content'] = content_column

    # Save to CSV
    csv_path = os.path.join(output_dir, f"{video_id}_editors_highlights.csv")
    editors_df.to_csv(csv_path, index=False)
    logger.info(f"Editor's highlight CSV saved to {csv_path} with columns rearranged for better readability")
    return csv_path

def run_integration(video_id):
    """
    Main function to integrate chat and audio analysis

    Args:
        video_id (str): The video ID to process

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Starting integration for video ID: {video_id}")

    # Initialize file manager
    file_manager = FileManager(video_id)

    # Get file paths using file manager
    audio_sentiment_path = file_manager.get_local_path("audio_sentiment")
    chat_analysis_path = file_manager.get_local_path("chat_sentiment")

    # Check if paths are None and provide fallbacks
    if audio_sentiment_path is None:
        audio_sentiment_path = Path(f"/tmp/output/Analysis/Audio/audio_{video_id}_sentiment.csv")
        logger.warning(f"Using fallback path for audio sentiment: {audio_sentiment_path}")

    if chat_analysis_path is None:
        chat_analysis_path = Path(f"/tmp/output/Analysis/Chat/{video_id}_chat_sentiment.csv")
        logger.warning(f"Using fallback path for chat sentiment: {chat_analysis_path}")

    # Define output directory and path
    output_dir = os.path.join(os.path.dirname(str(audio_sentiment_path)), "..", "Integrated")

    integrated_output_path = file_manager.get_local_path("integrated_analysis")
    if integrated_output_path is None:
        integrated_output_path = Path(f"/tmp/output/Analysis/integrated_{video_id}.json")
        logger.warning(f"Using fallback path for integrated analysis: {integrated_output_path}")

    # Log the paths we're using
    logger.info(f"Using audio sentiment path: {audio_sentiment_path}")
    logger.info(f"Using chat analysis path: {chat_analysis_path}")
    logger.info(f"Using output directory: {output_dir}")
    logger.info(f"Will save integrated analysis to: {integrated_output_path}")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.warning(f"Error creating output directory {output_dir}: {str(e)}")
        # Fallback to a simpler path
        output_dir = os.path.join("/tmp/output/Analysis/Integrated")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using fallback output directory: {output_dir}")

    try:
        os.makedirs(os.path.dirname(str(integrated_output_path)), exist_ok=True)
        logger.info(f"Created directory for integrated output: {os.path.dirname(str(integrated_output_path))}")
    except Exception as e:
        logger.warning(f"Error creating directory for integrated output: {str(e)}")
        # Fallback to a simpler path
        integrated_output_path = Path(f"{output_dir}/integrated_{video_id}.json")
        os.makedirs(os.path.dirname(str(integrated_output_path)), exist_ok=True)
        logger.info(f"Using fallback integrated output path: {integrated_output_path}")

    # Check if input files exist and try to download from GCS if missing
    if not os.path.exists(audio_sentiment_path):
        logger.warning(f"Audio sentiment file not found locally: {audio_sentiment_path}")
        # Try to download from GCS if enabled
        if USE_GCS and file_manager.download_from_gcs("audio_sentiment"):
            audio_sentiment_path = file_manager.get_local_path("audio_sentiment")
            logger.info(f"Downloaded audio sentiment file from GCS: {audio_sentiment_path}")
        else:
            # Try to find the file using the enhanced get_file_path method
            audio_sentiment_path = file_manager.get_file_path("audio_sentiment")
            if audio_sentiment_path and os.path.exists(audio_sentiment_path):
                logger.info(f"Found audio sentiment file at alternative path: {audio_sentiment_path}")
            else:
                # Try alternative paths
                alt_paths = [
                    Path(f"/tmp/output/Analysis/Audio/audio_{video_id}_sentiment.csv"),
                    Path(f"/tmp/output/Analysis/audio/audio_{video_id}_sentiment.csv"),
                    Path(f"output/Analysis/Audio/audio_{video_id}_sentiment.csv"),
                    Path(f"output/Analysis/audio/audio_{video_id}_sentiment.csv")
                ]

                for alt_path in alt_paths:
                    if alt_path.exists():
                        audio_sentiment_path = alt_path
                        logger.info(f"Found audio sentiment file at alternative path: {audio_sentiment_path}")
                        break

                if not os.path.exists(audio_sentiment_path):
                    logger.error(f"Could not find or download audio sentiment file")
                    return False

    if not os.path.exists(chat_analysis_path):
        logger.warning(f"Chat analysis file not found locally: {chat_analysis_path}")
        # Try to download from GCS if enabled
        if USE_GCS and file_manager.download_from_gcs("chat_sentiment"):
            chat_analysis_path = file_manager.get_local_path("chat_sentiment")
            logger.info(f"Downloaded chat analysis file from GCS: {chat_analysis_path}")
        else:
            # Try to find the file using the enhanced get_file_path method
            chat_analysis_path = file_manager.get_file_path("chat_sentiment")
            if chat_analysis_path and os.path.exists(chat_analysis_path):
                logger.info(f"Found chat analysis file at alternative path: {chat_analysis_path}")
            else:
                # Try alternative paths
                alt_paths = [
                    Path(f"/tmp/output/Analysis/Chat/{video_id}_chat_sentiment.csv"),
                    Path(f"/tmp/output/Analysis/chat/{video_id}_chat_sentiment.csv"),
                    Path(f"output/Analysis/Chat/{video_id}_chat_sentiment.csv"),
                    Path(f"output/Analysis/chat/{video_id}_chat_sentiment.csv")
                ]

                for alt_path in alt_paths:
                    if alt_path.exists():
                        chat_analysis_path = alt_path
                        logger.info(f"Found chat analysis file at alternative path: {chat_analysis_path}")
                        break

                if not os.path.exists(chat_analysis_path):
                    logger.warning(f"Could not find or download chat analysis file")
                    # Continue with empty chat data instead of failing
                    logger.warning("Will continue with audio-only analysis")
                    chat_df = pd.DataFrame(columns=['start_time', 'end_time', 'message_count',
                                                  'avg_sentiment', 'avg_highlight',
                                                  'avg_excitement', 'avg_funny', 'avg_happiness',
                                                  'avg_anger', 'avg_sadness', 'avg_neutral'])

    # Load audio sentiment data
    logger.info(f"Loading audio sentiment data from {audio_sentiment_path}")
    try:
        audio_df = pd.read_csv(audio_sentiment_path)
        logger.info(f"Loaded {len(audio_df)} segments from audio sentiment file")

        # Validate audio data
        if audio_df.empty:
            logger.error("Audio sentiment file is empty")
            return False

        # Check for required columns
        required_columns = ['start_time', 'end_time', 'text', 'sentiment_score',
                           'excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral']
        missing_columns = [col for col in required_columns if col not in audio_df.columns]

        if missing_columns:
            logger.error(f"Audio sentiment file missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(audio_df.columns)}")

            # Try to continue with missing columns by adding them with default values
            for col in missing_columns:
                if col in ['start_time', 'end_time']:
                    logger.error(f"Cannot continue without critical column: {col}")
                    return False
                elif col == 'text':
                    audio_df[col] = ""
                    logger.warning(f"Added empty '{col}' column to audio data")
                else:
                    audio_df[col] = 0.0
                    logger.warning(f"Added '{col}' column with default value 0.0 to audio data")
    except Exception as e:
        logger.error(f"Error loading audio sentiment data: {str(e)}")
        logger.error(f"File exists: {os.path.exists(audio_sentiment_path)}")
        if os.path.exists(audio_sentiment_path):
            logger.error(f"File size: {os.path.getsize(audio_sentiment_path)}")
            # Try to read the first few lines to see if it's valid CSV
            try:
                with open(audio_sentiment_path, 'r') as f:
                    first_lines = [next(f) for _ in range(5)]
                logger.error(f"First few lines of file:\n{''.join(first_lines)}")
            except Exception as read_error:
                logger.error(f"Error reading file: {str(read_error)}")
        return False

    # Load chat analysis data if we haven't already created an empty DataFrame
    if 'chat_df' not in locals():
        logger.info(f"Loading chat analysis data from {chat_analysis_path}")
        try:
            if os.path.exists(chat_analysis_path):
                chat_df = pd.read_csv(chat_analysis_path)
                logger.info(f"Loaded {len(chat_df)} segments from chat analysis file")

                # Validate chat data
                if chat_df.empty:
                    logger.warning("Chat analysis file is empty. Will continue with audio-only analysis.")
                    # Create an empty DataFrame with required columns
                    chat_df = pd.DataFrame(columns=['start_time', 'end_time', 'message_count',
                                                   'avg_sentiment', 'avg_highlight',
                                                   'avg_excitement', 'avg_funny', 'avg_happiness',
                                                   'avg_anger', 'avg_sadness', 'avg_neutral'])
                else:
                    # Check for required columns
                    required_columns = ['start_time', 'end_time']
                    missing_columns = [col for col in required_columns if col not in chat_df.columns]

                    if missing_columns:
                        logger.warning(f"Chat analysis file missing required columns: {missing_columns}")
                        logger.info(f"Available columns: {list(chat_df.columns)}")

                        # Cannot continue without start_time and end_time
                        if 'start_time' in missing_columns or 'end_time' in missing_columns:
                            logger.warning("Cannot continue without start_time and end_time columns in chat data")
                            # Create an empty DataFrame with required columns instead of failing
                            chat_df = pd.DataFrame(columns=['start_time', 'end_time', 'message_count',
                                                          'avg_sentiment', 'avg_highlight',
                                                          'avg_excitement', 'avg_funny', 'avg_happiness',
                                                          'avg_anger', 'avg_sadness', 'avg_neutral'])
                            logger.warning("Created empty chat DataFrame with required columns")
            else:
                logger.warning(f"Chat analysis file does not exist: {chat_analysis_path}")
                # Create an empty DataFrame with required columns
                chat_df = pd.DataFrame(columns=['start_time', 'end_time', 'message_count',
                                               'avg_sentiment', 'avg_highlight',
                                               'avg_excitement', 'avg_funny', 'avg_happiness',
                                               'avg_anger', 'avg_sadness', 'avg_neutral'])
                logger.warning("Created empty chat DataFrame with required columns")
        except Exception as e:
            logger.error(f"Error loading chat analysis data: {str(e)}")
            logger.error(f"File exists: {os.path.exists(chat_analysis_path)}")
            if os.path.exists(chat_analysis_path):
                logger.error(f"File size: {os.path.getsize(chat_analysis_path)}")
                # Try to read the first few lines to see if it's valid CSV
                try:
                    with open(chat_analysis_path, 'r') as f:
                        first_lines = [next(f) for _ in range(5)]
                    logger.error(f"First few lines of file:\n{''.join(first_lines)}")
                except Exception as read_error:
                    logger.error(f"Error reading file: {str(read_error)}")

            # Create an empty DataFrame with required columns instead of failing
            logger.warning("Creating empty chat DataFrame to continue with audio-only analysis")
            chat_df = pd.DataFrame(columns=['start_time', 'end_time', 'message_count',
                                           'avg_sentiment', 'avg_highlight',
                                           'avg_excitement', 'avg_funny', 'avg_happiness',
                                           'avg_anger', 'avg_sadness', 'avg_neutral'])
    else:
        logger.info("Using previously created empty chat DataFrame")

    try:
        # Integrate chat and audio analysis
        integrated_df = integrate_chat_and_audio_analysis(audio_df, chat_df)
        logger.info(f"Successfully integrated chat and audio analysis with {len(integrated_df)} segments")
    except Exception as e:
        logger.error(f"Error in integrate_chat_and_audio_analysis: {str(e)}")
        # Create a minimal integrated dataframe
        logger.warning("Creating minimal integrated dataframe")

        # Use audio_df as the base for the integrated dataframe
        integrated_df = audio_df.copy()

        # Add chat columns with default values
        chat_columns = {
            'message_count': 0,
            'avg_sentiment': 0.0,
            'avg_highlight': 0.0,
            'avg_excitement': 0.0,
            'avg_funny': 0.0,
            'avg_happiness': 0.0,
            'avg_anger': 0.0,
            'avg_sadness': 0.0,
            'avg_neutral': 1.0
        }

        for col, default_val in chat_columns.items():
            if col not in integrated_df.columns:
                integrated_df[col] = default_val

        # Add enhanced highlight score
        if 'enhanced_highlight_score' not in integrated_df.columns:
            if 'highlight_score' in integrated_df.columns:
                integrated_df['enhanced_highlight_score'] = integrated_df['highlight_score']
            else:
                integrated_df['enhanced_highlight_score'] = 0.0
                integrated_df['highlight_score'] = 0.0

        logger.info(f"Created minimal integrated dataframe with {len(integrated_df)} segments")

    # Save integrated analysis
    logger.info(f"Saving integrated analysis to {integrated_output_path}")

    # Rearrange columns to have start_time and end_time first, scores in the middle, and text at the end
    columns = list(integrated_df.columns)

    # Define the desired column order
    time_columns = ['start_time', 'end_time']
    text_columns = ['text']

    # Get all other columns (scores, metrics, etc.)
    score_columns = [col for col in columns if col not in time_columns + text_columns]

    # Create the new column order
    new_column_order = time_columns + score_columns + text_columns

    # Reorder the columns
    integrated_df = integrated_df[new_column_order]

    logger.info(f"Columns rearranged for better readability")
    integrated_df.to_csv(integrated_output_path, index=False)

    # Generate selected visualizations
    plot_highlight_comparison(integrated_df, output_dir, video_id)
    create_emotion_summary(integrated_df, ['excitement', 'funny', 'happiness', 'anger', 'sadness'],
                          output_dir, video_id, smooth=True, window_size=15)

    # Generate dedicated loudness visualization
    create_loudness_visualization(video_id, output_dir)

    # Create editor-focused visualizations
    create_editors_highlight_view(integrated_df, output_dir, video_id)

    # Extract top highlights
    top_highlights = integrated_df.nlargest(10, 'enhanced_highlight_score')

    # Rearrange columns to have start_time and end_time first, scores in the middle, and text at the end
    columns = list(top_highlights.columns)

    # Define the desired column order
    time_columns = ['start_time', 'end_time']
    text_columns = ['text']

    # Get all other columns (scores, metrics, etc.)
    score_columns = [col for col in columns if col not in time_columns + text_columns]

    # Create the new column order
    new_column_order = time_columns + score_columns + text_columns

    # Reorder the columns
    top_highlights = top_highlights[new_column_order]

    top_highlights_path = f"{output_dir}/{video_id}_top_highlights.csv"
    top_highlights.to_csv(top_highlights_path, index=False)
    logger.info(f"Top 10 highlights saved to {top_highlights_path} with columns rearranged for better readability")

    # Remove temporary files
    audio_dir = os.path.dirname(audio_sentiment_path)
    temp_files = [
        # Audio temporary files
        f"{audio_dir}/audio_{video_id}_top_highlights.csv",
        f"{audio_dir}/audio_{video_id}_top_excitement_moments.csv",
        f"{audio_dir}/audio_{video_id}_top_funny_moments.csv",
        f"{audio_dir}/audio_{video_id}_top_happiness_moments.csv",
        f"{audio_dir}/audio_{video_id}_top_anger_moments.csv",
        f"{audio_dir}/audio_{video_id}_top_sadness_moments.csv",
        f"{audio_dir}/emotion_analysis_plots_{video_id}.png",

        # Unnecessary visualization files
        f"{output_dir}/{video_id}_emotional_coherence.png",
        f"{output_dir}/{video_id}_fused_emotions_combined.png"
    ]

    # Also remove files in the emotions folder
    emotions_dir = os.path.join(output_dir, "emotions")
    if os.path.exists(emotions_dir):
        for emotion in ['excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral']:
            emotion_file = os.path.join(emotions_dir, f"{video_id}_{emotion}_comparison.png")
            if os.path.exists(emotion_file):
                try:
                    os.remove(emotion_file)
                    logger.info(f"Removed unnecessary file: {emotion_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove file {emotion_file}: {str(e)}")

        # Try to remove the emotions directory if it's empty
        try:
            os.rmdir(emotions_dir)
            logger.info(f"Removed empty directory: {emotions_dir}")
        except Exception:
            # Directory might not be empty or other error, just ignore
            pass

    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")

    # Upload integrated analysis file to GCS
    try:
        # Initialize Convex client
        convex_manager = ConvexManager()

        if USE_GCS:
            gcs_uri = None
            upload_success = False

            try:
                if file_manager.upload_to_gcs("integrated_analysis"):
                    logger.info(f"Successfully uploaded integrated analysis to GCS bucket: {ANALYSIS_BUCKET}")
                    # Get the GCS URI for the uploaded file
                    gcs_path = file_manager.get_gcs_path("integrated_analysis")
                    if gcs_path:
                        gcs_uri = f"gs://{ANALYSIS_BUCKET}/{gcs_path}"
                        upload_success = True
                else:
                    logger.warning("Failed to upload integrated analysis to GCS using file manager. Trying fallback method.")
            except Exception as upload_error:
                logger.warning(f"File manager upload failed: {str(upload_error)}. Trying fallback method.")

            # Try fallback method if primary upload failed
            if not upload_success:
                try:
                    upload_result = upload_integrated_analysis_to_gcs(video_id, integrated_output_path)
                    if upload_result:
                        logger.info(f"Successfully uploaded integrated analysis to GCS bucket using fallback method")
                        gcs_uri = upload_result.get("gcs_uri")
                        upload_success = True
                    else:
                        logger.warning("Fallback upload method also failed.")
                except Exception as fallback_error:
                    logger.warning(f"Fallback upload failed: {str(fallback_error)}")

            # Update Convex with the transcriptAnalysisUrl if we have a GCS URI
            if gcs_uri and upload_success:
                try:
                    logger.info(f"Updating Convex with transcriptAnalysisUrl: {gcs_uri}")
                    convex_manager.update_video_urls(video_id, {"transcriptAnalysisUrl": gcs_uri})
                except Exception as convex_error:
                    logger.warning(f"Failed to update Convex with GCS URI: {str(convex_error)}")
            else:
                logger.warning("No GCS URI available for transcriptAnalysisUrl update - continuing without upload")
                # In development mode, this is acceptable
                if os.environ.get('ENVIRONMENT', 'development') == 'development':
                    logger.info("Running in development mode - GCS upload failure is non-critical")
        else:
            logger.info("GCS uploads disabled. Skipping upload of integrated analysis.")
    except Exception as e:
        logger.error(f"Error in GCS upload process: {str(e)}")
        logger.warning("To fix GCS authentication issues, run: gcloud auth application-default login")
        # In development mode, continue even if upload fails
        if os.environ.get('ENVIRONMENT', 'development') == 'development':
            logger.info("Running in development mode - continuing despite GCS upload failure")

    logger.info("Integration completed successfully")
    return True

def upload_integrated_analysis_to_gcs(video_id, file_path):
    """
    Upload integrated analysis file to Google Cloud Storage
    This is a fallback method when the file manager upload fails.

    Args:
        video_id (str): The video ID
        file_path (str): Path to the integrated analysis file

    Returns:
        dict: Information about the uploaded file or None if upload failed
    """
    # Import os module to ensure it's available in this function's scope
    import os

    try:
        logger.info(f"Using fallback method to upload integrated analysis file to GCS bucket: {ANALYSIS_BUCKET}")

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        # Initialize file manager to get the correct GCS path
        file_manager = FileManager(video_id)
        gcs_path = file_manager.get_gcs_path("integrated_analysis")

        logger.info(f"Using GCS path from file manager: {gcs_path}")

        # Initialize GCS client
        try:
            # Use environment-based service account path or application default credentials
            if GCP_SERVICE_ACCOUNT_PATH and os.path.exists(GCP_SERVICE_ACCOUNT_PATH):
                # Use service account credentials if available
                logger.info(f"Using service account credentials from {GCP_SERVICE_ACCOUNT_PATH}")
                credentials = service_account.Credentials.from_service_account_file(GCP_SERVICE_ACCOUNT_PATH)
                client = storage.Client(credentials=credentials)
            else:
                # Use application default credentials
                logger.info("Using application default credentials")
                client = storage.Client()
        except Exception as e:
            logger.warning(f"Error initializing GCS client with credentials: {str(e)}")
            logger.info("Falling back to application default credentials")
            try:
                client = storage.Client()
            except Exception as auth_error:
                logger.error(f"Authentication error: {str(auth_error)}")
                logger.warning("To fix GCS authentication issues, run: gcloud auth application-default login")
                return None

        # Get bucket
        try:
            bucket = client.bucket(ANALYSIS_BUCKET)

            # Use the GCS path from file manager if available, otherwise use fallback
            if gcs_path:
                # The gcs_path includes the bucket name, so we need to extract just the blob name
                blob_name = gcs_path.split(f"{ANALYSIS_BUCKET}/")[1] if f"{ANALYSIS_BUCKET}/" in gcs_path else gcs_path
            else:
                # Fallback to old path format
                blob_name = f"{video_id}/{Path(file_path).name}"

            blob = bucket.blob(blob_name)

            # Upload file
            logger.info(f"Uploading {file_path} to gs://{ANALYSIS_BUCKET}/{blob_name}")
            blob.upload_from_filename(file_path)

            # Generate a GCS URI for the uploaded file
            gcs_uri = f"gs://{ANALYSIS_BUCKET}/{blob_name}"

            logger.info(f"Successfully uploaded integrated analysis to {gcs_uri}")

            # Return information about the uploaded file
            return {
                "file_path": str(file_path),
                "bucket": ANALYSIS_BUCKET,
                "blob_name": blob_name,
                "gcs_uri": gcs_uri
            }
        except GoogleAPIError as bucket_error:
            if "Permission denied" in str(bucket_error) or "Forbidden" in str(bucket_error):
                logger.error(f"Permission denied accessing bucket {ANALYSIS_BUCKET}. Check your permissions.")
                logger.warning("To fix GCS authentication issues, run: gcloud auth application-default login")
            elif "Not Found" in str(bucket_error):
                logger.error(f"Bucket {ANALYSIS_BUCKET} not found. Check if it exists.")
            else:
                logger.error(f"Google API error: {str(bucket_error)}")
            return None

    except GoogleAPIError as e:
        if "Reauthentication is needed" in str(e):
            logger.error("Google Cloud authentication expired.")
            logger.warning("To fix GCS authentication issues, run: gcloud auth application-default login")
        else:
            logger.error(f"Google API error uploading {file_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error uploading {file_path}: {str(e)}")
        return None

def create_loudness_visualization(video_id, output_dir):
    """
    Create a visualization focused solely on audio loudness over time.

    Args:
        video_id (str): The ID of the video to process
        output_dir (str): Directory to save the visualization

    Returns:
        str: Path to the saved visualization
    """
    logger.info(f"Generating loudness visualization for {video_id}")

    # Initialize file manager
    file_manager = FileManager(video_id)

    # Create figure
    plt.figure(figsize=(15, 6))

    # Get audio file path using file manager
    audio_file_path = file_manager.get_local_path("audio")

    # Check if audio file exists locally
    if not os.path.exists(audio_file_path):
        logger.warning(f"Audio file not found locally: {audio_file_path}")
        # Try to download from GCS if enabled
        if USE_GCS and file_manager.download_from_gcs("audio"):
            audio_file_path = file_manager.get_local_path("audio")
            logger.info(f"Downloaded audio file from GCS: {audio_file_path}")
        else:
            logger.warning(f"Could not find or download audio file. Trying fallback paths.")

            # Fallback to multiple possible paths for backward compatibility
            possible_audio_paths = [
                # Try paths with file manager first
                file_manager.get_local_path("audio"),
                # Then try various fallback paths for backward compatibility
                f"/tmp/output/Raw/Audio/audio_{video_id}.wav",
                f"/tmp/output/Raw/audio/audio_{video_id}.wav",
                f"output/Raw/Audio/audio_{video_id}.wav",  # uppercase 'Audio'
                f"output/Raw/audio/audio_{video_id}.wav",  # lowercase 'audio'
                f"output/Raw/audio_{video_id}.wav",        # directly in Raw
                f"output/Raw/audio_{video_id}.mp3"         # mp3 format
            ]

            for path in possible_audio_paths:
                if os.path.exists(path):
                    audio_file_path = path
                    logger.info(f"Found audio file for loudness visualization at fallback path: {audio_file_path}")
                    break

            if not os.path.exists(audio_file_path):
                logger.warning(f"Audio file not found in any location for loudness visualization")
                # Use the file manager path as default even if it doesn't exist
                audio_file_path = file_manager.get_local_path("audio")

    waveform_times, waveform_amplitudes = extract_audio_waveform(audio_file_path, num_points=2000)

    if len(waveform_times) == 0 or len(waveform_amplitudes) == 0:
        logger.error(f"Could not extract waveform data from {audio_file_path}")
        plt.text(0.5, 0.5, 'Audio waveform not available',
                horizontalalignment='center', verticalalignment='center')
        plt.title('Audio Loudness')

        # Save empty plot
        plot_path = os.path.join(output_dir, f"{video_id}_loudness.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path

    # Convert to absolute values to get loudness
    loudness = np.abs(waveform_amplitudes)

    # Apply smoothing with rolling window
    window_size = 50  # Adjust based on desired smoothness
    if len(loudness) > window_size:
        # Use pandas rolling window for smoothing
        loudness_series = pd.Series(loudness)
        smoothed_loudness = loudness_series.rolling(window=window_size, center=True).mean()
        # Fill NaN values at the edges using newer methods
        smoothed_loudness = smoothed_loudness.bfill().ffill()
    else:
        smoothed_loudness = loudness

    # Plot the raw loudness as a light fill
    plt.fill_between(waveform_times, 0, loudness, alpha=0.3, color='lightblue', label='Raw Loudness')

    # Plot the smoothed loudness as a solid line
    plt.plot(waveform_times, smoothed_loudness, 'b-', linewidth=2, label='Smoothed Loudness')

    # Calculate and plot the average loudness
    avg_loudness = np.mean(loudness)
    plt.axhline(y=avg_loudness, color='r', linestyle='--',
                label=f'Average Loudness: {avg_loudness:.2f}')

    # Add markers for high loudness points (e.g., top 5%)
    threshold = np.percentile(loudness, 95)
    high_points = np.where(smoothed_loudness > threshold)[0]

    # Group adjacent high points into segments
    segments = []
    if len(high_points) > 0:
        segment_start = high_points[0]
        for i in range(1, len(high_points)):
            if high_points[i] - high_points[i-1] > 5:  # Gap of more than 5 points
                segments.append((segment_start, high_points[i-1]))
                segment_start = high_points[i]
        segments.append((segment_start, high_points[-1]))  # Add the last segment

    # Mark high loudness segments
    for start, end in segments:
        mid_point = (start + end) // 2
        mid_time = waveform_times[mid_point]
        mid_loudness = smoothed_loudness[mid_point]

        # Format timestamp as MM:SS
        minutes = int(mid_time // 60)
        seconds = int(mid_time % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"

        # Add vertical span for the segment
        plt.axvspan(waveform_times[start], waveform_times[end],
                   alpha=0.2, color='red', label='_nolegend_')

        # Add annotation with timestamp
        plt.annotate(timestamp,
                    xy=(mid_time, mid_loudness),
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # Add stream timeline markers (every 5 minutes)
    max_time = waveform_times[-1]
    for minute in range(0, int(max_time/60) + 1, 5):
        time_sec = minute * 60
        if time_sec <= max_time:
            plt.axvline(x=time_sec, color='gray', linestyle=':', alpha=0.5)
            plt.text(time_sec, -0.05, f"{minute:02d}:00",
                    ha='center', va='top', fontsize=9, alpha=0.7)

    # Set title and labels
    plt.title(f'Audio Loudness Over Time - {video_id}', fontsize=16)
    plt.xlabel('Stream Time (minutes:seconds)', fontsize=12)
    plt.ylabel('Loudness (Absolute Amplitude)', fontsize=12)

    # Format x-axis as minutes:seconds
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60):02d}:{int(x%60):02d}'))

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Add grid for easier reading
    plt.grid(True, alpha=0.3)

    # Add text box with explanation
    explanation_text = (
        "Loudness Visualization Guide:\n"
        "• Blue area shows raw audio loudness\n"
        "• Blue line shows smoothed loudness\n"
        "• Red dashed line shows average loudness\n"
        "• Red highlighted areas mark loudness peaks\n"
        "• Timestamps indicate peak loudness moments"
    )
    plt.figtext(0.02, 0.02, explanation_text, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"{video_id}_loudness.png")
    plt.savefig(plot_path, dpi=150)  # Higher DPI for better quality
    plt.close()

    logger.info(f"Loudness visualization saved to {plot_path}")
    return plot_path
