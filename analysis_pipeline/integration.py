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
from datetime import datetime

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

    # Process each paragraph
    total_rows = len(integrated_df)
    logger.info(f"Processing {total_rows} paragraphs")

    for i, row in integrated_df.iterrows():
        # Log progress periodically
        if i % 100 == 0 or i == total_rows - 1:
            logger.info(f"Processing paragraph {i+1}/{total_rows} ({(i+1)/total_rows*100:.1f}%)")

        # Get time range for this paragraph
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
    Create a summary plot with small versions of all emotion comparisons

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
        emotions (list): List of emotions to include
        output_dir (str): Directory to save the plot
        video_id (str): Video ID for filename
        smooth (bool): Whether to apply smoothing
        window_size (int): Size of the smoothing window
    """
    # Calculate rows and columns for subplot grid
    n_emotions = len(emotions)
    n_cols = 2
    n_rows = (n_emotions + n_cols - 1) // n_cols  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Flatten axes array for easier indexing
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes[0], axes[1]]  # Handle case of single row

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

    # Hide any unused subplots
    for i in range(n_emotions, len(axes)):
        axes[i].set_visible(False)

    # Add a title to the entire figure
    if smooth:
        fig.suptitle('Emotion Summary (Smoothed)', fontsize=16)
    else:
        fig.suptitle('Emotion Summary', fontsize=16)

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
    create_editors_highlight_csv(integrated_df, top_highlights, editor_dir, video_id)

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
    try:
        import librosa
        import numpy as np

        # Load audio file
        logger.info(f"Loading audio file for waveform extraction: {audio_file_path}")
        y, sr = librosa.load(audio_file_path, sr=None)

        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)

        # Resample to desired number of points
        if len(y) > num_points:
            # Calculate points per second
            points_per_second = num_points / duration

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
        return np.array([]), np.array([])
    except Exception as e:
        logger.error(f"Error extracting audio waveform: {str(e)}")
        return np.array([]), np.array([])

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
        import plotly.express as px
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

        # Extract audio waveform
        audio_file_path = f"Output/Raw/Audio/audio_{video_id}.wav"
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

def create_editors_highlight_csv(integrated_df, top_highlights, output_dir, video_id):
    """
    Create a CSV file with detailed highlight information for video editors

    Args:
        integrated_df (DataFrame): Integrated analysis dataframe
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

    # Save to CSV
    csv_path = os.path.join(output_dir, f"{video_id}_editors_highlights.csv")
    editors_df.to_csv(csv_path, index=False)

    logger.info(f"Editor's highlight CSV saved to {csv_path}")
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

    # Define input and output paths
    audio_sentiment_path = f"Output/Analysis/Audio/audio_{video_id}_sentiment.csv"
    chat_analysis_path = f"Output/Analysis/Chat/{video_id}_highlight_analysis.csv"
    output_dir = "Output/Analysis/Integrated"
    integrated_output_path = f"{output_dir}/{video_id}_integrated_analysis.csv"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if input files exist
    if not os.path.exists(audio_sentiment_path):
        logger.error(f"Audio sentiment file not found: {audio_sentiment_path}")
        return False

    if not os.path.exists(chat_analysis_path):
        logger.error(f"Chat analysis file not found: {chat_analysis_path}")
        return False

    # Load audio sentiment data
    logger.info(f"Loading audio sentiment data from {audio_sentiment_path}")
    try:
        audio_df = pd.read_csv(audio_sentiment_path)
        logger.info(f"Loaded {len(audio_df)} paragraphs from audio sentiment file")
    except Exception as e:
        logger.error(f"Error loading audio sentiment data: {str(e)}")
        return False

    # Load chat analysis data
    logger.info(f"Loading chat analysis data from {chat_analysis_path}")
    try:
        chat_df = pd.read_csv(chat_analysis_path)
        logger.info(f"Loaded {len(chat_df)} segments from chat analysis file")
    except Exception as e:
        logger.error(f"Error loading chat analysis data: {str(e)}")
        return False

    # Integrate chat and audio analysis
    integrated_df = integrate_chat_and_audio_analysis(audio_df, chat_df)

    # Save integrated analysis
    logger.info(f"Saving integrated analysis to {integrated_output_path}")
    integrated_df.to_csv(integrated_output_path, index=False)

    # Generate visualizations
    plot_emotional_coherence(integrated_df, output_dir, video_id)
    plot_highlight_comparison(integrated_df, output_dir, video_id)
    plot_fused_emotions(integrated_df, output_dir, video_id)

    # Create editor-focused visualizations
    create_editors_highlight_view(integrated_df, output_dir, video_id)

    # Extract top highlights
    top_highlights = integrated_df.nlargest(10, 'enhanced_highlight_score')
    top_highlights_path = f"{output_dir}/{video_id}_top_highlights.csv"
    top_highlights.to_csv(top_highlights_path, index=False)
    logger.info(f"Top 10 highlights saved to {top_highlights_path}")

    logger.info("Integration completed successfully")
    return True
