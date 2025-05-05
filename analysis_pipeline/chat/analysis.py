"""
Chat Analysis Module

This module handles analysis of Twitch chat data to identify highlights and emotional peaks.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("chat_analysis", "chat_analysis.log")

def plot_metrics(stats, output_dir, window=5):
    """
    Create visualizations with smoothed lines for the interval analysis.

    Args:
        stats (pd.DataFrame): DataFrame containing the statistics
        output_dir (str): Directory to save the plots
        window (int): Window size for rolling average smoothing

    Returns:
        bool: True if plotting was successful, False otherwise
    """
    try:
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_theme()

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 35))
        gs = fig.add_gridspec(7, 1, hspace=0.4)

        # Calculate rolling averages for smoothing
        smooth_messages = stats['message_count'].rolling(window=window, center=True).mean()
        smooth_sentiment = stats['avg_sentiment'].rolling(window=window, center=True).mean()

        # Calculate weighted metrics
        stats['weighted_sentiment'] = stats['avg_sentiment'] * stats['message_count']
        smooth_weighted_sentiment = stats['weighted_sentiment'].rolling(window=window, center=True).mean()

        emotions = ['avg_excitement', 'avg_funny', 'avg_happiness', 'avg_anger', 'avg_sadness']
        for emotion in emotions:
            stats[f'weighted_{emotion}'] = stats[emotion] * stats['message_count']

        # Add smoothing for highlight scores
        smooth_highlight = stats['avg_highlight'].rolling(window=window, center=True).mean()
        stats['weighted_highlight'] = stats['avg_highlight'] * stats['message_count']
        smooth_weighted_highlight = stats['weighted_highlight'].rolling(window=window, center=True).mean()

        # Plot 1: Message Count over time
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(stats['time_mid'], stats['message_count'],
                    alpha=0.2, color='blue', s=20)
        ax1.plot(stats['time_mid'], smooth_messages,
                 color='darkblue', linewidth=2.5, label='Smoothed trend')

        ax1.set_title('Message Frequency Over Time', fontsize=14, pad=15)
        ax1.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax1.set_ylabel('Messages per 30s', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Plot 2: Raw Sentiment over time
        ax2 = fig.add_subplot(gs[1])
        ax2.scatter(stats['time_mid'], stats['avg_sentiment'],
                    alpha=0.2, color='green', s=20)
        ax2.plot(stats['time_mid'], smooth_sentiment,
                 color='darkgreen', linewidth=2.5, label='Smoothed trend')

        ax2.set_title('Average Sentiment Over Time (Raw)', fontsize=14, pad=15)
        ax2.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax2.set_ylabel('Sentiment Score', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        # Plot 3: Weighted Sentiment over time
        ax3 = fig.add_subplot(gs[2])
        ax3.scatter(stats['time_mid'], stats['weighted_sentiment'],
                    alpha=0.2, color='purple', s=20)
        ax3.plot(stats['time_mid'], smooth_weighted_sentiment,
                 color='darkmagenta', linewidth=2.5, label='Smoothed trend')

        ax3.set_title('Weighted Sentiment Over Time (Sentiment × Message Count)',
                     fontsize=14, pad=15)
        ax3.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax3.set_ylabel('Weighted Sentiment', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()

        # Plot 4: Raw Emotions over time
        ax4 = fig.add_subplot(gs[3])
        colors = sns.color_palette("husl", len(emotions))

        for emotion, color in zip(emotions, colors):
            smooth_emotion = stats[emotion].rolling(window=window, center=True).mean()
            ax4.plot(stats['time_mid'], smooth_emotion,
                     linewidth=2, label=emotion.replace('avg_', ''),
                     color=color)
            ax4.scatter(stats['time_mid'], stats[emotion],
                       alpha=0.1, s=10, color=color)

        ax4.set_title('Emotions Over Time (Raw)', fontsize=14, pad=15)
        ax4.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax4.set_ylabel('Emotion Score', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 5: Weighted Emotions over time
        ax5 = fig.add_subplot(gs[4])

        for emotion, color in zip(emotions, colors):
            weighted_emotion = f'weighted_{emotion}'
            smooth_weighted = stats[weighted_emotion].rolling(window=window, center=True).mean()
            ax5.plot(stats['time_mid'], smooth_weighted,
                     linewidth=2, label=emotion.replace('avg_', ''),
                     color=color)
            ax5.scatter(stats['time_mid'], stats[weighted_emotion],
                       alpha=0.1, s=10, color=color)

        ax5.set_title('Weighted Emotions Over Time (Emotion × Message Count)',
                     fontsize=14, pad=15)
        ax5.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax5.set_ylabel('Weighted Emotion Score', fontsize=12)
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 6: Highlight Scores over time
        ax6 = fig.add_subplot(gs[5])
        ax6.scatter(stats['time_mid'], stats['avg_highlight'],
                   alpha=0.2, color='purple', s=20)
        ax6.plot(stats['time_mid'], smooth_highlight,
                color='darkviolet', linewidth=2.5, label='Smoothed trend')

        ax6.set_title('Average Highlight Score Over Time', fontsize=14, pad=15)
        ax6.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax6.set_ylabel('Highlight Score (0-3)', fontsize=12)
        ax6.grid(True, linestyle='--', alpha=0.7)
        ax6.legend()

        # Plot 7: Weighted Highlight Score over time
        ax7 = fig.add_subplot(gs[6])
        ax7.scatter(stats['time_mid'], stats['weighted_highlight'],
                   alpha=0.2, color='orange', s=20)
        ax7.plot(stats['time_mid'], smooth_weighted_highlight,
                 color='darkorange', linewidth=2.5, label='Smoothed trend')

        ax7.set_title('Weighted Highlight Score Over Time (Highlight × Message Count)',
                     fontsize=14, pad=15)
        ax7.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax7.set_ylabel('Weighted Highlight Score', fontsize=12)
        ax7.grid(True, linestyle='--', alpha=0.7)
        ax7.legend()

        # Adjust layout and save
        plt.savefig(os.path.join(output_dir, 'chat_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        logger.error(f"Error during plotting: {str(e)}")
        plt.close()
        return False

def get_input_files(video_id, output_dir):
    """
    Get paths to required input files based on video ID.

    Args:
        video_id (str): The video ID to analyze
        output_dir (str): Base directory containing output files

    Returns:
        tuple: Paths to (sentiment_csv, segments_csv)
    """
    output_dir = Path(output_dir)
    sentiment_csv = output_dir / f"{video_id}_chat_sentiment.csv"
    segments_csv = Path(f'Output/Raw/Transcripts/audio_{video_id}_segments.csv')

    # Verify files exist
    if not os.path.exists(sentiment_csv):
        raise FileNotFoundError(f"Chat sentiment file not found: {sentiment_csv}")
    if not os.path.exists(segments_csv):
        raise FileNotFoundError(f"Sliding window segments file not found: {segments_csv}")

    return sentiment_csv, segments_csv

def analyze_chat_intervals(video_id, output_dir=None):
    """
    Analyze chat messages that correspond to each segment's time window.
    This function is kept for compatibility but now just returns None.

    Args:
        video_id (str): The video ID to analyze
        output_dir (str, optional): Directory to save output files

    Returns:
        None: This function no longer generates any output
    """
    logger.info("Chat interval analysis is deprecated - using highlight analysis only")
    return None

def analyze_chat_highlights(video_id, output_dir=None):
    """
    Analyze chat data to identify highlight moments and emotional peaks based on sliding window segments.

    This function uses the segments file generated by the sliding window generator to analyze chat data
    for each segment's time window. It calculates various statistics for each segment, including
    message count, sentiment scores, and emotion scores.

    Args:
        video_id (str): The video ID to analyze
        output_dir (str, optional): Directory where output files should be saved

    Returns:
        pd.DataFrame: DataFrame with highlight analysis results, or None if analysis failed
    """
    try:
        # Define output directory
        if output_dir is None:
            output_dir = Path('Output/Analysis/Chat')
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Get input file path
        input_file, segments_file = get_input_files(video_id, output_dir)

        # Check if file exists
        if not os.path.exists(input_file):
            logger.error(f"Required file not found: {input_file}")
            logger.info("Please run the full pipeline with a VOD URL first to generate the required data files.")
            return None

        # Load data
        data = pd.read_csv(input_file)

        # Remove any leading/trailing whitespaces in column names
        data.columns = data.columns.str.strip()

        # Check for missing necessary columns
        required_columns = [
            'time', 'sentiment_score', 'highlight_score',
            'excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral', 'message'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing columns in data: {missing_columns}")
            return None

        # Load segments data to get time windows
        segments = pd.read_csv(segments_file)

        # Initialize list to store highlight stats
        highlight_stats_list = []

        # Analyze messages for each segment's time window
        for _, segment in segments.iterrows():
            start_time = segment['start_time']
            # Add 15 second buffer after segment end since chat reactions can be delayed
            end_time = segment['end_time'] + 15

            # Get messages in this time window
            mask = (data['time'] >= start_time) & (data['time'] <= end_time)
            window_messages = data[mask]

            # Calculate statistics for the highlight window
            if len(window_messages) > 0:
                stats = {
                    'start_time': start_time,
                    'end_time': segment['end_time'],
                    'time_mid': (start_time + segment['end_time']) / 2,
                    'message_count': len(window_messages),
                    'avg_sentiment': window_messages['sentiment_score'].mean(),
                    'std_sentiment': window_messages['sentiment_score'].std(),
                    'avg_highlight': window_messages['highlight_score'].mean(),
                    'avg_excitement': window_messages['excitement'].mean(),
                    'avg_funny': window_messages['funny'].mean(),
                    'avg_happiness': window_messages['happiness'].mean(),
                    'avg_anger': window_messages['anger'].mean(),
                    'avg_sadness': window_messages['sadness'].mean(),
                    'avg_neutral': window_messages['neutral'].mean(),
                    'text': segment['text']  # Include segment text
                }
            else:
                # If no messages in window, use zeros for stats
                stats = {
                    'start_time': start_time,
                    'end_time': segment['end_time'],
                    'time_mid': (start_time + segment['end_time']) / 2,
                    'message_count': 0,
                    'avg_sentiment': 0,
                    'std_sentiment': 0,
                    'avg_highlight': 0,
                    'avg_excitement': 0,
                    'avg_funny': 0,
                    'avg_happiness': 0,
                    'avg_anger': 0,
                    'avg_sadness': 0,
                    'avg_neutral': 0,
                    'text': segment['text']
                }

            highlight_stats_list.append(stats)

        # Convert to DataFrame
        highlight_interval_stats = pd.DataFrame(highlight_stats_list)

        # Save to CSV with video_id in filename
        output_file = output_dir / f'{video_id}_highlight_analysis.csv'
        highlight_interval_stats.to_csv(output_file, index=False)
        logger.info(f"Chat highlight analysis saved to {output_file}")

        return highlight_interval_stats

    except Exception as e:
        logger.error(f"Error during highlight analysis: {str(e)}")
        return None
