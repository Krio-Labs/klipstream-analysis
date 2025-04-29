#!/usr/bin/env python3
"""
Visualization script for transcript analysis results.

This script creates visualizations to compare the results of different
sentiment and emotion analyses on transcript data.

Usage:
python visualize_analysis.py --input_file outputs/audio_VIDEOID_paragraphs_hf.csv --output_dir visualizations
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from matplotlib.ticker import PercentFormatter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sentiment_comparison_chart(df, output_dir):
    """
    Create a chart comparing the existing sentiment analysis with the Hugging Face sentiment analysis.

    Args:
        df: DataFrame with both sentiment analyses
        output_dir: Directory to save the chart
    """
    logger.info("Creating sentiment comparison chart...")

    # Map existing sentiment scores to categorical labels if not already done
    if 'existing_sentiment_label' not in df.columns and 'sentiment_score' in df.columns:
        df['existing_sentiment_label'] = df['sentiment_score'].apply(
            lambda x: "positive" if x > 0.3 else ("negative" if x < -0.3 else "neutral")
        )

    # Create a confusion matrix
    if 'existing_sentiment_label' in df.columns and 'hf_sentiment' in df.columns:
        confusion = pd.crosstab(
            df['existing_sentiment_label'],
            df['hf_sentiment'],
            rownames=['Existing'],
            colnames=['Hugging Face'],
            normalize='index'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt='.1%', cmap='Blues', vmin=0, vmax=1)
        plt.title('Sentiment Analysis Comparison\n(Existing vs. Hugging Face)')
        plt.tight_layout()

        # Save the chart
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'sentiment_comparison.png')
        plt.savefig(output_file, dpi=300)
        logger.info(f"Saved sentiment comparison chart to {output_file}")
        plt.close()
    else:
        logger.warning("Required columns for sentiment comparison not found")

def create_emotion_distribution_chart(df, output_dir):
    """
    Create a chart showing the distribution of emotions in the transcript.

    Args:
        df: DataFrame with emotion analysis
        output_dir: Directory to save the chart
    """
    logger.info("Creating emotion distribution chart...")

    if 'hf_emotion' in df.columns:
        # Count emotions
        emotion_counts = df['hf_emotion'].value_counts()

        plt.figure(figsize=(12, 8))
        ax = emotion_counts.plot(kind='bar', color=sns.color_palette('viridis', len(emotion_counts)))

        # Add percentage labels
        total = len(df)
        for i, count in enumerate(emotion_counts):
            percentage = count / total * 100
            ax.text(i, count + 0.1, f'{percentage:.1f}%', ha='center')

        plt.title('Distribution of Emotions in Transcript')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the chart
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'emotion_distribution.png')
        plt.savefig(output_file, dpi=300)
        logger.info(f"Saved emotion distribution chart to {output_file}")
        plt.close()
    else:
        logger.warning("Required column 'hf_emotion' not found")

def create_sentiment_timeline(df, output_dir):
    """
    Create a timeline chart showing sentiment over time.

    Args:
        df: DataFrame with sentiment analysis and timestamps
        output_dir: Directory to save the chart
    """
    logger.info("Creating sentiment timeline chart...")

    if 'start_time' in df.columns and 'hf_sentiment' in df.columns:
        # Sort by start time
        df_sorted = df.sort_values('start_time')

        # Create a numeric sentiment score for plotting
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1, 'error': np.nan}
        df_sorted['sentiment_numeric'] = df_sorted['hf_sentiment'].map(sentiment_map)

        # Create the plot
        plt.figure(figsize=(15, 6))

        # Plot the sentiment timeline
        plt.plot(df_sorted['start_time'], df_sorted['sentiment_numeric'], 'o-', alpha=0.6)

        # Add a rolling average
        window_size = min(30, len(df_sorted) // 10)  # Adjust window size based on data length
        if window_size > 0:
            rolling_avg = df_sorted['sentiment_numeric'].rolling(window=window_size, center=True).mean()
            plt.plot(df_sorted['start_time'], rolling_avg, 'r-', linewidth=2, label=f'Rolling Average (window={window_size})')

        # Add horizontal lines for sentiment categories
        plt.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Positive')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3, label='Neutral')
        plt.axhline(y=-1, color='red', linestyle='--', alpha=0.3, label='Negative')

        # Format the plot
        plt.title('Sentiment Timeline')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Sentiment')
        plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Convert x-axis to minutes:seconds format for better readability
        def format_time(seconds):
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f'{minutes}:{seconds:02d}'

        # Set x-axis ticks at regular intervals
        max_time = df_sorted['start_time'].max()
        interval = max(60, max_time // 10)  # At least every minute, or 10 divisions
        ticks = np.arange(0, max_time + interval, interval)
        plt.xticks(ticks, [format_time(t) for t in ticks], rotation=45)

        plt.tight_layout()

        # Save the chart
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'sentiment_timeline.png')
        plt.savefig(output_file, dpi=300)
        logger.info(f"Saved sentiment timeline chart to {output_file}")
        plt.close()
    else:
        logger.warning("Required columns for sentiment timeline not found")

def create_emotion_timeline(df, output_dir):
    """
    Create a timeline chart showing emotions over time.

    Args:
        df: DataFrame with emotion analysis and timestamps
        output_dir: Directory to save the chart
    """
    logger.info("Creating emotion timeline chart...")

    if 'start_time' in df.columns and 'hf_emotion' in df.columns:
        # Sort by start time
        df_sorted = df.sort_values('start_time')

        # Get the top 5 most common emotions
        top_emotions = df_sorted['hf_emotion'].value_counts().head(5).index.tolist()

        if len(top_emotions) == 0:
            logger.warning("No emotions found in the data")
            return

        # Handle the case where there's only one emotion
        if len(top_emotions) == 1:
            # Create a single plot
            plt.figure(figsize=(15, 5))
            emotion = top_emotions[0]
            emotion_binary = (df_sorted['hf_emotion'] == emotion).astype(int)
            plt.fill_between(df_sorted['start_time'], 0, emotion_binary, alpha=0.6)
            plt.ylabel(emotion.capitalize())
            plt.ylim(0, 1.1)
            plt.yticks([0, 1])
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time (seconds)')
            plt.title(f'Emotion Timeline: {emotion.capitalize()}')

            # Convert x-axis to minutes:seconds format for better readability
            def format_time(seconds):
                minutes = int(seconds // 60)
                seconds = int(seconds % 60)
                return f'{minutes}:{seconds:02d}'

            # Set x-axis ticks at regular intervals
            max_time = df_sorted['start_time'].max()
            interval = max(60, max_time // 10)  # At least every minute, or 10 divisions
            ticks = np.arange(0, max_time + interval, interval)
            plt.xticks(ticks, [format_time(t) for t in ticks], rotation=45)

        else:
            # Create a figure with subplots
            fig, axes = plt.subplots(len(top_emotions), 1, figsize=(15, 10), sharex=True)

            # Plot each emotion as a binary occurrence
            for i, emotion in enumerate(top_emotions):
                emotion_binary = (df_sorted['hf_emotion'] == emotion).astype(int)
                axes[i].fill_between(df_sorted['start_time'], 0, emotion_binary, alpha=0.6)
                axes[i].set_ylabel(emotion.capitalize())
                axes[i].set_ylim(0, 1.1)
                axes[i].set_yticks([0, 1])
                axes[i].grid(True, alpha=0.3)

            # Format the plot
            axes[-1].set_xlabel('Time (seconds)')
            fig.suptitle('Emotion Timeline (Top 5 Emotions)')

            # Convert x-axis to minutes:seconds format for better readability
            def format_time(seconds):
                minutes = int(seconds // 60)
                seconds = int(seconds % 60)
                return f'{minutes}:{seconds:02d}'

            # Set x-axis ticks at regular intervals
            max_time = df_sorted['start_time'].max()
            interval = max(60, max_time // 10)  # At least every minute, or 10 divisions
            ticks = np.arange(0, max_time + interval, interval)
            axes[-1].set_xticks(ticks)
            axes[-1].set_xticklabels([format_time(t) for t in ticks], rotation=45)

        plt.tight_layout()

        # Save the chart
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'emotion_timeline.png')
        plt.savefig(output_file, dpi=300)
        logger.info(f"Saved emotion timeline chart to {output_file}")
        plt.close()
    else:
        logger.warning("Required columns for emotion timeline not found")

def main():
    parser = argparse.ArgumentParser(description='Visualize transcript analysis results')
    parser.add_argument('--input_file', required=True, help='Path to the input CSV file with analysis results')
    parser.add_argument('--output_dir', default='visualizations', help='Directory to save visualizations')

    args = parser.parse_args()

    # Load analysis results
    logger.info(f"Loading analysis results from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df)} entries")
    except Exception as e:
        logger.error(f"Error loading analysis results: {e}")
        return

    # Create visualizations
    create_sentiment_comparison_chart(df, args.output_dir)
    create_emotion_distribution_chart(df, args.output_dir)
    create_sentiment_timeline(df, args.output_dir)
    create_emotion_timeline(df, args.output_dir)

    logger.info("Visualization complete!")

if __name__ == "__main__":
    main()
