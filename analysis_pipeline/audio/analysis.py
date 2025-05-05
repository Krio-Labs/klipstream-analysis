"""
Audio Analysis Module

This module handles analysis of audio transcriptions to identify highlights and emotion peaks.
"""

import pandas as pd
import numpy as np
import os
import librosa
import gc
from pathlib import Path

# Set matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("audio_analysis", "audio_analysis.log")

def plot_metrics(output_dir, video_id):
    """
    Plot various metrics including sentiment, highlight score, and individual emotions.

    Args:
        output_dir (str): Directory to save the plots
        video_id (str): The ID of the video to process

    Returns:
        bool: True if plotting was successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Look for the CSV file in the output directory
        file_path = output_dir / f"audio_{video_id}_sentiment.csv"
        if not file_path.exists():
            # Try in Raw directory
            raw_file_path = Path(f'Output/Raw/Transcripts/audio_{video_id}_segments.csv')
            if os.path.exists(raw_file_path):
                file_path = raw_file_path
                logger.info(f"Using segments file from Raw directory: {file_path}")
            else:
                raise FileNotFoundError(f"Could not find segments file in {file_path} or {raw_file_path}")
        else:
            logger.info(f"Using sentiment file from: {file_path}")

        df = pd.read_csv(file_path)

        # Check for required columns and add them if missing
        required_columns = [
            'sentiment_score', 'highlight_score',
            'excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral'
        ]

        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column in data: {col}, adding with default values")
                if col in ['sentiment_score', 'highlight_score']:
                    df[col] = 0.0
                elif col in ['excitement', 'funny', 'happiness', 'anger', 'sadness']:
                    df[col] = 0.0
                elif col == 'neutral':
                    df[col] = 1.0

        # Create figure with subplots
        emotions = ['excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral']
        num_plots = len(emotions) + 3  # +3 for waveform, sentiment and highlight

        fig, axs = plt.subplots(num_plots, 1, figsize=(15, 4*num_plots))

        # Plot 1: Audio Waveform
        ax = axs[0]
        try:
            # Look for the audio file in the output directory
            audio_path = Path(f'Output/Raw/Audio/audio_{video_id}.wav')

            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path)
                logger.info(f"Using audio file from: {audio_path}")
            else:
                raise FileNotFoundError(f"Could not find audio file in {audio_path}")

            times = np.arange(len(y)) / sr

            # Plot waveform
            ax.plot(times, y, color='blue', alpha=0.7, linewidth=1)
            ax.set_title('Audio Waveform')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting waveform: {str(e)}")
            ax.text(0.5, 0.5, 'Waveform not available',
                   horizontalalignment='center', verticalalignment='center')

        # Plot 2: Sentiment Score with moving average and confidence bands
        ax = axs[1]
        window = 5  # Moving average window size
        smoothed_sentiment = df['sentiment_score'].rolling(window=window, center=True).mean()
        std_sentiment = df['sentiment_score'].rolling(window=window, center=True).std()

        # Plot raw data, moving average and confidence bands
        ax.fill_between(df['start_time'],
                       smoothed_sentiment - std_sentiment,
                       smoothed_sentiment + std_sentiment,
                       color='lightblue', alpha=0.3, label='Confidence Band')
        ax.plot(df['start_time'], df['sentiment_score'],
                label='Raw Sentiment', color='lightblue', alpha=0.3, linewidth=1)
        ax.plot(df['start_time'], smoothed_sentiment,
                label=f'Moving Average ({window} intervals)',
                color='blue', linewidth=2)
        ax.set_title('Sentiment Score Over Time')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Highlight Score with moving average and peaks
        ax = axs[2]
        smoothed_highlight = df['highlight_score'].rolling(window=window, center=True).mean()

        # Find peaks in highlight score
        peaks, _ = find_peaks(smoothed_highlight, height=0.6, distance=50)

        ax.plot(df['start_time'], df['highlight_score'],
                label='Raw Highlight', color='plum', alpha=0.3, linewidth=1)
        ax.plot(df['start_time'], smoothed_highlight,
                label=f'Moving Average ({window} intervals)',
                color='purple', linewidth=2)

        # Plot peaks
        if len(peaks) > 0:
            ax.scatter(df['start_time'].iloc[peaks],
                      smoothed_highlight.iloc[peaks],
                      color='red', s=100, label='Peak Moments')

        ax.set_title('Highlight Score Over Time')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Get audio waveform amplitude envelope
        try:
            # Look for the audio file in the output directory
            audio_path = Path(f'Output/Raw/Audio/audio_{video_id}.wav')

            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path)
            else:
                raise FileNotFoundError(f"Could not find audio file in {audio_path}")

            # Calculate the RMS energy envelope
            frame_length = int(sr * 0.03)  # 30ms frames
            hop_length = int(sr * 0.01)    # 10ms hop
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        except Exception as e:
            logger.error(f"Error loading audio for RMS calculation: {str(e)}")
            # Create dummy RMS data
            rms = np.ones(100)
            sr = 16000
            frame_length = int(sr * 0.03)
            hop_length = int(sr * 0.01)

        # Resample RMS to match emotion data length
        rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        resampled_rms = np.interp(df['start_time'], rms_times, rms)

        # Normalize RMS to [0.5, 1.5] range for amplification
        normalized_rms = ((resampled_rms - resampled_rms.min()) /
                         (resampled_rms.max() - resampled_rms.min())) + 0.5

        # Individual emotion plots with waveform-weighted amplification
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FF99CC', '#99CCFF', '#FFB366']
        for i, (emotion, color) in enumerate(zip(emotions, colors)):
            ax = axs[i + 3]

            # Apply waveform-based amplification
            amplified_emotion = df[emotion] * normalized_rms
            smoothed_emotion = amplified_emotion.rolling(window=window, center=True).mean()

            # Create area plot with amplified values
            ax.fill_between(df['start_time'], 0, amplified_emotion,
                          alpha=0.2, color=color,
                          label='Amplified Emotion')

            # Plot smoothed data
            ax.plot(df['start_time'], smoothed_emotion,
                   label=f'Moving Average ({window} intervals)',
                   color=color,
                   linewidth=2)

            # Add original emotion line for comparison
            ax.plot(df['start_time'], df[emotion],
                   label='Original Emotion',
                   color=color,
                   alpha=0.3,
                   linewidth=1,
                   linestyle='--')

            ax.set_title(f'{emotion.capitalize()} Over Time (Waveform Weighted)')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f'emotion_analysis_plots_{video_id}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

        return True

    except Exception as e:
        logger.error(f"Error during plotting: {str(e)}")
        plt.close()
        return False

def analyze_transcription_highlights(video_id, input_file=None, output_dir=None):
    """
    Analyze transcription data to identify highlights and emotion peaks

    Args:
        video_id (str): The ID of the video to analyze
        input_file (str, optional): Path to the input segments CSV file
        output_dir (str, optional): Directory to save output files

    Returns:
        pd.DataFrame: DataFrame containing the top highlights, or None if analysis failed
    """
    try:
        # Define output directory
        if output_dir is None:
            output_dir = Path('Output/Analysis/Audio')
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Determine input file path
        if input_file is None:
            input_file = Path(f'Output/Raw/Transcripts/audio_{video_id}_segments.csv')
        else:
            input_file = Path(input_file)

        # Check if file exists
        if not os.path.exists(input_file):
            logger.error(f"Required file not found: {input_file}")
            logger.info("Please run the full pipeline first to generate the required transcription file.")
            return None

        # Load transcription data
        data = pd.read_csv(input_file)
        data.columns = data.columns.str.strip()

        # Load audio file
        try:
            # Try to find the audio file in multiple locations
            audio_path = Path(f'Output/Raw/Audio/audio_{video_id}.wav')
            tmp_audio_path = Path(f'/tmp/outputs/audio_{video_id}.wav')

            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path)
                logger.info(f"Using audio file from: {audio_path}")
            elif os.path.exists(tmp_audio_path):
                y, sr = librosa.load(tmp_audio_path)
                logger.info(f"Using audio file from: {tmp_audio_path}")
            else:
                raise FileNotFoundError(f"Could not find audio file in {audio_path} or {tmp_audio_path}")

            # Calculate audio features
            frame_length = int(sr * 0.03)  # 30ms frames
            hop_length = int(sr * 0.01)    # 10ms hop

            # Get RMS energy (loudness)
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            # Get spectral centroid (brightness/pitch)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr,
                n_fft=frame_length,
                hop_length=hop_length
            )[0]

            # Get timestamps for audio features
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

            # Create interpolation functions for audio features
            rms_interp = np.interp(data['start_time'], times, rms)
            centroid_interp = np.interp(data['start_time'], times, spectral_centroid)

            # Normalize audio features to [0,1] range
            rms_norm = (rms_interp - rms_interp.min()) / (rms_interp.max() - rms_interp.min())
            centroid_norm = (centroid_interp - centroid_interp.min()) / (centroid_interp.max() - centroid_interp.min())

            # Combine audio features
            audio_intensity = (rms_norm * 0.7 + centroid_norm * 0.3)  # Weighted combination

        except Exception as e:
            logger.warning(f"Could not load audio file: {str(e)}")
            audio_intensity = np.ones(len(data))  # Fallback if audio processing fails

        # Check for required columns
        required_columns = [
            'start_time', 'end_time', 'text',
            'sentiment_score', 'highlight_score',
            'excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]

        # If columns are missing, try to load the sentiment file instead
        if missing_columns:
            logger.warning(f"Missing columns in data: {missing_columns}")

            # Try to load the sentiment file
            sentiment_file = Path(f'Output/Analysis/Audio/audio_{video_id}_sentiment.csv')
            if os.path.exists(sentiment_file):
                logger.info(f"Loading sentiment data from {sentiment_file}")
                data = pd.read_csv(sentiment_file)
                data.columns = data.columns.str.strip()

                # Check again for required columns
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    logger.error(f"Still missing columns after loading sentiment file: {missing_columns}")

                    # Add missing columns with default values
                    for col in missing_columns:
                        if col in ['sentiment_score', 'highlight_score']:
                            data[col] = 0.0
                        elif col in ['excitement', 'funny', 'happiness', 'anger', 'sadness']:
                            data[col] = 0.0
                        elif col == 'neutral':
                            data[col] = 1.0
            else:
                logger.warning(f"Sentiment file not found: {sentiment_file}")

                # Add missing columns with default values
                for col in missing_columns:
                    if col in ['sentiment_score', 'highlight_score']:
                        data[col] = 0.0
                    elif col in ['excitement', 'funny', 'happiness', 'anger', 'sadness']:
                        data[col] = 0.0
                    elif col == 'neutral':
                        data[col] = 1.0

        # Calculate a combined emotion score
        data['emotion_intensity'] = data[['excitement', 'funny', 'happiness', 'anger', 'sadness']].max(axis=1)

        # Create weighted highlight score (now including audio intensity)
        data['weighted_highlight_score'] = (
            data['highlight_score'] * 0.5 +      # Base highlight score
            data['emotion_intensity'] * 0.2 +     # Emotion contribution
            abs(data['sentiment_score']) * 0.1 +  # Sentiment intensity contribution
            audio_intensity * 0.2                 # Audio intensity contribution
        )

        # Find peaks in weighted highlight score
        peaks, properties = find_peaks(
            data['weighted_highlight_score'],
            distance=20,        # Minimum samples between peaks
            prominence=0.3,     # Minimum prominence of peaks
            height=0.4         # Minimum height of peaks
        )

        # Create DataFrame with peak moments
        peak_moments = data.iloc[peaks].copy()
        peak_moments['prominence'] = properties['prominences']
        peak_moments['audio_intensity'] = audio_intensity[peaks]

        # Sort by weighted_highlight_score
        peak_moments = peak_moments.sort_values(
            by='weighted_highlight_score',
            ascending=False
        )

        # Select top 10 moments
        top_highlights = peak_moments.head(10)

        # Add duration column
        top_highlights['duration'] = top_highlights['end_time'] - top_highlights['start_time']

        # Select and reorder columns for output
        output_columns = [
            'start_time', 'end_time', 'duration', 'text',
            'weighted_highlight_score', 'highlight_score',
            'emotion_intensity', 'sentiment_score', 'audio_intensity'
        ]
        top_highlights = top_highlights[output_columns]

        # Save to CSV
        output_file = output_dir / f"audio_{video_id}_top_highlights.csv"
        top_highlights.to_csv(output_file, index=False)
        logger.info(f"Top 10 highlights saved to {output_file}")

        # List of emotions to analyze
        emotions = ['excitement', 'funny', 'happiness', 'anger', 'sadness']

        # Process each emotion
        for emotion in emotions:
            # Create weighted emotion score including audio intensity
            data[f'weighted_{emotion}_score'] = (
                data[emotion] * 0.7 +          # Base emotion score
                audio_intensity * 0.3          # Audio intensity contribution
            )

            # Find peaks for this emotion
            emotion_peaks, emotion_properties = find_peaks(
                data[f'weighted_{emotion}_score'],
                distance=20,        # Minimum samples between peaks
                prominence=0.2,     # Lower prominence threshold for emotions
                height=0.3         # Lower height threshold for emotions
            )

            # Create DataFrame with emotion peak moments
            emotion_peak_moments = data.iloc[emotion_peaks].copy()
            emotion_peak_moments['prominence'] = emotion_properties['prominences']
            emotion_peak_moments['audio_intensity'] = audio_intensity[emotion_peaks]

            # Sort by weighted emotion score
            emotion_peak_moments = emotion_peak_moments.sort_values(
                by=f'weighted_{emotion}_score',
                ascending=False
            )

            # Select top 5 moments
            top_emotion_moments = emotion_peak_moments.head(5)

            # Add duration column
            top_emotion_moments['duration'] = top_emotion_moments['end_time'] - top_emotion_moments['start_time']

            # Select columns for output
            emotion_output_columns = [
                'start_time', 'end_time', 'duration', 'text',
                f'weighted_{emotion}_score', emotion,
                'audio_intensity', 'prominence'
            ]
            top_emotion_moments = top_emotion_moments[emotion_output_columns]

            # Save to CSV
            emotion_output_file = output_dir / f"audio_{video_id}_top_{emotion}_moments.csv"
            top_emotion_moments.to_csv(emotion_output_file, index=False)
            logger.info(f"Top 5 {emotion} moments saved to {emotion_output_file}")

        return top_highlights

    except Exception as e:
        logger.error(f"Error in analyze_transcription_highlights: {str(e)}")
        return None
