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
        # Import here to avoid circular imports
        from utils.config import BASE_DIR, USE_GCS
        from utils.file_manager import FileManager

        # Initialize file manager
        file_manager = FileManager(video_id)

        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # First try to use the sentiment file from the output directory
        sentiment_file = output_dir / f"audio_{video_id}_sentiment.csv"

        # Also check for sentiment file in test_output directory (for testing)
        test_sentiment_file = Path(f"test_output/audio_{video_id}_sentiment.csv")

        # Try to get the segments file using the file manager
        segments_file = file_manager.get_file_path("segments")

        # Log the paths we're checking
        logger.info(f"Checking sentiment file: {sentiment_file}")
        logger.info(f"Checking test sentiment file: {test_sentiment_file}")
        logger.info(f"Checking segments file: {segments_file}")

        # Check which file exists and use it
        if sentiment_file.exists():
            file_path = sentiment_file
            logger.info(f"Using sentiment file from output directory: {file_path}")
        elif test_sentiment_file.exists():
            file_path = test_sentiment_file
            logger.info(f"Using sentiment file from test directory: {file_path}")
        elif segments_file and segments_file.exists():
            file_path = segments_file
            logger.info(f"Using segments file from file manager: {file_path}")
        else:
            # Try alternative paths for segments file
            alt_segments_file = Path(f'output/Raw/Transcripts/audio_{video_id}_segments.csv')
            tmp_segments_file = Path(f'/tmp/output/Raw/Transcripts/audio_{video_id}_segments.csv')

            if alt_segments_file.exists():
                file_path = alt_segments_file
                logger.info(f"Using segments file from alternative path: {file_path}")
            elif tmp_segments_file.exists():
                file_path = tmp_segments_file
                logger.info(f"Using segments file from tmp path: {file_path}")
            else:
                # Try to download from GCS as a last resort
                if USE_GCS and file_manager.download_from_gcs("segments"):
                    file_path = file_manager.get_local_path("segments")
                    logger.info(f"Downloaded segments file from GCS: {file_path}")
                else:
                    raise FileNotFoundError(f"Could not find sentiment or segments file for video ID: {video_id}")

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
            # Check if we have pre-calculated audio metrics
            has_nebius_features = all(col in df.columns for col in ['speech_rate', 'absolute_intensity', 'relative_intensity'])

            # Even if we have pre-calculated metrics, we still need to load the audio file for the waveform plot
            # Try to get the audio file using the file manager
            audio_path = file_manager.get_file_path("audio")

            if audio_path and os.path.exists(audio_path):
                y, sr = librosa.load(audio_path)
                logger.info(f"Using audio file from file manager: {audio_path}")
            else:
                # Try alternative paths
                alt_audio_path = Path(f'output/Raw/Audio/audio_{video_id}.wav')
                tmp_audio_path = Path(f'/tmp/output/Raw/Audio/audio_{video_id}.wav')

                if os.path.exists(alt_audio_path):
                    y, sr = librosa.load(alt_audio_path)
                    logger.info(f"Using audio file from alternative path: {alt_audio_path}")
                elif os.path.exists(tmp_audio_path):
                    y, sr = librosa.load(tmp_audio_path)
                    logger.info(f"Using audio file from tmp path: {tmp_audio_path}")
                else:
                    # Try to download from GCS as a last resort
                    if USE_GCS and file_manager.download_from_gcs("audio"):
                        audio_path = file_manager.get_local_path("audio")
                        y, sr = librosa.load(audio_path)
                        logger.info(f"Downloaded audio file from GCS: {audio_path}")
                    else:
                        raise FileNotFoundError(f"Could not find audio file for video ID: {video_id}")

            times = np.arange(len(y)) / sr

            # Plot waveform
            ax.plot(times, y, color='blue', alpha=0.7, linewidth=1)
            ax.set_title('Audio Waveform')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)

            # Free memory
            del y
            gc.collect()
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

        # Check if we have pre-calculated audio metrics
        has_nebius_features = all(col in df.columns for col in ['speech_rate', 'absolute_intensity', 'relative_intensity'])

        if has_nebius_features:
            # Use pre-calculated absolute_intensity as our RMS equivalent
            logger.info("Using pre-calculated absolute_intensity for RMS")

            # Normalize to [0.5, 1.5] range for amplification
            # First ensure it's in [0, 1] range (it should be already)
            abs_intensity = df['absolute_intensity'].values
            abs_intensity_norm = np.clip(abs_intensity, 0, 1)

            # Then shift to [0.5, 1.5] range
            normalized_rms = abs_intensity_norm + 0.5
        else:
            # Need to calculate RMS from audio file
            logger.info("Calculating RMS from audio file")

            try:
                # Look for the audio file in the output directory
                audio_path = Path(f'output/Raw/Audio/audio_{video_id}.wav')

                if os.path.exists(audio_path):
                    y, sr = librosa.load(audio_path)
                else:
                    raise FileNotFoundError(f"Could not find audio file in {audio_path}")

                # Calculate the RMS energy envelope
                frame_length = int(sr * 0.03)  # 30ms frames
                hop_length = int(sr * 0.01)    # 10ms hop
                rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

                # Resample RMS to match emotion data length
                rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
                resampled_rms = np.interp(df['start_time'], rms_times, rms)

                # Normalize RMS to [0.5, 1.5] range for amplification
                normalized_rms = ((resampled_rms - resampled_rms.min()) /
                                (resampled_rms.max() - resampled_rms.min())) + 0.5

                # Free memory
                del y, sr, rms, rms_times, resampled_rms
                gc.collect()
            except Exception as e:
                logger.error(f"Error loading audio for RMS calculation: {str(e)}")
                # Create dummy RMS data
                normalized_rms = np.ones(len(df)) * 0.75  # Middle of our [0.5, 1.5] range

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

        # Adjust layout but don't save to reduce output files
        plt.tight_layout()
        # Skip saving the plot
        logger.info(f"Emotion analysis plots generated (not saving to file)")
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
        # Import here to avoid circular imports
        from utils.config import BASE_DIR, USE_GCS
        from utils.file_manager import FileManager

        # Initialize file manager
        file_manager = FileManager(video_id)

        # Define output directory
        if output_dir is None:
            output_dir = Path('output/Analysis/Audio')
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Determine input file path
        if input_file is None:
            # Try to get the segments file using the file manager
            input_file = file_manager.get_file_path("segments")
            if input_file is None:
                # Try alternative paths
                tmp_path = Path(f'/tmp/output/Raw/Transcripts/audio_{video_id}_segments.csv')
                alt_path = Path(f'output/Raw/Transcripts/audio_{video_id}_segments.csv')

                if os.path.exists(tmp_path):
                    input_file = tmp_path
                    logger.info(f"Using segments file from tmp path: {input_file}")
                elif os.path.exists(alt_path):
                    input_file = alt_path
                    logger.info(f"Using segments file from alternative path: {input_file}")
                else:
                    # Try to download from GCS as a last resort
                    if USE_GCS and file_manager.download_from_gcs("segments"):
                        input_file = file_manager.get_local_path("segments")
                        logger.info(f"Downloaded segments file from GCS: {input_file}")
                    else:
                        logger.error(f"Could not find or download segments file for video ID: {video_id}")
                        return None
        else:
            input_file = Path(input_file)
            logger.info(f"Using provided input file: {input_file}")

        # Check if file exists
        if not os.path.exists(input_file):
            logger.error(f"Required file not found: {input_file}")
            logger.info("Please run the full pipeline first to generate the required transcription file.")
            return None

        # Load transcription data
        data = pd.read_csv(input_file)
        data.columns = data.columns.str.strip()

        # Check if we have pre-calculated audio metrics from sentiment_nebius.py
        has_nebius_features = all(col in data.columns for col in ['speech_rate', 'absolute_intensity', 'relative_intensity'])

        if has_nebius_features:
            # Use pre-calculated audio metrics
            logger.info("Using pre-calculated audio metrics from sentiment file")

            # Use absolute_intensity directly as audio_intensity
            audio_intensity = data['absolute_intensity'].values

            # No need to load audio file or calculate features
            logger.info("Skipping audio file loading and feature extraction")
        else:
            # Need to calculate audio features from scratch
            logger.info("Pre-calculated audio metrics not found, calculating from audio file")

            try:
                # Try to get the audio file using the file manager
                audio_path = file_manager.get_file_path("audio")

                if audio_path and os.path.exists(audio_path):
                    y, sr = librosa.load(audio_path)
                    logger.info(f"Using audio file from file manager: {audio_path}")
                else:
                    # Try alternative paths
                    alt_audio_path = Path(f'output/Raw/Audio/audio_{video_id}.wav')
                    tmp_audio_path = Path(f'/tmp/output/Raw/Audio/audio_{video_id}.wav')
                    tmp_outputs_path = Path(f'/tmp/outputs/audio_{video_id}.wav')

                    if os.path.exists(alt_audio_path):
                        y, sr = librosa.load(alt_audio_path)
                        logger.info(f"Using audio file from alternative path: {alt_audio_path}")
                    elif os.path.exists(tmp_audio_path):
                        y, sr = librosa.load(tmp_audio_path)
                        logger.info(f"Using audio file from tmp path: {tmp_audio_path}")
                    elif os.path.exists(tmp_outputs_path):
                        y, sr = librosa.load(tmp_outputs_path)
                        logger.info(f"Using audio file from tmp outputs path: {tmp_outputs_path}")
                    else:
                        # Try to download from GCS as a last resort
                        if USE_GCS and file_manager.download_from_gcs("audio"):
                            audio_path = file_manager.get_local_path("audio")
                            y, sr = librosa.load(audio_path)
                            logger.info(f"Downloaded audio file from GCS: {audio_path}")
                        else:
                            raise FileNotFoundError(f"Could not find audio file for video ID: {video_id}")

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

                # Free memory
                del y, sr, rms, spectral_centroid, times, rms_interp, centroid_interp, rms_norm, centroid_norm
                gc.collect()

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
            sentiment_file = Path(f'output/Analysis/Audio/audio_{video_id}_sentiment.csv')
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

        # We already checked for Nebius features above, but we'll keep this for clarity
        # has_nebius_features is already defined

        if has_nebius_features:
            # If we have data from sentiment_nebius.py, use the highlight_score directly
            # with minimal adjustment for audio intensity variations
            logger.info("Using Nebius-generated highlight score with minimal adjustment")

            # The highlight_score from sentiment_nebius.py already incorporates:
            # - Model score (50%)
            # - Speech rate (15%)
            # - Relative loudness (15%)
            # - Absolute loudness (10%)
            # - Emotional intensity (10%)

            # We'll just make a small adjustment based on the current audio intensity
            # to account for any local variations not captured in the original calculation

            # Check if the lengths match
            if len(data) != len(audio_intensity):
                logger.warning(f"Length mismatch: data has {len(data)} rows, audio_intensity has {len(audio_intensity)} elements")

                # Resize audio_intensity to match data length
                if len(data) < len(audio_intensity):
                    logger.info(f"Truncating audio_intensity from {len(audio_intensity)} to {len(data)} elements")
                    audio_intensity = audio_intensity[:len(data)]
                else:
                    logger.info(f"Extending audio_intensity from {len(audio_intensity)} to {len(data)} elements")
                    # Extend audio_intensity by repeating the last value
                    extension = np.ones(len(data) - len(audio_intensity)) * audio_intensity[-1]
                    audio_intensity = np.concatenate([audio_intensity, extension])

            data['weighted_highlight_score'] = (
                data['highlight_score'] * 0.95 +    # Nebius highlight score (95%)
                audio_intensity * 0.05              # Current audio intensity (5%)
            )
        else:
            # For legacy data without Nebius features, use the original calculation
            logger.info("Using legacy highlight score calculation")

            # Check if the lengths match
            if len(data) != len(audio_intensity):
                logger.warning(f"Length mismatch: data has {len(data)} rows, audio_intensity has {len(audio_intensity)} elements")

                # Resize audio_intensity to match data length
                if len(data) < len(audio_intensity):
                    logger.info(f"Truncating audio_intensity from {len(audio_intensity)} to {len(data)} elements")
                    audio_intensity = audio_intensity[:len(data)]
                else:
                    logger.info(f"Extending audio_intensity from {len(audio_intensity)} to {len(data)} elements")
                    # Extend audio_intensity by repeating the last value
                    extension = np.ones(len(data) - len(audio_intensity)) * audio_intensity[-1]
                    audio_intensity = np.concatenate([audio_intensity, extension])

            data['weighted_highlight_score'] = (
                data['highlight_score'] * 0.5 +      # Base highlight score
                data['emotion_intensity'] * 0.2 +    # Emotion contribution
                abs(data['sentiment_score']) * 0.1 + # Sentiment intensity contribution
                audio_intensity * 0.2                # Audio intensity contribution
            )

        # Find peaks in weighted highlight score
        # Adjust peak detection parameters based on data source
        if has_nebius_features:
            # For Nebius data, use more sensitive parameters
            # since the highlight scores are already well-calibrated
            peaks, properties = find_peaks(
                data['weighted_highlight_score'],
                distance=15,        # Reduced minimum distance between peaks
                prominence=0.25,    # Lower prominence threshold
                height=0.35         # Lower height threshold
            )
            logger.info(f"Using Nebius-optimized peak detection (found {len(peaks)} peaks)")
        else:
            # For legacy data, use the original parameters
            peaks, properties = find_peaks(
                data['weighted_highlight_score'],
                distance=20,        # Minimum samples between peaks
                prominence=0.3,     # Minimum prominence of peaks
                height=0.4          # Minimum height of peaks
            )
            logger.info(f"Using legacy peak detection (found {len(peaks)} peaks)")

        # Create DataFrame with peak moments
        peak_moments = data.iloc[peaks].copy()
        peak_moments['prominence'] = properties['prominences']

        # Check if peaks indices are within audio_intensity bounds
        valid_indices = [i for i in peaks if i < len(audio_intensity)]
        if len(valid_indices) < len(peaks):
            logger.warning(f"Some peak indices are out of bounds for audio_intensity. Using valid indices only.")
            # If we have no valid indices, use zeros
            if len(valid_indices) == 0:
                peak_moments['audio_intensity'] = 0.0
            else:
                # Use the valid indices and fill any missing values with the mean
                peak_moments['audio_intensity'] = np.mean(audio_intensity[valid_indices])
        else:
            peak_moments['audio_intensity'] = audio_intensity[peaks]

        # Sort by weighted_highlight_score
        peak_moments = peak_moments.sort_values(
            by='weighted_highlight_score',
            ascending=False
        )

        # Select top 10 moments and create a copy to avoid SettingWithCopyWarning
        top_highlights = peak_moments.head(10).copy()

        # Add duration column
        top_highlights['duration'] = top_highlights['end_time'] - top_highlights['start_time']

        # Select and reorder columns for output - time columns first, then scores, text at the end
        output_columns = [
            'start_time', 'end_time', 'duration',
            'weighted_highlight_score', 'highlight_score',
            'emotion_intensity', 'sentiment_score', 'audio_intensity',
            'text'
        ]
        top_highlights = top_highlights[output_columns]

        # Save to CSV temporarily for integration
        output_file = output_dir / f"audio_{video_id}_top_highlights.csv"
        top_highlights.to_csv(output_file, index=False)
        logger.info(f"Top 10 highlights saved to {output_file} (will be removed after integration)")

        # List of emotions to analyze
        emotions = ['excitement', 'funny', 'happiness', 'anger', 'sadness']

        # Process each emotion
        for emotion in emotions:
            # Create weighted emotion score including audio intensity
            # Use consistent emotion score weighting for both Nebius and legacy data
            data[f'weighted_{emotion}_score'] = (
                data[emotion] * 0.7 +          # Base emotion score (70%)
                audio_intensity * 0.3          # Audio intensity contribution (30%)
            )

            if has_nebius_features:
                # For Nebius data, use more sensitive peak detection parameters
                emotion_peaks, emotion_properties = find_peaks(
                    data[f'weighted_{emotion}_score'],
                    distance=15,        # Reduced minimum distance
                    prominence=0.15,    # Lower prominence threshold
                    height=0.25         # Lower height threshold
                )
                logger.info(f"Using Nebius-optimized peak detection for {emotion} (found {len(emotion_peaks)} peaks)")
            else:
                # For legacy data, use the original parameters
                emotion_peaks, emotion_properties = find_peaks(
                    data[f'weighted_{emotion}_score'],
                    distance=20,        # Minimum samples between peaks
                    prominence=0.2,     # Lower prominence threshold for emotions
                    height=0.3          # Lower height threshold for emotions
                )
                logger.info(f"Using legacy peak detection for {emotion} (found {len(emotion_peaks)} peaks)")

            # Create DataFrame with emotion peak moments
            emotion_peak_moments = data.iloc[emotion_peaks].copy()
            emotion_peak_moments['prominence'] = emotion_properties['prominences']

            # Check if emotion_peaks indices are within audio_intensity bounds
            valid_indices = [i for i in emotion_peaks if i < len(audio_intensity)]
            if len(valid_indices) < len(emotion_peaks):
                logger.warning(f"Some emotion peak indices are out of bounds for audio_intensity. Using valid indices only.")
                emotion_peak_moments['audio_intensity'] = audio_intensity[valid_indices]
            else:
                emotion_peak_moments['audio_intensity'] = audio_intensity[emotion_peaks]

            # Sort by weighted emotion score
            emotion_peak_moments = emotion_peak_moments.sort_values(
                by=f'weighted_{emotion}_score',
                ascending=False
            )

            # Select top 5 moments and create a copy to avoid SettingWithCopyWarning
            top_emotion_moments = emotion_peak_moments.head(5).copy()

            # Add duration column
            top_emotion_moments['duration'] = top_emotion_moments['end_time'] - top_emotion_moments['start_time']

            # Select columns for output - time columns first, then scores, text at the end
            emotion_output_columns = [
                'start_time', 'end_time', 'duration',
                f'weighted_{emotion}_score', emotion,
                'audio_intensity', 'prominence',
                'text'
            ]
            top_emotion_moments = top_emotion_moments[emotion_output_columns]

            # Save to CSV temporarily for integration
            emotion_output_file = output_dir / f"audio_{video_id}_top_{emotion}_moments.csv"
            top_emotion_moments.to_csv(emotion_output_file, index=False)
            logger.info(f"Top 5 {emotion} moments saved to {emotion_output_file} (will be removed after integration)")

        return top_highlights

    except Exception as e:
        logger.error(f"Error in analyze_transcription_highlights: {str(e)}")
        return None
