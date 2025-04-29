import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import librosa
import gc


def plot_metrics(output_dir, video_id):
    """
    Plot various metrics including sentiment, highlight score, and individual emotions.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Look for the CSV file in the output directory
        file_path = f'{output_dir}/audio_{video_id}_paragraphs.csv'

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logging.info(f"Using paragraphs file from: {file_path}")
        else:
            raise FileNotFoundError(f"Could not find paragraphs file in {file_path}")

        # Create figure with subplots
        emotions = ['excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral']
        num_plots = len(emotions) + 3  # +3 for waveform, sentiment and highlight

        fig, axs = plt.subplots(num_plots, 1, figsize=(15, 4*num_plots))

        # Plot 1: Audio Waveform
        ax = axs[0]
        try:
            # Look for the audio file in the output directory
            audio_path = f'{output_dir}/audio_{video_id}.wav'

            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path)
                logging.info(f"Using audio file from: {audio_path}")
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
            logging.error(f"Error plotting waveform: {str(e)}")
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
            audio_path = f'{output_dir}/audio_{video_id}.wav'

            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path)
            else:
                raise FileNotFoundError(f"Could not find audio file in {audio_path}")

            # Calculate the RMS energy envelope
            frame_length = int(sr * 0.03)  # 30ms frames
            hop_length = int(sr * 0.01)    # 10ms hop
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        except Exception as e:
            logging.error(f"Error loading audio for RMS calculation: {str(e)}")
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
        plt.savefig(os.path.join(output_dir, f'emotion_analysis_plots_{video_id}.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")
        raise

def analyze_transcription_highlights(input_file, output_dir):
    """
    Analyze transcription data to identify highlights and emotion peaks
    """
    try:
        # Check if file exists
        if not os.path.exists(input_file):
            logging.error(f"Required file not found: {input_file}")
            logging.info("Please run the full pipeline first to generate the required transcription file.")
            return None

        # Load transcription data
        data = pd.read_csv(input_file)
        data.columns = data.columns.str.strip()

        # Get video_id from input_file path
        video_id = os.path.basename(input_file).split('_')[1].split('.')[0]

        # Load audio file
        try:
            # Try to find the audio file in multiple locations
            audio_path = f'{output_dir}/audio_{video_id}.wav'
            tmp_audio_path = f'/tmp/outputs/audio_{video_id}.wav'

            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path)
                logging.info(f"Using audio file from: {audio_path}")
            elif os.path.exists(tmp_audio_path):
                y, sr = librosa.load(tmp_audio_path)
                logging.info(f"Using audio file from: {tmp_audio_path}")
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
            logging.warning(f"Could not load audio file: {str(e)}")
            audio_intensity = np.ones(len(data))  # Fallback if audio processing fails

        # Check for required columns
        required_columns = [
            'start_time', 'end_time', 'text',
            'sentiment_score', 'highlight_score',
            'excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing columns in data: {missing_columns}")
            return None

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
        output_file = os.path.join(output_dir, "top_highlights.csv")
        top_highlights.to_csv(output_file, index=False)
        logging.info(f"Top 10 highlights saved to {output_file}")

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
            emotion_output_file = os.path.join(output_dir, f"top_{emotion}_moments.csv")
            top_emotion_moments.to_csv(emotion_output_file, index=False)
            logging.info(f"Top 5 {emotion} moments saved to {emotion_output_file}")

        return top_highlights

    except Exception as e:
        logging.error(f"Error in analyze_transcription_highlights: {str(e)}")
        return None

def process_chunk(row):
    """Process a chunk of rows with emotion and sentiment data"""
    try:
        # Create result dictionary with specific column order
        result_dict = {
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'paragraph': row['paragraph'],
            'sentiment_score': row['sentiment_score'],
            'excitement': row['excitement'],
            'funny': row['funny'],
            'happiness': row['happiness'],
            'anger': row['anger'],
            'sadness': row['sadness'],
            'neutral': row['neutral']
        }

        return result_dict

    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def main():
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Audio analysis script')
        parser.add_argument('--video-id', type=str, required=True,
                          help='Video ID for processing')
        args = parser.parse_args()

        output_dir = '/tmp/outputs'
        os.makedirs(output_dir, exist_ok=True)

        logging.info("Starting audio analysis...")

        # Use video_id in file paths with .wav extension
        audio_path = os.path.join(output_dir, f'audio_{args.video_id}.wav')
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return

        # Load and analyze transcription data
        transcription_path = os.path.join(output_dir, f'audio_{args.video_id}_paragraphs.csv')
        if os.path.exists(transcription_path):
            # Analyze transcription highlights
            top_highlights = analyze_transcription_highlights(transcription_path, output_dir)
            if top_highlights is not None:
                logging.info("Successfully analyzed transcription highlights")

            # Generate emotion, sentiment, and highlight score plots
            plot_metrics(output_dir, args.video_id)
            logging.info("Generated emotion analysis plots")
        else:
            logging.error(f"Transcription file not found: {transcription_path}")

        logging.info("Analysis completed!")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
    finally:
        plt.close('all')
        gc.collect()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
