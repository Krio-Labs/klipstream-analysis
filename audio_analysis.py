import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline
import matplotlib.dates as mdates
import os
import librosa
import soundfile as sf
import gc


def plot_metrics(interval_stats, output_dir, video_id):
    """
    Plot various metrics including sentiment, highlight score, and individual emotions.
    """
    try:
        # Read the CSV file directly
        df = pd.read_csv(f'outputs/audio_{video_id}_paragraphs.csv')
        
        # Create figure with subplots
        emotions = ['excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral']
        num_plots = len(emotions) + 3  # +3 for waveform, sentiment and highlight
        
        fig, axs = plt.subplots(num_plots, 1, figsize=(15, 4*num_plots))
        
        # Plot 1: Audio Waveform
        ax = axs[0]
        try:
            # Load audio file
            y, sr = librosa.load(f'outputs/audio_{video_id}.mp3')
            times = np.arange(len(y)) / sr
            
            # Plot waveform
            ax.plot(times, y, color='blue', alpha=0.7, linewidth=1)
            ax.set_title('Audio Waveform')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error plotting waveform: {str(e)}")
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

        # Individual emotion plots with moving averages and area plots
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FF99CC', '#99CCFF', '#FFB366']
        for i, (emotion, color) in enumerate(zip(emotions, colors)):
            ax = axs[i + 3]
            smoothed_emotion = df[emotion].rolling(window=window, center=True).mean()
            
            # Create area plot
            ax.fill_between(df['start_time'], 0, df[emotion], 
                          alpha=0.2, color=color)
            
            # Plot smoothed data
            ax.plot(df['start_time'], smoothed_emotion,
                   label=f'Moving Average ({window} intervals)',
                   color=color,
                   linewidth=2)
            
            ax.set_title(f'{emotion.capitalize()} Over Time')
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
        print(f"Error during plotting: {str(e)}")

def analyze_chat_intervals(input_file, output_dir, interval_seconds=30):
    """
    Analyze chat messages in intervals and calculate statistics.
    """
    try:
        # Read the data
        df = pd.read_csv(input_file)
        
        # Use start_time instead of time column
        if 'time' not in df.columns and 'start_time' in df.columns:
            df['time'] = df['start_time']  # Create time column from start_time
        
        # Convert time to numeric for binning
        df['time'] = pd.to_numeric(df['time'])
        
        # Create interval bins
        max_time = df['time'].max()
        bins = range(0, int(max_time) + interval_seconds, interval_seconds)
        labels = [f"({bins[i]}, {bins[i+1]}]" for i in range(len(bins)-1)]
        df['interval'] = pd.cut(df['time'], bins=bins, labels=labels)
        
        # Calculate statistics for each interval
        interval_stats = df.groupby('interval', observed=True).agg({
            'paragraph': 'count',  # Changed from 'message' to 'paragraph'
            'sentiment_score': ['mean', 'std'],
            'excitement': 'mean',
            'funny': 'mean',
            'happiness': 'mean',
            'anger': 'mean',
            'sadness': 'mean',
            'neutral': 'mean'
        }).round(3)
        
        # Flatten column names
        interval_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                for col in interval_stats.columns]
        
        # Rename count column
        interval_stats = interval_stats.rename(columns={'paragraph_count': 'message_count'})
        
        return interval_stats
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

def analyze_chat_highlights(input_file, output_dir):
    """
    Analyze chat data to identify highlight moments and emotional peaks.

    Args:
        input_file (str): Path to the input CSV file containing chat analysis data.
        output_dir (str): Directory where output files should be saved.
    """
    # Check if file exists
    if not os.path.exists(input_file):
        logging.error(f"Required file not found: {input_file}")
        logging.info("Please run the full pipeline with a VOD URL first to generate the required data files.")
        return None

    # Load data
    data = pd.read_csv(input_file)

    # Remove any leading/trailing whitespaces in column names
    data.columns = data.columns.str.strip()

    # Display available columns
    print("Available columns in the original data:", list(data.columns))

    # Check for missing necessary columns
    required_columns = [
        'time', 'sentiment_score', 'highlight_score',
        'excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral', 'message'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns in data: {missing_columns}")
        return None

    # Group by 'time' and aggregate the necessary columns
    grouped_data = data.groupby('time').agg({
        'sentiment_score': 'mean',
        'highlight_score': 'mean',
        'excitement': 'mean',
        'funny': 'mean',
        'happiness': 'mean',
        'anger': 'mean',
        'sadness': 'mean',
        'neutral': 'mean',
        'message': 'count'  # Count the number of messages per time window
    }).rename(columns={'message': 'message_count'})

    # Convert time to minutes for better readability in plots
    grouped_data['time_minutes'] = grouped_data.index / 60

    # Compute weighted_highlight_score based on highlight_score and message_count
    grouped_data['weighted_highlight_score'] = (
        grouped_data['highlight_score'] * 0.7 +  # Weight for highlight_score
        grouped_data['message_count'] * 0.3      # Weight for message_count (chat density)
    )

    # Detect peaks in weighted_highlight_score
    peaks, properties = find_peaks(
        grouped_data['weighted_highlight_score'],
        prominence=0.7,    # Adjusted prominence
        height=0.1,        # Adjusted minimum height
        width=1            # Minimum width of 1 window
    )

    # Create a DataFrame with all peak moments
    peak_windows = grouped_data.iloc[peaks].copy()
    peak_windows['prominence'] = properties['prominences']

    # Sort peaks by weighted_highlight_score first, then prominence
    peak_windows = peak_windows.sort_values(
        by=['weighted_highlight_score', 'prominence'],
        ascending=[False, False]
    )

    # Function to filter peaks with minimum time distance (10 minutes)
    def filter_peaks_with_distance(peaks_df, min_distance_secs=600):
        selected_peaks = []
        used_times = []

        for idx, row in peaks_df.iterrows():
            current_time = idx  # Using raw time value
            if not any(abs(current_time - used_time) < min_distance_secs for used_time in used_times):
                selected_peaks.append(idx)
                used_times.append(current_time)

        return peaks_df.loc[selected_peaks]

    # Get top peaks while maintaining minimum 10 minutes distance
    top_windows = filter_peaks_with_distance(peak_windows, min_distance_secs=600).head(10)

    # Sort final results by time for clearer output
    top_windows = top_windows.sort_values('time')

    # Save top highlights to CSV
    top_highlights_file = os.path.join(output_dir, "top_highlights.csv")
    top_windows.to_csv(top_highlights_file)
    print(f"Top highlights saved to {top_highlights_file}")

    # Identify and save top 5 prominent time regions for each emotion
    emotions_of_interest = ['funny', 'anger', 'excitement', 'happiness', 'sadness']
    for emotion in emotions_of_interest:
        # Detect peaks in the specific emotion
        emotion_peaks, emotion_properties = find_peaks(
            grouped_data[emotion],
            prominence=0.5,    # Adjusted prominence for emotion
            height=0.1,        # Adjusted minimum height for emotion
            width=1            # Minimum width of 1 window
        )

        # Create a DataFrame with emotion peak moments
        emotion_peak_windows = grouped_data.iloc[emotion_peaks].copy()
        emotion_peak_windows['prominence'] = emotion_properties['prominences']

        # Sort peaks by emotion score first, then prominence
        emotion_peak_windows = emotion_peak_windows.sort_values(
            by=[emotion, 'prominence'],
            ascending=[False, False]
        )

        # Filter peaks with minimum time distance
        top_emotion_peaks = filter_peaks_with_distance(emotion_peak_windows, min_distance_secs=600).head(5)

        # Sort by time
        top_emotion_peaks = top_emotion_peaks.sort_values('time')

        # Save to CSV
        emotion_file = os.path.join(output_dir, f"top_5_{emotion}_regions.csv")
        top_emotion_peaks.to_csv(emotion_file)
        print(f"Top 5 {emotion.capitalize()} regions saved to {emotion_file}")

    # Plotting the Smoothened Highlight Score
    plt.figure(figsize=(15, 7))
    
    # Get time in minutes for x-axis
    x = grouped_data.index / 60  # Convert seconds to minutes
    y = grouped_data['weighted_highlight_score']

    # Prepare data for smoothing
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    spline = make_interp_spline(x, y, k=1)  # Linear interpolation
    y_smooth = spline(x_smooth)

    # Plot the smooth line
    plt.plot(x_smooth, y_smooth, '-', 
            label='Weighted Highlight Score (Smoothed)', color='blue', alpha=0.7)

    # Read top highlights from CSV and mark them
    top_highlights = pd.read_csv('twitch_chat_analysis/outputs/top_highlights.csv')
    
    # Mark peaks with X and highlight windows
    for _, highlight in top_highlights.iterrows():
        peak_minutes = highlight['time'] / 60  # Convert seconds to minutes
        # Add yellow highlight window
        plt.axvspan(peak_minutes - 2.5, peak_minutes + 2.5,  # +/- 2.5 minutes
                   color='yellow', alpha=0.2)
        # Add X marker at peak
        plt.plot(peak_minutes, 
                highlight['weighted_highlight_score'],
                'rx', markersize=12, markeredgewidth=2,
                label='_nolegend_')  # Add large red X marker

    # Configure axes
    plt.title("Smoothened Weighted Highlight Score Over Time")
    plt.xlabel("Stream Time (minutes)")
    plt.ylabel("Weighted Highlight Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "highlight_score_smooth_plot.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Smoothened highlight score plot saved to {plot_file}")

def process_chunk(rows):
    """Process a chunk of rows with emotion and sentiment data"""
    try:
        # Define emotion clusters
        emotion_clusters = {
            'excitement': ['excitement'],  # Already matches column
            'funny': ['funny'],           # Already matches column 
            'happiness': ['happiness'],    # Already matches column
            'anger': ['anger'],           # Already matches column
            'sadness': ['sadness'],       # Already matches column
            'neutral': ['neutral']        # Already matches column
        }
        
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

def plot_audio_waveform(audio_path, output_dir):
    """
    Plot the waveform of an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save the plot
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path)
        
        # Create time array
        time = np.arange(len(y)) / sr
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot waveform
        plt.plot(time, y, color='blue', alpha=0.7)
        
        # Customize plot
        plt.title('Audio Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'audio_waveform.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Waveform plot saved to {output_path}")
        
    except Exception as e:
        print(f"Error plotting waveform: {str(e)}")

def separate_speech_and_noise(audio_path, output_dir):
    """
    Separate speech from background noise and save the files without creating spectrograms
    """
    try:
        print("Starting speech separation...")
        
        # Load the audio file with a lower sampling rate
        print("Loading audio file...")
        y, sr = librosa.load(audio_path, sr=22050)  # Lower sampling rate
        
        # Compute the spectrogram
        print("Computing spectrogram...")
        D = librosa.stft(y)
        
        # Compute a mask for speech frequencies
        print("Creating frequency mask...")
        freqs = librosa.fft_frequencies(sr=sr)
        mask = (freqs >= 100) & (freqs <= 3000)
        mask = mask[:, np.newaxis]
        
        # Apply the mask
        print("Applying mask...")
        D_speech = D * mask
        D_noise = D * ~mask
        
        # Convert back to time domain
        print("Converting back to time domain...")
        y_speech = librosa.istft(D_speech)
        y_noise = librosa.istft(D_noise)
        
        # Save separated audio files
        print("Saving audio files...")
        speech_path = os.path.join(output_dir, 'speech.wav')
        noise_path = os.path.join(output_dir, 'noise.wav')
        
        sf.write(speech_path, y_speech, sr)
        sf.write(noise_path, y_noise, sr)
        
        print("Speech separation completed successfully!")
        return speech_path, noise_path
        
    except Exception as e:
        print(f"Error in speech separation: {str(e)}")
        return None, None
    finally:
        # Force garbage collection
        import gc
        gc.collect()

def plot_combined_loudness(original_path, speech_path, noise_path, output_dir):
    """
    Plot the loudness of original, speech, and noise audio in one figure
    """
    try:
        plt.figure(figsize=(15, 8))
        
        # Function to process each audio file
        def process_audio(file_path, label, color):
            y, sr = librosa.load(file_path)
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
            times = librosa.times_like(rms, sr=sr, hop_length=256)
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            plt.plot(times, rms_db, color=color, alpha=0.7, linewidth=1.5, label=label)
            return rms_db
            
        # Process each audio file
        print("Processing original audio...")
        orig_db = process_audio(original_path, 'Original', '#2E86C1')
        
        print("Processing speech audio...")
        speech_db = process_audio(speech_path, 'Speech', '#28B463')
        
        print("Processing noise audio...")
        noise_db = process_audio(noise_path, 'Noise', '#E74C3C')
        
        # Customize plot
        plt.title('Combined Audio Loudness Analysis', fontsize=12, pad=15)
        plt.xlabel('Time (seconds)', fontsize=10)
        plt.ylabel('Loudness (dB)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='upper right', framealpha=0.9)
        
        # Set y-axis limits with some padding
        min_db = min(np.min(orig_db), np.min(speech_db), np.min(noise_db))
        max_db = max(np.max(orig_db), np.max(speech_db), np.max(noise_db))
        plt.ylim(min_db - 5, max_db + 5)
        
        # Save and cleanup
        output_path = os.path.join(output_dir, 'combined_loudness.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Combined loudness plot saved to {output_path}")
        
    except Exception as e:
        print(f"Error plotting combined loudness: {str(e)}")
    finally:
        plt.close('all')

def main():
    try:
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting audio analysis...")
        
        # First separate speech and noise
        speech_path, noise_path = separate_speech_and_noise('outputs/audio.mp3', output_dir)
        
        if speech_path and noise_path:
            # Then create combined loudness plot
            plot_combined_loudness('outputs/audio.mp3', speech_path, noise_path, output_dir)
        
        print("Analysis completed!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        plt.close('all')
        gc.collect()

if __name__ == "__main__":
    main()
