"""
Audio Analysis Pipeline Package

This package provides tools for downloading, transcribing, and analyzing audio from Twitch streams.
It includes modules for:
- Audio downloading from Twitch
- Audio transcription using AssemblyAI
- Sentiment analysis using Google's Gemini
- Audio waveform visualization
- Advanced audio analysis and speech separation
"""

from .audio_downloader import TwitchVideoDownloader
from .audio_transcription import TranscriptionHandler
from .audio_sentiment import main as sentiment_analysis
from .audio_waveform import process_audio_file
from .audio_analysis import (
    analyze_chat_highlights,
    plot_audio_waveform,
    separate_speech_and_noise,
    plot_combined_loudness,
    plot_metrics,
    analyze_chat_intervals
)

__version__ = "1.0.0"
__author__ = "Aman Lohia"

# Export main classes and functions
__all__ = [
    'TwitchVideoDownloader',
    'TranscriptionHandler',
    'sentiment_analysis',
    'process_audio_file',
    'analyze_chat_highlights',
    'plot_audio_waveform',
    'separate_speech_and_noise',
    'plot_combined_loudness',
    'plot_metrics',
    'analyze_chat_intervals'
] 