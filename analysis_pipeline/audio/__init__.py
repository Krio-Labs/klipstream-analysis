"""
Audio Analysis Module

This module handles the analysis of audio transcriptions, including:
- Sentiment analysis
- Highlight detection
- Metrics visualization
"""

from .sentiment import analyze_audio_sentiment
from .analysis import analyze_transcription_highlights, plot_metrics

__all__ = ['analyze_audio_sentiment', 'analyze_transcription_highlights', 'plot_metrics']
