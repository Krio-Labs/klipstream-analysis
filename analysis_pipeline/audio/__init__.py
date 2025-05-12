"""
Audio Analysis Module

This module handles the analysis of audio transcriptions, including:
- Sentiment analysis (using either local models or Nebius API)
- Highlight detection
- Metrics visualization
"""

# Import sentiment analysis module and analysis functions
from .sentiment_nebius import analyze_audio_sentiment
from .analysis import analyze_transcription_highlights, plot_metrics

# Export all functions
__all__ = [
    'analyze_audio_sentiment',
    'analyze_transcription_highlights',
    'plot_metrics'
]
