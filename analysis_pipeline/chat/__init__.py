"""
Chat Analysis Module

This module handles the analysis of Twitch chat data, including:
- Chat processing and normalization
- Sentiment analysis
- Interval analysis
- Highlight detection
"""

from .processor import process_chat_data
from .sentiment import analyze_chat_sentiment
from .analysis import analyze_chat_intervals, analyze_chat_highlights

__all__ = ['process_chat_data', 'analyze_chat_sentiment', 'analyze_chat_intervals', 'analyze_chat_highlights']
