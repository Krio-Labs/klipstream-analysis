"""
Analysis Pipeline Package

This package handles the analysis of raw files for Twitch VODs, including:
- Audio sentiment analysis
- Audio transcription analysis
- Chat processing and analysis
- Chat sentiment analysis

All analysis files are stored in the Output/Analysis directory structure.
"""

from .processor import process_analysis

__all__ = ['process_analysis']
