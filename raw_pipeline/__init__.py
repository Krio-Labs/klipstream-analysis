"""
Raw Pipeline Package

This package handles the raw file processing for Twitch VODs, including:
- Video downloading and audio extraction
- Transcript generation
- Waveform generation
- Chat downloading
- File uploading to Google Cloud Storage

All raw files are stored in the Output/Raw directory structure.
"""

from .processor import process_raw_files

__all__ = ['process_raw_files']
