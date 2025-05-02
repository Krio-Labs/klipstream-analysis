# KlipStream Analysis

A comprehensive pipeline for analyzing Twitch VODs, extracting insights from audio, transcripts, and chat data.

## Overview

KlipStream Analysis is a tool that processes Twitch VODs to extract valuable insights through various analysis techniques:

- Audio transcription and sentiment analysis
- Chat log analysis and sentiment detection
- Highlight detection and extraction
- Metadata processing and organization

The pipeline can be run as a Cloud Function or locally, with results stored in Convex or Google Cloud Storage.

## Features

- Download and process Twitch VODs
- Transcribe audio content
- Analyze sentiment in both audio transcripts and chat logs
- Detect highlights and key moments
- Process and organize chat data
- Upload results to cloud storage

## Getting Started

### Prerequisites

- Python 3.8+
- FFmpeg
- Google Cloud SDK (for cloud deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/klipstream-analysis.git
cd klipstream-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Usage

#### Local Execution

To process a Twitch VOD locally:

```bash
python main.py https://www.twitch.tv/videos/YOUR_VIDEO_ID
```

#### Cloud Function Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions on deploying as a Cloud Function.

## Project Structure

- `main.py`: Main entry point for the pipeline
- `raw_pipeline/`: Modules for initial VOD processing
- `audio_analysis.py`: Audio analysis functions
- `audio_sentiment.py`: Audio sentiment analysis
- `chat_processor.py`: Chat data processing
- `chat_sentiment.py`: Chat sentiment analysis
- `chat_analysis.py`: Chat analysis functions
- `utils/`: Utility functions and helpers

## License

This project is licensed under the MIT License - see the LICENSE file for details.