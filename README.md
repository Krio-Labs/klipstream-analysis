# KlipStream Analysis

KlipStream Analysis is a comprehensive system for analyzing Twitch VODs, extracting insights from both streamer content and audience reactions, and identifying highlight-worthy moments. The system processes video, audio, transcripts, and chat data to generate detailed analysis that can be used to create highlight clips and understand viewer engagement.

## System Overview

The KlipStream Analysis system consists of two main pipelines:

1. **Raw Pipeline**: Downloads and processes raw data from Twitch VODs
2. **Analysis Pipeline**: Analyzes the processed data to extract insights and identify highlights

All processed files are uploaded to Google Cloud Storage buckets with a structured organization by VOD ID:
- `klipstream-vods-raw`: Video, audio, and waveform files
- `klipstream-transcripts`: Transcript files and segment data
- `klipstream-chatlogs`: Chat log files
- `klipstream-analysis`: Integrated analysis files

## Raw Pipeline

The raw pipeline handles the initial data acquisition and processing for Twitch VODs.

### Components

- **Downloader**: Downloads Twitch VODs and extracts audio using TwitchDownloaderCLI
- **Transcriber**: Transcribes audio using Deepgram with the nova-3 model
- **Sliding Window Generator**: Creates time-based segments (60-second windows with 30-second overlap)
- **Waveform Generator**: Creates audio waveform data for visualization
- **Chat Downloader**: Downloads and processes Twitch chat logs
- **Uploader**: Uploads all raw files to Google Cloud Storage buckets

### Data Flow

1. The pipeline starts with a Twitch VOD URL
2. The downloader extracts the video ID and downloads the VOD
3. Audio is extracted from the video and saved as a WAV file
4. The transcriber processes the audio to generate transcript files
5. The sliding window generator creates segment files for analysis
6. The waveform generator creates audio visualization data
7. The chat downloader retrieves and processes chat logs
8. All files are uploaded to their respective GCS buckets
9. Temporary directories (downloads and data) are cleaned up

### Outputs

The raw pipeline generates the following files in the `output/Raw` directory:
- `Videos/video_{video_id}.mp4`: The downloaded Twitch VOD
- `Audio/audio_{video_id}.wav`: Extracted audio in WAV format
- `Transcripts/audio_{video_id}_transcript.json`: Full transcript data from Deepgram
- `Transcripts/audio_{video_id}_segments.csv`: Time-segmented transcript data
- `Waveforms/audio_{video_id}_waveform.csv`: Audio waveform data
- `Chat/{video_id}_chat.csv`: Processed chat log data

## Analysis Pipeline

The analysis pipeline processes the raw data to extract insights and identify highlights.

### Components

- **Audio Sentiment Analysis**: Analyzes transcript segments for sentiment and emotions using Nebius API
- **Audio Highlight Analysis**: Identifies potential highlights based on audio features
- **Chat Processing**: Processes chat data for sentiment and engagement metrics
- **Chat Sentiment Analysis**: Analyzes chat for emotional reactions
- **Chat Highlight Analysis**: Identifies potential highlights based on chat activity
- **Integration**: Combines audio and chat analysis for comprehensive insights

### Data Flow

1. The pipeline starts with the video ID from the raw pipeline
2. Chat data is processed to extract message patterns and metrics
3. Chat sentiment analysis identifies emotional reactions
4. Chat highlight analysis identifies moments of high engagement
5. Audio sentiment analysis processes transcript segments for emotions
6. Audio highlight analysis identifies potential highlights based on audio features
7. The integration module combines audio and chat analysis
8. Final integrated analysis is uploaded to the GCS analysis bucket

### Outputs

The analysis pipeline generates the following files in the `output/Analysis` directory:
- `Audio/{video_id}_sentiment.csv`: Audio sentiment analysis results
- `Audio/{video_id}_highlights.csv`: Audio-based highlight analysis
- `Audio/{video_id}_*.png`: Various visualization plots
- `Chat/{video_id}_highlight_analysis.csv`: Chat-based highlight analysis
- `{video_id}_integrated_analysis.csv`: Combined audio and chat analysis

## Running the Pipelines

### Prerequisites

- Python 3.8+
- Required Python packages (see `requirements.txt`)
- Google Cloud Storage account with configured buckets
- Deepgram API key
- Nebius API key for sentiment analysis
- Git LFS (for downloading large binary files)

### Environment Setup

1. Install Git LFS if you don't have it already:
   - Download from [git-lfs.github.com](https://git-lfs.github.com/)
   - Or install via package manager:
     - macOS: `brew install git-lfs`
     - Ubuntu/Debian: `sudo apt install git-lfs`
     - Windows: `winget install -e --id GitHub.GitLFS`

2. Clone the repository with Git LFS:
   ```bash
   git lfs install
   git clone https://github.com/Krio-Labs/klipstream-analysis.git
   cd klipstream-analysis
   git lfs pull
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Create a `.env` file with your API keys:
   ```
   DEEPGRAM_API_KEY=your_deepgram_api_key
   NEBIUS_API_KEY=your_nebius_api_key
   GCP_SERVICE_ACCOUNT_PATH=path_to_your_service_account_key.json
   ```

7. Verify binary files were downloaded correctly:
   ```bash
   # Check TwitchDownloaderCLI and FFMPEG binaries
   ls -la raw_pipeline/bin/

   # Check model files
   ls -la analysis_pipeline/chat/models/
   ```

### Running the Complete Pipeline

To run the complete pipeline (raw + analysis):

```bash
python main.py https://www.twitch.tv/videos/YOUR_VIDEO_ID
```

### Running Individual Pipelines

To run only the raw pipeline:

```bash
python run_raw_pipeline.py https://www.twitch.tv/videos/YOUR_VIDEO_ID
```

To run only the analysis pipeline (requires raw pipeline to have been run first):

```bash
python run_analysis_pipeline.py YOUR_VIDEO_ID [--concurrency N] [--timeout N]
```

Optional parameters:
- `--concurrency`: Maximum number of concurrent API requests for sentiment analysis
- `--timeout`: API request timeout in seconds

## Binary Files and Models

The repository includes several large binary files that are managed using Git LFS:

### TwitchDownloaderCLI

The system uses TwitchDownloaderCLI to download Twitch VODs and chat logs. Platform-specific binaries are included:
- `raw_pipeline/bin/TwitchDownloaderCLI` (Linux)
- `raw_pipeline/bin/TwitchDownloaderCLI_mac` (macOS)
- `raw_pipeline/bin/TwitchDownloaderCLI.exe` (Windows)

### FFMPEG

FFMPEG is used for audio extraction and processing. Platform-specific binaries are included:
- `raw_pipeline/bin/ffmpeg` (Linux)
- `raw_pipeline/bin/ffmpeg_mac` (macOS)
- `raw_pipeline/bin/ffmpeg.exe` (Windows)

### Model Files

The chat analysis pipeline uses pre-trained machine learning models:
- `analysis_pipeline/chat/models/emotion_classifier_pipe_lr.pkl`: Classifies chat messages by emotion
- `analysis_pipeline/chat/models/highlight_classifier_pipe_lr.pkl`: Identifies potential highlights from chat

## Configuration Options

The system can be configured through environment variables and command-line parameters:

### Environment Variables

- `DEEPGRAM_API_KEY`: API key for Deepgram transcription
- `NEBIUS_API_KEY`: API key for Nebius sentiment analysis
- `GCP_SERVICE_ACCOUNT_PATH`: Path to Google Cloud service account key file

### Directory Structure

The system uses the following directory structure:
- `output/Raw`: Raw files from the raw pipeline
- `output/Analysis`: Analysis files from the analysis pipeline
- `downloads`: Temporary directory for downloaded files
- `data`: Temporary directory for intermediate data

### Google Cloud Storage Buckets

The system uses the following GCS buckets:
- `klipstream-vods-raw`: For video, audio, and waveform files
- `klipstream-transcripts`: For transcript files
- `klipstream-chatlogs`: For chat log files
- `klipstream-analysis`: For integrated analysis files

## Testing

For testing purposes, you can use the following Twitch VOD URL:
```
https://www.twitch.tv/videos/2434635255
```

## Cloud Function Deployment

The system can be deployed as a Google Cloud Function using the provided deployment script:

```bash
./deploy_cloud_function.sh
```

This script:
1. Reads configuration from `.env.yaml`
2. Deploys a 2nd generation Cloud Function with:
   - 16GB memory
   - 60-minute timeout
   - HTTP trigger
   - `run_pipeline` entry point
3. Configures the service account with necessary permissions

### Cloud Function Limitations

When deploying as a Cloud Function, be aware of these limitations:

- **Execution Time**: Maximum 60 minutes (3600 seconds)
- **Memory**: Maximum 16GB RAM
- **Disk Space**: Maximum 10GB in `/tmp` directory
- **Cold Starts**: Functions that are not frequently used may experience cold starts
- **Concurrency**: Limited ability to handle multiple concurrent requests

For longer or more resource-intensive workloads, consider using Cloud Run instead.

For more detailed deployment instructions, see [decision_docs/DEPLOYMENT.md](decision_docs/DEPLOYMENT.md).
