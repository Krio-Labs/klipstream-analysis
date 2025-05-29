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

- **Downloader**: Downloads Twitch VODs at 720p quality and extracts audio using TwitchDownloaderCLI
- **Transcriber**: Transcribes audio using Deepgram with the nova-3 model, enabling smart_format, paragraphs, and punctuation
- **Sliding Window Generator**: Creates time-based segments (60-second windows with 30-second overlap)
- **Waveform Generator**: Creates audio waveform data for visualization
- **Chat Downloader**: Downloads and processes Twitch chat logs using multi-threading for faster downloads
- **Uploader**: Uploads all raw files to Google Cloud Storage buckets

### Data Flow

1. The pipeline starts with a Twitch VOD URL
2. The downloader extracts the video ID and downloads the VOD
3. Audio is extracted from the video and saved as a WAV file (16kHz mono format)
4. The transcriber processes the audio to generate transcript files
5. The sliding window generator creates segment files for analysis
6. The waveform generator creates audio visualization data
7. The chat downloader retrieves and processes chat logs
8. All files are uploaded to their respective GCS buckets
9. Temporary directories (downloads and data) are cleaned up

### Outputs

The raw pipeline generates the following files in the `output/raw` directory:
- `videos/video_{video_id}.mp4`: The downloaded Twitch VOD
- `audio/audio_{video_id}.wav`: Extracted audio in WAV format
- `transcripts/audio_{video_id}_transcript.json`: Full transcript data from Deepgram
- `transcripts/audio_{video_id}_segments.csv`: Time-segmented transcript data
- `waveforms/audio_{video_id}_waveform.png`: Audio waveform visualization
- `chat/{video_id}_chat.json`: Processed chat log data

## Analysis Pipeline

The analysis pipeline processes the raw data to extract insights and identify highlights.

### Components

- **Audio Sentiment Analysis**: Analyzes transcript segments for sentiment and emotions using Nebius API with Meta-Llama-3.1-8B-Instruct model
- **Audio Highlight Analysis**: Identifies potential highlights based on audio features, speech rate, and emotional intensity
- **Chat Processing**: Processes chat data for sentiment and engagement metrics
- **Chat Sentiment Analysis**: Analyzes chat for emotional reactions using pre-trained ML models
- **Chat Highlight Analysis**: Identifies potential highlights based on chat activity and emotional coherence
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

The analysis pipeline generates the following files in the `output/analysis` directory:
- `audio/{video_id}_sentiment.csv`: Audio sentiment analysis results
- `audio/{video_id}_highlights.csv`: Audio-based highlight analysis
- `audio/{video_id}_*.png`: Various visualization plots
- `chat/{video_id}_highlight_analysis.csv`: Chat-based highlight analysis
- `{video_id}_integrated_analysis.json`: Combined audio and chat analysis

## Running the Pipelines

### Prerequisites

- Python 3.8+
- Required Python packages (see `requirements.txt`)
- Google Cloud Storage account with configured buckets
- Deepgram API key
- Nebius API key for sentiment analysis
- Convex URL and API key for database integration
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
   CONVEX_URL=your_convex_url
   CONVEX_API_KEY=your_convex_api_key
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
- `raw_pipeline/bin/ffmpeg` (Linux - symbolic link to ffmpeg_mac)
- `raw_pipeline/bin/ffmpeg_mac` (macOS)
- `raw_pipeline/bin/ffmpeg.exe` (Windows)

**Note**: All binary files are managed via Git LFS (Large File Storage) for efficient repository management. When cloning the repository, ensure you have Git LFS installed and run `git lfs pull` to download the binary files.

### Model Files

The chat analysis pipeline uses pre-trained machine learning models:
- `analysis_pipeline/chat/models/emotion_classifier_pipe_lr.pkl`: Classifies chat messages by emotion
- `analysis_pipeline/chat/models/highlight_classifier_pipe_lr.pkl`: Identifies potential highlights from chat

## Configuration Options

The system can be configured through environment variables and command-line parameters:

### Environment Variables

- `DEEPGRAM_API_KEY`: API key for Deepgram transcription
- `NEBIUS_API_KEY`: API key for Nebius sentiment analysis
- `CONVEX_URL`: URL for Convex database integration
- `CONVEX_API_KEY`: API key for Convex database integration
- `GCP_SERVICE_ACCOUNT_PATH`: Path to Google Cloud service account key file (optional)
- `BASE_DIR`: Base directory for file operations (defaults to `/tmp` in cloud environments)
- `USE_GCS`: Whether to use Google Cloud Storage (defaults to `true` in cloud environments)

### Directory Structure

The system uses the following directory structure:
- `output/raw`: Raw files from the raw pipeline
- `output/analysis`: Analysis files from the analysis pipeline
- `downloads`: Temporary directory for downloaded files
- `data`: Temporary directory for intermediate data

### Google Cloud Storage Buckets

The system uses the following GCS buckets:
- `klipstream-vods-raw`: For video, audio, and waveform files
- `klipstream-transcripts`: For transcript files
- `klipstream-chatlogs`: For chat log files
- `klipstream-analysis`: For integrated analysis files

## Convex Database Integration

The system integrates with a Convex database to provide real-time status updates and file URLs. This integration allows the frontend to display the current status of the pipeline and to access the generated files once they are available.

### Status Updates

The pipeline updates the `status` field in the Convex database at the following stages:

1. **Queued**: Before download begins
2. **Downloading**: When raw pipeline starts downloading
3. **Fetching chat**: When chat download begins
4. **Transcribing**: When transcription begins
5. **Analyzing**: When analysis pipeline starts
6. **Finding highlights**: When highlight detection begins
7. **Completed**: When pipeline finishes successfully
8. **Failed**: If any step fails

### URL Updates

The pipeline updates the following URL fields in the Convex database:

1. **transcriptUrl**: Updated after the transcript segments file is uploaded
2. **chatUrl**: Updated after the chat file is uploaded
3. **transcriptAnalysisUrl**: Updated after the analysis file is uploaded
4. **audiowaveUrl**: Updated after the waveform.png file is uploaded
5. **transcriptWordUrl**: Updated after the transcript words file is uploaded

### Testing Convex Integration

To test the Convex integration:

```bash
python test_convex_integration.py --video-id YOUR_VIDEO_ID --test-all
```

For more details, see [Convex Integration Documentation](decision_docs/convex_integration_updated.md).

## Testing

For testing purposes, you can use the following Twitch VOD URL:
```
https://www.twitch.tv/videos/2434635255
```

## Deployment Options

The KlipStream Analysis system can be deployed in two ways:

### Cloud Run Deployment (Recommended)

For production use, we recommend deploying as a Google Cloud Run service using the provided script:

```bash
./deploy_cloud_run_simple.sh
```

This script:
1. Reads configuration from `.env.yaml`
2. Builds and deploys a Docker container with:
   - 8 vCPU cores
   - 32GB memory
   - 1-hour timeout (3600 seconds)
   - HTTP trigger
3. Configures the service account with necessary permissions

#### Cloud Run Benefits

- **Longer Execution Time**: Up to 24 hours
- **More Memory**: Up to 32GB RAM
- **More CPU**: Up to 8 vCPU cores
- **Better Scaling**: Handles multiple concurrent requests efficiently
- **Cost Efficiency**: Pay only for the time your service is processing requests

### Cloud Function Deployment (Alternative)

For development or testing, you can deploy as a Google Cloud Function, though this option is less actively used in the current codebase.

#### Cloud Function Limitations

- **Execution Time**: Maximum 60 minutes (3600 seconds)
- **Memory**: Maximum 16GB RAM
- **Disk Space**: Maximum 10GB in `/tmp` directory
- **Cold Starts**: Functions that are not frequently used may experience cold starts
- **Concurrency**: Limited ability to handle multiple concurrent requests

For more detailed deployment instructions, see [decision_docs/DEPLOYMENT.md](decision_docs/DEPLOYMENT.md).

## Documentation

Detailed documentation about implementation decisions, deployment procedures, and other important information can be found in the `decision_docs` directory:

- [Audio Highlight Score Analysis](decision_docs/audio_highlight_score_analysis.md)
- [Cloud Function Deployment](decision_docs/cloud_function_deployment.md)
- [Convex Integration](decision_docs/convex_integration_updated.md)
- [Deepgram Migration](decision_docs/DEEPGRAM_MIGRATION.md)
- [Deployment](decision_docs/DEPLOYMENT.md)
- [Download Scripts](decision_docs/DOWNLOAD_SCRIPTS.md)
- [GCP Authentication](decision_docs/gcp_authentication.md)
- [Nebius Sentiment Implementation](decision_docs/nebius_sentiment_implementation.md)
- [Sentiment Nebius Migration](decision_docs/sentiment_nebius_migration.md)
- [Video Quality](decision_docs/video_quality.md)
