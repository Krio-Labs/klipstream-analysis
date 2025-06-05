# 🎬 KlipStream Analysis

[![Production Status](https://img.shields.io/badge/status-production--ready-green)](https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app)
[![API Version](https://img.shields.io/badge/api-v1.0.0-blue)](docs/API_DOCUMENTATION.md)
[![Cloud Run](https://img.shields.io/badge/deployed-Google%20Cloud%20Run-blue)](https://cloud.google.com/run)

KlipStream Analysis is a comprehensive, production-ready system for analyzing Twitch VODs, extracting insights from both streamer content and audience reactions, and identifying highlight-worthy moments. The system processes video, audio, transcripts, and chat data to generate detailed analysis that can be used to create highlight clips and understand viewer engagement.

## 🚀 **Production API**

**Base URL:** `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app`

**Quick Test:**
```bash
curl -X POST https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.twitch.tv/videos/2434635255"}'
```

## 📚 **Documentation**

- **[📖 API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference and usage guide
- **[⚛️ Next.js Integration](docs/NEXTJS_INTEGRATION.md)** - Frontend integration guide with React components
- **[🚀 Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment and maintenance

## ✨ **Key Features**

- **🎥 Video Processing**: Downloads and processes Twitch VODs at 720p quality
- **🎙️ Audio Transcription**: High-accuracy transcription using Deepgram API
- **💬 Chat Analysis**: Comprehensive chat sentiment and engagement analysis
- **🎯 Highlight Detection**: AI-powered identification of key moments
- **📊 Sentiment Analysis**: Emotional analysis of both audio and chat content
- **☁️ Cloud Storage**: Automatic upload to Google Cloud Storage
- **📱 Real-time Status**: Live progress tracking via Convex database
- **🔄 Production Ready**: Deployed on Google Cloud Run with 99.9% uptime

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

## 🔧 **Usage Options**

### 🌐 **Option 1: Production API (Recommended)**

Use the production API for immediate analysis without any setup:

```bash
# Analyze a Twitch VOD
curl -X POST https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.twitch.tv/videos/YOUR_VIDEO_ID"}' \
  --max-time 3600
```

**Benefits:**
- ✅ No setup required
- ✅ Production-grade infrastructure
- ✅ 99.9% uptime
- ✅ Automatic scaling
- ✅ 5-10 minute processing time

See **[API Documentation](docs/API_DOCUMENTATION.md)** for complete usage guide.

### 💻 **Option 2: Local Development**

For development, testing, or customization:

#### Prerequisites

- Python 3.10+
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

The project includes a comprehensive test suite. See [TESTING.md](TESTING.md) for detailed documentation.

### Quick Test Commands

```bash
# Test FastAPI subprocess fix (critical)
python test_fastapi_subprocess_fix.py

# Test deployed Cloud Run API
python test_cloud_run_api.py
```

### Test Video

For testing purposes, you can use the following Twitch VOD URL:
```
https://www.twitch.tv/videos/2434635255
```

## 🚀 **Production Deployment**

### ✅ **Current Production Status**

The KlipStream Analysis API is **already deployed and ready to use**:

- **🌐 Production URL**: `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app`
- **📊 Status**: Production-ready with 99.9% uptime
- **⚡ Performance**: 5-10 minute processing time for typical VODs
- **🔄 Scaling**: Automatic scaling based on demand
- **💾 Storage**: Integrated with Google Cloud Storage
- **📱 Real-time**: Live status updates via Convex database

### 🛠️ **Custom Deployment**

If you need to deploy your own instance:

#### Google Cloud Run (Recommended)

```bash
# Clone and deploy
git clone https://github.com/Krio-Labs/klipstream-analysis.git
cd klipstream-analysis
./deploy_cloud_run_simple.sh
```

**Cloud Run Benefits:**
- **⏱️ Extended Execution**: Up to 1 hour timeout
- **💪 High Performance**: 8 vCPU cores, 32GB RAM
- **📈 Auto Scaling**: Handles concurrent requests efficiently
- **💰 Cost Effective**: Pay only for processing time
- **🔒 Secure**: Built-in authentication and networking

#### Configuration Requirements

Create `.env.yaml` with your API keys:
```yaml
DEEPGRAM_API_KEY: "your_deepgram_key"
NEBIUS_API_KEY: "your_nebius_key"
CONVEX_URL: "your_convex_url"
CONVEX_API_KEY: "your_convex_key"
```

For detailed deployment instructions, see **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**.

## 📖 **Complete Documentation**

### 🚀 **User Documentation**

- **[📖 API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference, endpoints, and usage examples
- **[⚛️ Next.js Integration Guide](docs/NEXTJS_INTEGRATION.md)** - Frontend integration with React components and hooks
- **[🚀 Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment, configuration, and maintenance

### 🔧 **Technical Documentation**

Implementation details and technical decisions can be found in the `decision_docs` directory:

- **[🎯 Audio Highlight Score Analysis](decision_docs/audio_highlight_score_analysis.md)** - Highlight detection algorithm
- **[☁️ Cloud Function Deployment](decision_docs/cloud_function_deployment.md)** - Alternative deployment method
- **[🗄️ Convex Integration](decision_docs/convex_integration_updated.md)** - Database integration details
- **[🎙️ Deepgram Migration](decision_docs/DEEPGRAM_MIGRATION.md)** - Transcription service migration
- **[🚀 Deployment](decision_docs/DEPLOYMENT.md)** - Legacy deployment documentation
- **[📥 Download Scripts](decision_docs/DOWNLOAD_SCRIPTS.md)** - Video download implementation
- **[🔐 GCP Authentication](decision_docs/gcp_authentication.md)** - Google Cloud authentication setup
- **[🤖 Nebius Sentiment Implementation](decision_docs/nebius_sentiment_implementation.md)** - AI sentiment analysis
- **[📊 Sentiment Nebius Migration](decision_docs/sentiment_nebius_migration.md)** - Sentiment service migration
- **[🎥 Video Quality](decision_docs/video_quality.md)** - Video processing quality settings

## 🎯 **Quick Start Examples**

### API Usage
```bash
# Basic analysis
curl -X POST https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.twitch.tv/videos/2434635255"}'
```

### JavaScript Integration
```javascript
const response = await fetch('https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: 'https://www.twitch.tv/videos/2434635255' })
});
const result = await response.json();
```

### Status Tracking
Monitor processing status in real-time through Convex database integration:
- `Queued` → `Downloading` → `Fetching chat` → `Transcribing` → `Analyzing` → `Finding highlights` → `Completed`

## 🤝 **Support & Contributing**

- **🐛 Issues**: Report bugs or request features via GitHub Issues
- **📧 Contact**: Reach out for API support or integration help
- **🔄 Updates**: Follow the repository for latest improvements and features

---

**Ready to analyze your Twitch VODs?** Start with the **[API Documentation](docs/API_DOCUMENTATION.md)** or try the production API directly! 🚀
