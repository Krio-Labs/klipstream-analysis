# 🚀 KlipStream Analysis API Documentation

## Overview

The KlipStream Analysis API is a comprehensive video analysis service that processes Twitch VODs to generate transcripts, sentiment analysis, highlights, and detailed analytics. The service is deployed on Google Cloud Run and provides a RESTful API for integration with frontend applications.

## Base Information

- **Base URL:** `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app`
- **Protocol:** HTTPS
- **Content-Type:** `application/json`
- **Timeout:** 3600 seconds (1 hour)
- **Rate Limiting:** No explicit limits (reasonable use expected)

## Authentication

Currently, the API is publicly accessible and does not require authentication. This may change in future versions.

## API Endpoints

### POST / - Analyze Video

Initiates analysis of a Twitch VOD.

**Request:**
```http
POST https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app
Content-Type: application/json

{
  "url": "https://www.twitch.tv/videos/VIDEO_ID"
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | Full Twitch VOD URL |

**URL Format:**
- Valid: `https://www.twitch.tv/videos/2434635255`
- Invalid: `https://twitch.tv/videos/123` (missing www)
- Invalid: `https://www.twitch.tv/user123` (not a VOD)

**Response (Success - 200):**
```json
{
  "status": "success",
  "message": "Analysis completed successfully",
  "video_id": "2434635255",
  "execution_time": {
    "raw_pipeline": 241.87,
    "analysis_pipeline": 127.98,
    "total": 370.08
  },
  "files": {
    "audio_file": "gs://klipstream-vods-raw/audio_2434635255.wav",
    "video_file": "gs://klipstream-vods-raw/video_2434635255.mp4",
    "waveform_file": "gs://klipstream-vods-raw/waveform_2434635255.png",
    "transcript_files": {
      "json": "gs://klipstream-transcripts/audio_2434635255_transcript.json",
      "words": "gs://klipstream-transcripts/audio_2434635255_words.csv",
      "paragraphs": "gs://klipstream-transcripts/audio_2434635255_paragraphs.csv"
    },
    "chat_files": {
      "raw": "gs://klipstream-chatlogs/chat_2434635255.json",
      "processed": "gs://klipstream-chatlogs/chat_2434635255_processed.csv"
    },
    "analysis_files": {
      "integrated": "gs://klipstream-analysis/integrated_analysis_2434635255.json",
      "sentiment": "gs://klipstream-analysis/audio_2434635255_sentiment.csv",
      "highlights": "gs://klipstream-analysis/highlights_2434635255.json"
    }
  }
}
```

**Response (Error - 400):**
```json
{
  "error": "Invalid Twitch URL format",
  "status": "failed"
}
```

**Response (Error - 500):**
```json
{
  "error": "Internal server error during processing",
  "status": "failed"
}
```

## Processing Pipeline

The analysis consists of two main pipelines:

### 1. Raw Pipeline (2-4 minutes)
- **Video Download:** Downloads video and audio from Twitch
- **Chat Download:** Extracts chat messages and metadata
- **Transcription:** Generates transcript using Deepgram API
- **Waveform Generation:** Creates audio waveform visualization

### 2. Analysis Pipeline (1-3 minutes)
- **Sentiment Analysis:** Analyzes emotional content in audio and chat
- **Highlight Detection:** Identifies key moments and emotional peaks
- **Integration:** Combines all analysis into comprehensive report

## Status System

The API integrates with Convex database for real-time status tracking:

| Status | Description | Typical Duration |
|--------|-------------|------------------|
| `Queued` | Video queued for processing | < 1 second |
| `Downloading` | Video download in progress | 2-3 minutes |
| `Fetching chat` | Chat data download | 10-30 seconds |
| `Transcribing` | Audio transcription | 15-30 seconds |
| `Analyzing` | Sentiment analysis | 1-2 minutes |
| `Finding highlights` | Highlight detection | 30-60 seconds |
| `Completed` | Pipeline finished successfully | Final state |
| `Failed` | Error occurred at any stage | Error state |

## File Storage

All generated files are stored in Google Cloud Storage:

### Storage Buckets
- **Raw Files:** `klipstream-vods-raw/`
  - Video files (MP4)
  - Audio files (WAV)
  - Waveform images (PNG)

- **Transcripts:** `klipstream-transcripts/`
  - Full transcript (JSON)
  - Word-level timestamps (CSV)
  - Paragraph segments (CSV)

- **Chat Data:** `klipstream-chatlogs/`
  - Raw chat messages (JSON)
  - Processed chat data (CSV)

- **Analysis Results:** `klipstream-analysis/`
  - Integrated analysis (JSON)
  - Sentiment analysis (CSV)
  - Highlights data (JSON)

### File Naming Convention
All files follow the pattern: `{type}_{video_id}.{extension}`
- Example: `audio_2434635255.wav`
- Example: `integrated_analysis_2434635255.json`

## Error Handling

### Common Error Scenarios

1. **Invalid URL Format (400)**
   - Missing `www` in domain
   - Not a VOD URL (e.g., channel URL)
   - Invalid video ID format

2. **Video Not Found (400)**
   - Private or deleted video
   - Invalid video ID
   - Geo-restricted content

3. **Processing Timeout (504)**
   - Very large files (>2 hours)
   - Network issues during download
   - API service timeouts

4. **Server Error (500)**
   - Transcription service failures
   - Storage upload failures
   - Internal processing errors

### Error Response Format
```json
{
  "error": "Detailed error message",
  "status": "failed",
  "video_id": "2434635255",
  "stage": "transcribing"
}
```

## Performance Characteristics

### Typical Processing Times
- **1-hour VOD:** 5-7 minutes total
- **2-hour VOD:** 8-12 minutes total
- **30-minute VOD:** 3-5 minutes total

### Resource Usage
- **CPU:** 8 cores
- **Memory:** 32GB
- **Storage:** Temporary (cleaned after processing)
- **Network:** High bandwidth for video download

### Scalability
- **Concurrent Requests:** Limited by Cloud Run instances
- **Queue System:** Automatic scaling based on demand
- **Rate Limiting:** Fair use policy (no hard limits)

## Integration Examples

### cURL
```bash
curl -X POST https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.twitch.tv/videos/2434635255"}' \
  --max-time 3600
```

### JavaScript/TypeScript
```javascript
const response = await fetch('https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: twitchUrl }),
  signal: AbortSignal.timeout(3600000) // 1 hour
});
```

### Python
```python
import requests

response = requests.post(
    'https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app',
    json={'url': twitch_url},
    timeout=3600
)
```

## Monitoring and Logging

### Health Monitoring
- **Uptime:** 99.9% target availability
- **Response Time:** < 10 seconds for API response
- **Processing Time:** Variable based on content length

### Logging
- **Cloud Run Logs:** Detailed processing logs
- **Error Tracking:** Comprehensive error reporting
- **Performance Metrics:** Processing time tracking

## Support and Troubleshooting

### Common Issues

1. **Timeout Errors**
   - Increase client timeout to 1 hour
   - Check video length (very long videos take more time)
   - Retry after a few minutes

2. **Invalid URL Errors**
   - Ensure URL includes `https://www.twitch.tv/videos/`
   - Verify video ID is numeric
   - Check if video is public and accessible

3. **Processing Failures**
   - Check video availability on Twitch
   - Verify video has audio content
   - Contact support for persistent issues

### Getting Help
- **Documentation:** This document and README.md
- **Logs:** Available in Google Cloud Console
- **Support:** Contact development team for issues

## Changelog

### Version 1.0.0 (Current)
- Initial release with full pipeline
- Deepgram transcription integration
- Comprehensive error handling
- Real-time status updates
- File storage in Google Cloud Storage

## Future Enhancements

### Planned Features
- Authentication and API keys
- Webhook notifications for completion
- Batch processing capabilities
- Custom analysis parameters
- Additional output formats

### Performance Improvements
- Parallel processing optimization
- Caching for repeated requests
- Reduced processing times
- Enhanced error recovery
