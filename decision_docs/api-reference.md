# KlipStream Analysis API Reference

## 🌐 Base URL
```
Production: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app
```

## 📋 API Endpoints

### Analysis Operations

#### POST /api/v1/analysis
Start a new video analysis job.

**Request Body:**
```json
{
  "url": "string (required) - Twitch VOD URL"
}
```

**Response (200):**
```json
{
  "job_id": "string - Unique job identifier",
  "status": "string - Current job status",
  "message": "string - Human readable message",
  "progress": {
    "percentage": "number - Completion percentage (0-100)",
    "current_stage": "string - Current processing stage",
    "estimated_completion_seconds": "number - ETA in seconds"
  },
  "created_at": "string - ISO timestamp",
  "video_id": "string - Extracted video ID"
}
```

**Error Responses:**
- `400` - Invalid URL format or missing parameters
- `409` - Analysis already in progress for this video
- `429` - Rate limit exceeded
- `500` - Internal server error

---

#### GET /api/v1/analysis/{job_id}/status
Get current status of an analysis job.

**Path Parameters:**
- `job_id` (string, required) - Job identifier from start analysis

**Response (200):**
```json
{
  "job_id": "string",
  "status": "string - queued|downloading|transcribing|analyzing|completed|failed",
  "progress": {
    "percentage": "number",
    "current_stage": "string",
    "message": "string - Detailed status message",
    "estimated_completion_seconds": "number"
  },
  "created_at": "string - ISO timestamp",
  "updated_at": "string - ISO timestamp",
  "video_id": "string"
}
```

**Status Values:**
- `queued` - Job created, waiting to start
- `downloading` - Downloading video and chat data
- `transcribing` - Converting audio to text
- `analyzing` - Performing sentiment analysis
- `completed` - Analysis finished successfully
- `failed` - Analysis failed with error

**Error Responses:**
- `404` - Job not found
- `500` - Internal server error

---

#### GET /api/v1/analysis/{job_id}/results
Get analysis results for completed job.

**Path Parameters:**
- `job_id` (string, required) - Job identifier

**Response (200):**
```json
{
  "job_id": "string",
  "status": "string",
  "results": {
    "video_url": "string - GCS URL to processed video",
    "audio_url": "string - GCS URL to extracted audio",
    "transcript_url": "string - GCS URL to transcript JSON",
    "analysis_url": "string - GCS URL to full analysis",
    "waveform_url": "string - GCS URL to waveform data",
    "highlights": [
      {
        "start_time": "number - Start time in seconds",
        "end_time": "number - End time in seconds",
        "emotion": "string - Primary emotion",
        "score": "number - Confidence score (0-1)",
        "description": "string - Human readable description",
        "clip_url": "string - Direct link to highlight clip"
      }
    ],
    "sentiment_summary": {
      "overall_sentiment": "string - positive|negative|neutral",
      "emotion_breakdown": {
        "excitement": "number - Percentage (0-1)",
        "happiness": "number",
        "funny": "number",
        "anger": "number",
        "sadness": "number"
      },
      "total_duration": "number - Video duration in seconds",
      "processed_segments": "number - Number of analyzed segments"
    },
    "metadata": {
      "video_title": "string",
      "streamer_name": "string",
      "duration": "number",
      "view_count": "number",
      "created_at": "string"
    }
  }
}
```

**Error Responses:**
- `404` - Job not found or not completed
- `500` - Internal server error

---

#### DELETE /api/v1/analysis/{job_id}
Cancel a running analysis job.

**Path Parameters:**
- `job_id` (string, required) - Job identifier

**Response (200):**
```json
{
  "job_id": "string",
  "status": "cancelled",
  "message": "Job cancelled successfully"
}
```

**Error Responses:**
- `404` - Job not found
- `409` - Job already completed or failed
- `500` - Internal server error

### Queue Management

#### GET /api/v1/queue/status
Get current queue status and system health.

**Response (200):**
```json
{
  "queue_length": "number - Jobs in queue",
  "active_jobs": "number - Currently processing",
  "completed_today": "number - Jobs completed today",
  "average_processing_time": "number - Average time in seconds",
  "system_status": "string - healthy|degraded|down",
  "estimated_wait_time": "number - Queue wait time in seconds"
}
```

#### GET /api/v1/queue/jobs
List recent jobs (admin endpoint).

**Query Parameters:**
- `limit` (number, optional) - Number of jobs to return (default: 50)
- `status` (string, optional) - Filter by status
- `since` (string, optional) - ISO timestamp to filter from

**Response (200):**
```json
{
  "jobs": [
    {
      "job_id": "string",
      "status": "string",
      "created_at": "string",
      "updated_at": "string",
      "video_id": "string",
      "progress": {
        "percentage": "number",
        "current_stage": "string"
      }
    }
  ],
  "total": "number",
  "page": "number",
  "limit": "number"
}
```

### System Endpoints

#### GET /health
System health check.

**Response (200):**
```json
{
  "status": "healthy|degraded|unhealthy",
  "version": "string - API version",
  "timestamp": "string - ISO timestamp",
  "uptime": "number - Uptime in seconds",
  "checks": {
    "database": "boolean",
    "storage": "boolean",
    "external_apis": "boolean"
  }
}
```

#### GET /docs
Interactive API documentation (Swagger UI).

#### GET /openapi.json
OpenAPI specification in JSON format.

#### GET /api/v1/monitoring/dashboard
System monitoring dashboard (HTML).

## 🔧 Error Handling

### Standard Error Response Format
```json
{
  "error": {
    "code": "string - Error code",
    "message": "string - Human readable message",
    "details": "object - Additional error details",
    "timestamp": "string - ISO timestamp",
    "request_id": "string - Unique request identifier"
  }
}
```

### Common Error Codes
- `INVALID_URL` - Malformed or unsupported URL
- `VIDEO_NOT_FOUND` - Twitch video not accessible
- `ANALYSIS_IN_PROGRESS` - Duplicate analysis request
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `PROCESSING_FAILED` - Analysis pipeline error
- `STORAGE_ERROR` - File storage issue
- `TIMEOUT` - Request timeout

## 📊 Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| POST /api/v1/analysis | 10 requests | 1 hour |
| GET /api/v1/analysis/{id}/status | 60 requests | 1 minute |
| GET /api/v1/analysis/{id}/results | 100 requests | 1 hour |
| GET /health | 1000 requests | 1 minute |

**Rate Limit Headers:**
- `X-RateLimit-Limit` - Request limit per window
- `X-RateLimit-Remaining` - Requests remaining in window
- `X-RateLimit-Reset` - Window reset time (Unix timestamp)

## 🔐 Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible with rate limiting.

**Future Authentication (Planned):**
- API Key authentication
- JWT token support
- OAuth2 integration

## 📝 Request/Response Examples

### Start Analysis
```bash
curl -X POST "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.twitch.tv/videos/1234567890"}'
```

### Check Status
```bash
curl "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis/job-uuid/status"
```

### Get Results
```bash
curl "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis/job-uuid/results"
```

## 🚀 SDKs and Libraries

### JavaScript/TypeScript
```bash
npm install @klipstream/analysis-client
```

### Python
```bash
pip install klipstream-analysis
```

### Go
```bash
go get github.com/klipstream/analysis-go
```

## 📞 Support

- **Documentation**: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/docs
- **Status Page**: https://status.klipstream.com
- **GitHub Issues**: https://github.com/klipstream/analysis/issues
- **Email**: support@klipstream.com
