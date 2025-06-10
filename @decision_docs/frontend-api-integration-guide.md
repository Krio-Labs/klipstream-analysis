# KlipStream Analysis API - Frontend Integration Guide

**Version:** 2.0.0  
**Last Updated:** December 10, 2024  
**API Base URL:** `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app`

## 📋 Table of Contents

1. [Overview](#overview)
2. [API Endpoints](#api-endpoints)
3. [Video Analysis Workflow](#video-analysis-workflow)
4. [TypeScript Interfaces](#typescript-interfaces)
5. [React Integration Examples](#react-integration-examples)
6. [Error Handling](#error-handling)
7. [Performance & Limitations](#performance--limitations)
8. [Environment Configuration](#environment-configuration)

## 🎯 Overview

The KlipStream Analysis API provides asynchronous video processing capabilities with real-time progress tracking. It processes Twitch videos to generate transcripts, sentiment analysis, and highlight detection.

### Key Features
- ✅ Asynchronous video processing
- ✅ Real-time progress tracking via Server-Sent Events (SSE)
- ✅ Multiple transcription methods (Deepgram, GPU-accelerated)
- ✅ Comprehensive error handling with fallback mechanisms
- ✅ Automatic Convex database integration
- ✅ Cost optimization and intelligent method selection

### Current Deployment
- **Mode:** CPU-only (Deepgram transcription)
- **Resources:** 4 CPU cores, 16GB memory
- **Timeout:** 3600 seconds (1 hour)
- **Concurrency:** Up to 5 concurrent requests
- **Max Instances:** 3

## 🌐 API Endpoints

### Base Information

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-10T10:30:00Z",
  "version": "2.0.0"
}
```

#### API Information
```http
GET /
```

**Response:**
```json
{
  "name": "KlipStream Analysis API",
  "version": "2.0.0",
  "status": "operational",
  "features": [
    "Asynchronous video processing",
    "Real-time progress tracking",
    "Comprehensive error handling",
    "Server-sent events support"
  ],
  "endpoints": {
    "analysis": "/api/v1/analysis",
    "status": "/api/v1/analysis/{job_id}/status",
    "stream": "/api/v1/analysis/{job_id}/stream",
    "docs": "/docs"
  }
}
```

#### Available Transcription Methods
```http
GET /api/v1/transcription/methods
```

**Response:**
```json
{
  "status": "success",
  "message": "Available transcription methods",
  "methods": {
    "auto": {
      "name": "Automatic Selection",
      "description": "Automatically selects the best method based on file duration and cost optimization",
      "cost_per_hour": "Variable (optimized)",
      "gpu_required": false,
      "recommended_for": "Most use cases - optimal cost/performance balance"
    },
    "deepgram": {
      "name": "Deepgram API",
      "description": "Cloud-based transcription using Deepgram Nova-3 model",
      "cost_per_hour": "$0.27 (API calls)",
      "gpu_required": false,
      "recommended_for": "Long files (> 4 hours) or when GPU is not available"
    }
  },
  "configuration": {
    "default_method": "auto",
    "cost_optimization_enabled": true,
    "gpu_fallback_enabled": true,
    "estimated_processing_speed": {
      "deepgram": "5-10x real-time"
    }
  }
}
```

## 🎬 Video Analysis Workflow

### Step 1: Start Analysis

```http
POST /api/v1/analysis
Content-Type: application/json
```

**Request Body:**
```json
{
  "url": "https://www.twitch.tv/videos/2345678901",
  "user_id": "jh7en1zt0460hfzf2qa470p1k17epm40",
  "team_id": "js7bj9zgdkyj9ykvr4m6jarxkh7ep9fa",
  "options": {
    "transcription_method": "auto",
    "enable_highlights": true,
    "enable_sentiment": true,
    "chunk_duration": 10
  }
}
```

**Field Descriptions:**
- `url` (required): Twitch video URL
- `user_id` (optional): Convex user ID for database updates
- `team_id` (optional): Convex team ID for database updates
- `options.transcription_method`: `"auto"`, `"deepgram"`, `"parakeet"`, or `"hybrid"`
- `options.enable_highlights`: Enable highlight detection (default: true)
- `options.enable_sentiment`: Enable sentiment analysis (default: true)
- `options.chunk_duration`: Processing chunk size in minutes (default: 10)

**Response:**
```json
{
  "status": "success",
  "job_id": "analysis_20241210_103045_abc123",
  "message": "Analysis started successfully",
  "estimated_duration": "15-30 minutes",
  "stream_url": "/api/v1/analysis/analysis_20241210_103045_abc123/stream"
}
```

### Step 2: Monitor Progress (Polling Method)

```http
GET /api/v1/analysis/{job_id}/status
```

**Response:**
```json
{
  "job_id": "analysis_20241210_103045_abc123",
  "status": "processing",
  "progress": 45,
  "current_stage": "transcription",
  "stages": {
    "download": "completed",
    "audio_conversion": "completed",
    "transcription": "processing",
    "sentiment_analysis": "pending",
    "highlight_detection": "pending",
    "upload": "pending"
  },
  "estimated_completion": "2024-12-10T10:45:00Z",
  "error": null,
  "results": {
    "video_url": "https://storage.googleapis.com/...",
    "audio_url": "https://storage.googleapis.com/...",
    "transcript_url": "https://storage.googleapis.com/..."
  }
}
```

**Status Values:**
- `queued`: Job is waiting to be processed
- `processing`: Job is currently being processed
- `completed`: Job finished successfully
- `failed`: Job failed with error

**Stage Values:**
- `pending`: Stage not started
- `processing`: Stage currently running
- `completed`: Stage finished successfully
- `failed`: Stage failed

### Step 3: Real-time Updates (Server-Sent Events)

```http
GET /api/v1/analysis/{job_id}/stream
Accept: text/event-stream
```

**Event Stream Examples:**
```
data: {"type": "progress", "progress": 25, "stage": "transcription", "message": "Processing audio chunks..."}

data: {"type": "stage_complete", "stage": "transcription", "duration": "12.5s"}

data: {"type": "completion", "status": "completed", "results_url": "https://storage.googleapis.com/..."}

data: {"type": "error", "error": "Transcription failed", "stage": "transcription"}
```

## 📝 TypeScript Interfaces

```typescript
// Request Interfaces
interface AnalysisRequest {
  url: string;
  user_id?: string;
  team_id?: string;
  options?: {
    transcription_method?: 'auto' | 'deepgram' | 'parakeet' | 'hybrid';
    enable_highlights?: boolean;
    enable_sentiment?: boolean;
    chunk_duration?: number;
  };
}

// Response Interfaces
interface AnalysisResponse {
  status: 'success' | 'error';
  job_id?: string;
  message: string;
  estimated_duration?: string;
  stream_url?: string;
  error?: string;
}

interface StatusResponse {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number; // 0-100
  current_stage: string;
  stages: Record<string, 'pending' | 'processing' | 'completed' | 'failed'>;
  estimated_completion?: string;
  error?: string;
  results?: {
    video_url?: string;
    audio_url?: string;
    waveform_url?: string;
    transcript_url?: string;
    transcriptWords_url?: string;
    chat_url?: string;
    analysis_url?: string;
    highlights?: Array<{
      start_time: number;
      end_time: number;
      type: 'excitement' | 'funny' | 'happiness' | 'anger' | 'sadness';
      score: number;
      description?: string;
    }>;
  };
}

// SSE Event Interface
interface SSEEvent {
  type: 'progress' | 'stage_complete' | 'completion' | 'error';
  progress?: number;
  stage?: string;
  message?: string;
  status?: string;
  results_url?: string;
  error?: string;
  duration?: string;
}

// Transcription Methods Interface
interface TranscriptionMethod {
  name: string;
  description: string;
  cost_per_hour: string;
  gpu_required: boolean;
  recommended_for: string;
}

interface TranscriptionMethodsResponse {
  status: 'success' | 'error';
  message: string;
  methods: Record<string, TranscriptionMethod>;
  configuration: {
    default_method: string;
    cost_optimization_enabled: boolean;
    gpu_fallback_enabled: boolean;
    estimated_processing_speed: Record<string, string>;
  };
}
```

## 🔧 React Integration Examples

### Custom Hook for Analysis

```typescript
import { useState, useEffect, useCallback } from 'react';

interface UseAnalysisResult {
  startAnalysis: (request: AnalysisRequest) => Promise<void>;
  status: StatusResponse | null;
  isLoading: boolean;
  error: string | null;
  progress: number;
  jobId: string | null;
}

export function useAnalysis(): UseAnalysisResult {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const API_BASE = 'https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app';

  const startAnalysis = useCallback(async (request: AnalysisRequest) => {
    setIsLoading(true);
    setError(null);
    setStatus(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result: AnalysisResponse = await response.json();

      if (result.status === 'success' && result.job_id) {
        setJobId(result.job_id);
        // Start SSE connection for real-time updates
        connectToStream(result.job_id);
      } else {
        setError(result.error || 'Analysis failed to start');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const connectToStream = useCallback((jobId: string) => {
    const eventSource = new EventSource(`${API_BASE}/api/v1/analysis/${jobId}/stream`);

    eventSource.onmessage = (event) => {
      try {
        const data: SSEEvent = JSON.parse(event.data);

        switch (data.type) {
          case 'progress':
            setStatus(prev => prev ? {
              ...prev,
              progress: data.progress || 0,
              current_stage: data.stage || prev.current_stage,
              stages: {
                ...prev.stages,
                [data.stage || '']: 'processing'
              }
            } : null);
            break;

          case 'stage_complete':
            setStatus(prev => prev ? {
              ...prev,
              stages: {
                ...prev.stages,
                [data.stage || '']: 'completed'
              }
            } : null);
            break;

          case 'completion':
            eventSource.close();
            // Fetch final status with results
            fetchFinalStatus(jobId);
            break;

          case 'error':
            eventSource.close();
            setError(data.error || 'Analysis failed');
            break;
        }
      } catch (err) {
        console.error('Failed to parse SSE data:', err);
      }
    };

    eventSource.onerror = (event) => {
      console.error('SSE connection error:', event);
      eventSource.close();
      // Fallback to polling
      startPolling(jobId);
    };

    return eventSource;
  }, []);

  const fetchFinalStatus = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/analysis/${jobId}/status`);
      if (response.ok) {
        const statusData: StatusResponse = await response.json();
        setStatus(statusData);
      }
    } catch (err) {
      console.error('Failed to fetch final status:', err);
    }
  }, []);

  const startPolling = useCallback((jobId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/v1/analysis/${jobId}/status`);
        if (response.ok) {
          const statusData: StatusResponse = await response.json();
          setStatus(statusData);

          if (statusData.status === 'completed' || statusData.status === 'failed') {
            clearInterval(pollInterval);
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 5000); // Poll every 5 seconds

    return pollInterval;
  }, []);

  return {
    startAnalysis,
    status,
    isLoading,
    error,
    progress: status?.progress || 0,
    jobId
  };
}
```

### React Component Example

```tsx
import React, { useState } from 'react';
import { useAnalysis } from './hooks/useAnalysis';

interface VideoAnalysisProps {
  onComplete?: (results: any) => void;
}

export const VideoAnalysis: React.FC<VideoAnalysisProps> = ({ onComplete }) => {
  const [videoUrl, setVideoUrl] = useState('');
  const { startAnalysis, status, isLoading, error, progress } = useAnalysis();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!videoUrl.trim()) return;

    await startAnalysis({
      url: videoUrl,
      user_id: 'jh7en1zt0460hfzf2qa470p1k17epm40', // Replace with actual user ID
      team_id: 'js7bj9zgdkyj9ykvr4m6jarxkh7ep9fa', // Replace with actual team ID
      options: {
        transcription_method: 'auto',
        enable_highlights: true,
        enable_sentiment: true,
        chunk_duration: 10
      }
    });
  };

  React.useEffect(() => {
    if (status?.status === 'completed' && onComplete) {
      onComplete(status.results);
    }
  }, [status, onComplete]);

  const getStageColor = (stageStatus: string) => {
    switch (stageStatus) {
      case 'completed': return 'text-green-600';
      case 'processing': return 'text-blue-600';
      case 'failed': return 'text-red-600';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Video Analysis</h2>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="mb-8">
        <div className="flex gap-4">
          <input
            type="url"
            value={videoUrl}
            onChange={(e) => setVideoUrl(e.target.value)}
            placeholder="https://www.twitch.tv/videos/..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            disabled={isLoading || status?.status === 'processing'}
          />
          <button
            type="submit"
            disabled={isLoading || status?.status === 'processing' || !videoUrl.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? 'Starting...' : 'Analyze Video'}
          </button>
        </div>
      </form>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Progress Display */}
      {status && (
        <div className="space-y-6">
          {/* Overall Progress */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">Overall Progress</span>
              <span className="text-sm text-gray-600">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Current Stage */}
          <div>
            <p className="text-sm text-gray-600 mb-2">
              Current Stage: <span className="font-medium">{status.current_stage}</span>
            </p>
          </div>

          {/* Stage Details */}
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(status.stages).map(([stage, stageStatus]) => (
              <div key={stage} className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  stageStatus === 'completed' ? 'bg-green-500' :
                  stageStatus === 'processing' ? 'bg-blue-500 animate-pulse' :
                  stageStatus === 'failed' ? 'bg-red-500' : 'bg-gray-300'
                }`} />
                <span className={`text-sm capitalize ${getStageColor(stageStatus)}`}>
                  {stage.replace('_', ' ')}
                </span>
              </div>
            ))}
          </div>

          {/* Estimated Completion */}
          {status.estimated_completion && (
            <p className="text-sm text-gray-600">
              Estimated completion: {new Date(status.estimated_completion).toLocaleTimeString()}
            </p>
          )}

          {/* Results */}
          {status.status === 'completed' && status.results && (
            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="font-medium text-green-800 mb-3">Analysis Complete!</h3>
              <div className="space-y-2">
                {status.results.transcript_url && (
                  <a
                    href={status.results.transcript_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-blue-600 hover:underline"
                  >
                    📄 View Transcript
                  </a>
                )}
                {status.results.analysis_url && (
                  <a
                    href={status.results.analysis_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-blue-600 hover:underline"
                  >
                    📊 View Analysis Results
                  </a>
                )}
                {status.results.highlights && status.results.highlights.length > 0 && (
                  <div>
                    <p className="font-medium">Highlights Found: {status.results.highlights.length}</p>
                    <div className="mt-2 space-y-1">
                      {status.results.highlights.slice(0, 3).map((highlight, index) => (
                        <div key={index} className="text-sm">
                          <span className="font-medium capitalize">{highlight.type}</span> at {Math.floor(highlight.start_time / 60)}:{(highlight.start_time % 60).toFixed(0).padStart(2, '0')}
                          <span className="text-gray-600"> (Score: {highlight.score.toFixed(2)})</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
```

## ⚠️ Error Handling

### Common Error Response Format

```json
{
  "status": "error",
  "error": "Invalid video URL format",
  "code": "INVALID_URL",
  "details": "URL must be a valid Twitch video URL",
  "timestamp": "2024-12-10T10:30:00Z"
}
```

### Error Codes Reference

| Code | Description | User Action |
|------|-------------|-------------|
| `INVALID_URL` | Invalid video URL format | Check URL format |
| `VIDEO_NOT_FOUND` | Video not accessible or doesn't exist | Verify video exists and is public |
| `VIDEO_TOO_LONG` | Video exceeds maximum duration | Use shorter video |
| `PROCESSING_FAILED` | Analysis pipeline failed | Retry or contact support |
| `QUOTA_EXCEEDED` | API rate limit exceeded | Wait and retry |
| `INTERNAL_ERROR` | Server error | Retry or contact support |
| `TIMEOUT` | Processing timeout | Video too long or complex |
| `INSUFFICIENT_RESOURCES` | Server overloaded | Retry later |

### Error Handling Best Practices

```typescript
const handleApiError = (error: any): string => {
  // Network errors
  if (error.name === 'TypeError' && error.message.includes('fetch')) {
    return 'Network connection failed. Please check your internet connection.';
  }

  // HTTP errors
  if (error.status) {
    switch (error.status) {
      case 400:
        return 'Invalid request. Please check your video URL.';
      case 404:
        return 'Video not found. Please verify the URL is correct.';
      case 429:
        return 'Too many requests. Please wait a moment and try again.';
      case 500:
        return 'Server error. Please try again later.';
      case 503:
        return 'Service temporarily unavailable. Please try again later.';
      default:
        return `Request failed with status ${error.status}`;
    }
  }

  // API error responses
  if (error.code) {
    switch (error.code) {
      case 'INVALID_URL':
        return 'Please enter a valid Twitch video URL.';
      case 'VIDEO_NOT_FOUND':
        return 'Video not found. Please check if the video exists and is public.';
      case 'VIDEO_TOO_LONG':
        return 'Video is too long for processing. Maximum duration is 4 hours.';
      case 'QUOTA_EXCEEDED':
        return 'Rate limit exceeded. Please wait a few minutes before trying again.';
      default:
        return error.error || 'An unexpected error occurred.';
    }
  }

  return error.message || 'An unexpected error occurred.';
};
```

### Retry Logic Example

```typescript
const retryWithBackoff = async (
  fn: () => Promise<any>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<any> => {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      // Don't retry on client errors (4xx)
      if (error.status >= 400 && error.status < 500) {
        throw error;
      }

      if (attempt === maxRetries) {
        throw error;
      }

      // Exponential backoff
      const delay = baseDelay * Math.pow(2, attempt - 1);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
};
```

## 🚀 Performance & Limitations

### Processing Times

| Video Duration | Estimated Processing Time | Transcription Method |
|----------------|--------------------------|---------------------|
| 30 minutes | 3-6 minutes | Deepgram |
| 1 hour | 6-12 minutes | Deepgram |
| 2 hours | 12-24 minutes | Deepgram |
| 4 hours | 24-48 minutes | Deepgram |

### Resource Limits

- **Maximum Video Duration:** 4 hours
- **Maximum File Size:** 2GB
- **Concurrent Requests:** 5 per instance
- **Request Timeout:** 3600 seconds (1 hour)
- **Rate Limiting:** 100 requests per minute per IP

### Optimization Tips

1. **Use Auto Method:** Let the API choose the optimal transcription method
2. **Enable Cost Optimization:** Set `cost_optimization: true` in options
3. **Chunk Processing:** Use smaller chunk sizes (5-10 minutes) for better progress tracking
4. **SSE over Polling:** Use Server-Sent Events for real-time updates instead of polling
5. **Error Recovery:** Implement retry logic with exponential backoff

### Browser Compatibility

- **Server-Sent Events:** Supported in all modern browsers
- **Fetch API:** Supported in all modern browsers
- **EventSource:** Supported in all modern browsers
- **IE Support:** Not supported (use polyfills if needed)

## 🔧 Environment Configuration

### Required Environment Variables (API Side)

The API requires these environment variables to be configured:

```bash
# Convex Database
CONVEX_URL=https://your-convex-deployment.convex.cloud

# Transcription Services
DEEPGRAM_API_KEY=your_deepgram_api_key

# Sentiment Analysis
NEBIUS_API_KEY=your_nebius_api_key

# Google Cloud Storage
GCS_PROJECT_ID=klipstream
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# API Configuration
FASTAPI_MODE=true
CLOUD_ENVIRONMENT=true
ENABLE_GPU_TRANSCRIPTION=false
TRANSCRIPTION_METHOD=deepgram
```

### Frontend Environment Variables

```bash
# API Configuration
NEXT_PUBLIC_KLIPSTREAM_API_URL=https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app
NEXT_PUBLIC_API_VERSION=2.0.0

# Convex (for direct database access if needed)
NEXT_PUBLIC_CONVEX_URL=https://your-convex-deployment.convex.cloud

# Feature Flags
NEXT_PUBLIC_ENABLE_REAL_TIME_UPDATES=true
NEXT_PUBLIC_ENABLE_PROGRESS_TRACKING=true
NEXT_PUBLIC_MAX_VIDEO_DURATION_HOURS=4
```

### API Client Configuration

```typescript
// api/config.ts
export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_KLIPSTREAM_API_URL || 'https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app',
  version: process.env.NEXT_PUBLIC_API_VERSION || '2.0.0',
  timeout: 30000, // 30 seconds for API calls
  retryAttempts: 3,
  retryDelay: 1000,
  enableRealTimeUpdates: process.env.NEXT_PUBLIC_ENABLE_REAL_TIME_UPDATES === 'true',
  maxVideoDurationHours: parseInt(process.env.NEXT_PUBLIC_MAX_VIDEO_DURATION_HOURS || '4'),
};

// api/client.ts
export class KlipStreamApiClient {
  private baseUrl: string;

  constructor(config = API_CONFIG) {
    this.baseUrl = config.baseUrl;
  }

  async startAnalysis(request: AnalysisRequest): Promise<AnalysisResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/analysis`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': `KlipStream-Frontend/${API_CONFIG.version}`
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getStatus(jobId: string): Promise<StatusResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/analysis/${jobId}/status`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  createEventSource(jobId: string): EventSource {
    return new EventSource(`${this.baseUrl}/api/v1/analysis/${jobId}/stream`);
  }
}
```

## 📚 Additional Resources

### API Documentation
- **Interactive Docs:** `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/docs`
- **OpenAPI Spec:** `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/openapi.json`

### Testing Endpoints
- **Health Check:** `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/health`
- **API Info:** `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/`

### Support
- **GitHub Repository:** [klipstream-analysis](https://github.com/Krio-Labs/klipstream-analysis)
- **Issues:** Report bugs and feature requests via GitHub Issues
- **Documentation:** This guide and inline API documentation

---

**Last Updated:** December 10, 2024
**API Version:** 2.0.0
**Document Version:** 1.0.0
