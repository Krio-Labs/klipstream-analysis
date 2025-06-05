# KlipStream Analysis API - Next.js Integration Guide

## 📋 Overview

This guide provides comprehensive documentation for integrating the Next.js frontend application with the KlipStream Analysis API deployed on Google Cloud Run.

**Production API Endpoint**: `https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app`

## 🏗️ Architecture Overview

```
┌─────────────────┐    HTTP/REST     ┌──────────────────────┐
│   Next.js App   │ ────────────────► │  Cloud Run FastAPI  │
│  (Frontend)     │                  │   (Backend API)      │
└─────────────────┘                  └──────────────────────┘
                                               │
                                               ▼
                                     ┌──────────────────────┐
                                     │  Analysis Pipeline   │
                                     │  - Video Download    │
                                     │  - Transcription     │
                                     │  - Sentiment Analysis│
                                     │  - Highlight Detection│
                                     └──────────────────────┘
                                               │
                                               ▼
                                     ┌──────────────────────┐
                                     │  Google Cloud Storage│
                                     │  - Raw Files         │
                                     │  - Analysis Results  │
                                     │  - Generated Assets  │
                                     └──────────────────────┘
```

## 🚀 Quick Start

### 1. Environment Variables

Add these environment variables to your Next.js application:

```env
# .env.local
NEXT_PUBLIC_KLIPSTREAM_API_URL=https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app
NEXT_PUBLIC_KLIPSTREAM_API_VERSION=v1
```

### 2. API Client Setup

Create an API client utility:

```typescript
// lib/klipstream-api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_KLIPSTREAM_API_URL;
const API_VERSION = process.env.NEXT_PUBLIC_KLIPSTREAM_API_VERSION || 'v1';

export class KlipStreamAPI {
  private baseUrl: string;

  constructor() {
    this.baseUrl = `${API_BASE_URL}/api/${API_VERSION}`;
  }

  async startAnalysis(twitchUrl: string): Promise<AnalysisResponse> {
    const response = await fetch(`${this.baseUrl}/analysis`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: twitchUrl }),
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getAnalysisStatus(jobId: string): Promise<StatusResponse> {
    const response = await fetch(`${this.baseUrl}/analysis/${jobId}/status`);
    
    if (!response.ok) {
      throw new Error(`Status check failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getAnalysisResults(jobId: string): Promise<ResultsResponse> {
    const response = await fetch(`${this.baseUrl}/analysis/${jobId}/results`);
    
    if (!response.ok) {
      throw new Error(`Results fetch failed: ${response.statusText}`);
    }

    return response.json();
  }
}

export const klipStreamAPI = new KlipStreamAPI();
```

## 📡 API Endpoints

### Core Analysis Endpoints

#### 1. Start Analysis
```http
POST /api/v1/analysis
Content-Type: application/json

{
  "url": "https://www.twitch.tv/videos/1234567890"
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "message": "Analysis job created successfully",
  "progress": {
    "percentage": 0.0,
    "current_stage": "Queued",
    "estimated_completion_seconds": 1800
  }
}
```

#### 2. Check Status
```http
GET /api/v1/analysis/{job_id}/status
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "downloading|transcribing|analyzing|completed|failed",
  "progress": {
    "percentage": 45.5,
    "current_stage": "Transcribing",
    "message": "Processing audio transcription...",
    "estimated_completion_seconds": 900
  },
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:15:00Z"
}
```

#### 3. Get Results
```http
GET /api/v1/analysis/{job_id}/results
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "results": {
    "video_url": "https://storage.googleapis.com/bucket/video.mp4",
    "audio_url": "https://storage.googleapis.com/bucket/audio.wav",
    "transcript_url": "https://storage.googleapis.com/bucket/transcript.json",
    "analysis_url": "https://storage.googleapis.com/bucket/analysis.json",
    "highlights": [
      {
        "start_time": 120.5,
        "end_time": 180.5,
        "emotion": "excitement",
        "score": 0.85,
        "description": "High energy moment"
      }
    ],
    "sentiment_summary": {
      "overall_sentiment": "positive",
      "emotion_breakdown": {
        "excitement": 0.35,
        "happiness": 0.25,
        "funny": 0.20,
        "anger": 0.15,
        "sadness": 0.05
      }
    }
  }
}
```

### Utility Endpoints

#### Health Check
```http
GET /health
```

#### API Documentation
```http
GET /docs
```

#### Queue Status
```http
GET /api/v1/queue/status
```

## 🔄 Real-time Status Updates

### Polling Implementation

```typescript
// hooks/useAnalysisStatus.ts
import { useState, useEffect } from 'react';
import { klipStreamAPI } from '@/lib/klipstream-api';

export function useAnalysisStatus(jobId: string | null) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const pollStatus = async () => {
      try {
        setLoading(true);
        const response = await klipStreamAPI.getAnalysisStatus(jobId);
        setStatus(response);
        setError(null);

        // Stop polling if completed or failed
        if (['completed', 'failed'].includes(response.status)) {
          return;
        }

        // Continue polling
        setTimeout(pollStatus, 5000); // Poll every 5 seconds
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    pollStatus();
  }, [jobId]);

  return { status, loading, error };
}
```

### Usage in Components

```tsx
// components/AnalysisProgress.tsx
import { useAnalysisStatus } from '@/hooks/useAnalysisStatus';

interface AnalysisProgressProps {
  jobId: string;
  onComplete: (results: ResultsResponse) => void;
}

export function AnalysisProgress({ jobId, onComplete }: AnalysisProgressProps) {
  const { status, loading, error } = useAnalysisStatus(jobId);

  useEffect(() => {
    if (status?.status === 'completed') {
      // Fetch results when completed
      klipStreamAPI.getAnalysisResults(jobId)
        .then(onComplete)
        .catch(console.error);
    }
  }, [status?.status, jobId, onComplete]);

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!status) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="analysis-progress">
      <h3>Analysis Progress</h3>
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${status.progress.percentage}%` }}
        />
      </div>
      <p>{status.progress.current_stage}: {status.progress.percentage.toFixed(1)}%</p>
      {status.progress.message && <p>{status.progress.message}</p>}
      
      {status.progress.estimated_completion_seconds && (
        <p>Estimated completion: {Math.ceil(status.progress.estimated_completion_seconds / 60)} minutes</p>
      )}
    </div>
  );
}
```

## 🎯 Complete Integration Example

```tsx
// pages/analysis.tsx
import { useState } from 'react';
import { klipStreamAPI } from '@/lib/klipstream-api';
import { AnalysisProgress } from '@/components/AnalysisProgress';

export default function AnalysisPage() {
  const [twitchUrl, setTwitchUrl] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startAnalysis = async () => {
    if (!twitchUrl) return;

    try {
      setLoading(true);
      setError(null);
      
      const response = await klipStreamAPI.startAnalysis(twitchUrl);
      setJobId(response.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start analysis');
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = (analysisResults: ResultsResponse) => {
    setResults(analysisResults);
    setJobId(null); // Stop polling
  };

  return (
    <div className="analysis-page">
      <h1>KlipStream Analysis</h1>
      
      {!jobId && !results && (
        <div className="input-section">
          <input
            type="url"
            value={twitchUrl}
            onChange={(e) => setTwitchUrl(e.target.value)}
            placeholder="Enter Twitch VOD URL"
            className="url-input"
          />
          <button 
            onClick={startAnalysis} 
            disabled={loading || !twitchUrl}
            className="start-button"
          >
            {loading ? 'Starting...' : 'Start Analysis'}
          </button>
          {error && <div className="error">{error}</div>}
        </div>
      )}

      {jobId && (
        <AnalysisProgress 
          jobId={jobId} 
          onComplete={handleComplete}
        />
      )}

      {results && (
        <div className="results-section">
          <h2>Analysis Complete!</h2>
          <div className="results-grid">
            <div className="video-section">
              <h3>Video</h3>
              <video controls src={results.results.video_url} />
            </div>
            
            <div className="highlights-section">
              <h3>Highlights</h3>
              {results.results.highlights.map((highlight, index) => (
                <div key={index} className="highlight-item">
                  <span className="emotion">{highlight.emotion}</span>
                  <span className="time">
                    {Math.floor(highlight.start_time / 60)}:
                    {(highlight.start_time % 60).toFixed(0).padStart(2, '0')}
                  </span>
                  <span className="score">{(highlight.score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            
            <div className="sentiment-section">
              <h3>Sentiment Analysis</h3>
              <div className="sentiment-breakdown">
                {Object.entries(results.results.sentiment_summary.emotion_breakdown)
                  .map(([emotion, score]) => (
                    <div key={emotion} className="emotion-bar">
                      <span>{emotion}</span>
                      <div className="bar">
                        <div 
                          className="fill" 
                          style={{ width: `${score * 100}%` }}
                        />
                      </div>
                      <span>{(score * 100).toFixed(1)}%</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

## 🔧 Error Handling

### Common Error Scenarios

1. **Invalid Twitch URL**
   - Status: 400 Bad Request
   - Handle: Show user-friendly validation message

2. **Video Not Found**
   - Status: 404 Not Found
   - Handle: Suggest checking URL or video availability

3. **Analysis Timeout**
   - Status: 408 Request Timeout
   - Handle: Offer retry option

4. **Server Error**
   - Status: 500 Internal Server Error
   - Handle: Show generic error message and retry option

### Error Handling Implementation

```typescript
// utils/error-handler.ts
export function handleAPIError(error: any): string {
  if (error.status === 400) {
    return 'Invalid Twitch URL. Please check the URL format.';
  }
  if (error.status === 404) {
    return 'Video not found. Please check if the video exists and is public.';
  }
  if (error.status === 408) {
    return 'Analysis timed out. Please try again with a shorter video.';
  }
  if (error.status >= 500) {
    return 'Server error. Please try again later.';
  }
  return error.message || 'An unexpected error occurred.';
}
```

## 📊 TypeScript Definitions

```typescript
// types/klipstream.ts
export interface AnalysisResponse {
  job_id: string;
  status: string;
  message: string;
  progress: ProgressInfo;
}

export interface StatusResponse {
  job_id: string;
  status: 'queued' | 'downloading' | 'transcribing' | 'analyzing' | 'completed' | 'failed';
  progress: ProgressInfo;
  created_at: string;
  updated_at: string;
}

export interface ProgressInfo {
  percentage: number;
  current_stage: string;
  message?: string;
  estimated_completion_seconds?: number;
}

export interface ResultsResponse {
  job_id: string;
  status: string;
  results: AnalysisResults;
}

export interface AnalysisResults {
  video_url: string;
  audio_url: string;
  transcript_url: string;
  analysis_url: string;
  highlights: Highlight[];
  sentiment_summary: SentimentSummary;
}

export interface Highlight {
  start_time: number;
  end_time: number;
  emotion: string;
  score: number;
  description: string;
}

export interface SentimentSummary {
  overall_sentiment: string;
  emotion_breakdown: {
    excitement: number;
    happiness: number;
    funny: number;
    anger: number;
    sadness: number;
  };
}
```

## 🚦 Rate Limiting & Best Practices

### Rate Limits
- **Analysis requests**: 10 per hour per IP
- **Status checks**: 1 per second per job
- **Results fetching**: 100 per hour per IP

### Best Practices

1. **Implement exponential backoff** for failed requests
2. **Cache results** when possible
3. **Use polling intervals** of 5-10 seconds for status checks
4. **Handle timeouts gracefully** (analysis can take 15-30 minutes)
5. **Validate URLs** before sending to API
6. **Show progress indicators** for better UX
7. **Implement retry logic** for transient failures

## 🔐 Security Considerations

1. **No authentication required** for the current API
2. **CORS enabled** for web applications
3. **Rate limiting** prevents abuse
4. **Input validation** on server side
5. **HTTPS only** communication

## 📞 Support & Troubleshooting

### Health Check
Always verify API health before making requests:
```typescript
const healthCheck = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.ok;
};
```

### Debug Mode
Enable debug logging in development:
```typescript
const DEBUG = process.env.NODE_ENV === 'development';

if (DEBUG) {
  console.log('API Request:', { url, method, body });
}
```

### Common Issues
1. **CORS errors**: Ensure requests are made from allowed origins
2. **Timeout errors**: Increase timeout for long-running operations
3. **Network errors**: Implement retry logic with exponential backoff

## 📚 Additional Resources

- **API Documentation**: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/docs
- **OpenAPI Spec**: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/openapi.json
- **Health Check**: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/health
- **Queue Status**: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/queue/status

## 🔄 Changelog

### v2.0.0 (Current)
- ✅ FastAPI subprocess wrapper fix implemented
- ✅ Status update consistency improvements
- ✅ Real-time progress tracking
- ✅ Enhanced error handling
- ✅ Production-ready deployment

### v1.0.0 (Legacy)
- Basic analysis pipeline
- Simple status tracking
- Limited error handling
