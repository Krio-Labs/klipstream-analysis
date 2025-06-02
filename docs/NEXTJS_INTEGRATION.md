# 🔗 Next.js Integration Guide for KlipStream Analysis API

## Quick Start

This guide provides everything you need to integrate the KlipStream Analysis API into your Next.js application.

## Prerequisites

- Next.js 13+ (App Router or Pages Router)
- TypeScript (recommended)
- React 18+

## Installation

No additional packages required beyond standard Next.js dependencies. The API uses native `fetch`.

## 1. API Service Layer

Create `lib/klipstream-api.ts`:

```typescript
// Types
export interface AnalysisRequest {
  url: string;
}

export interface AnalysisResponse {
  status: 'success' | 'failed';
  message: string;
  video_id: string;
  execution_time?: {
    raw_pipeline: number;
    analysis_pipeline: number;
    total: number;
  };
  files?: {
    audio_file: string;
    video_file: string;
    waveform_file: string;
    transcript_files: {
      json: string;
      words: string;
      paragraphs: string;
    };
    chat_files: {
      raw: string;
      processed: string;
    };
    analysis_files: {
      integrated: string;
      sentiment: string;
      highlights: string;
    };
  };
  error?: string;
}

// API Configuration
const API_BASE_URL = 'https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app';
const API_TIMEOUT = 3600000; // 1 hour in milliseconds

// Main API function
export async function analyzeVideo(twitchUrl: string): Promise<AnalysisResponse> {
  if (!validateTwitchUrl(twitchUrl)) {
    throw new Error('Invalid Twitch URL format');
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);

  try {
    const response = await fetch(API_BASE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: twitchUrl }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Request timed out after 1 hour');
    }
    throw error;
  }
}

// Utility functions
export function extractVideoId(twitchUrl: string): string | null {
  const match = twitchUrl.match(/\/videos\/(\d+)/);
  return match ? match[1] : null;
}

export function validateTwitchUrl(url: string): boolean {
  const twitchUrlPattern = /^https:\/\/www\.twitch\.tv\/videos\/\d+/;
  return twitchUrlPattern.test(url);
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${remainingSeconds}s`;
  }
  return `${minutes}m ${remainingSeconds}s`;
}
```

## 2. React Hook

Create `hooks/useVideoAnalysis.ts`:

```typescript
import { useState, useCallback } from 'react';
import { analyzeVideo, AnalysisResponse } from '@/lib/klipstream-api';

interface UseVideoAnalysisReturn {
  analyze: (url: string) => Promise<void>;
  isLoading: boolean;
  result: AnalysisResponse | null;
  error: string | null;
  reset: () => void;
  progress: number;
}

export function useVideoAnalysis(): UseVideoAnalysisReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const analyze = useCallback(async (url: string) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);

    // Simulate progress for better UX
    const progressInterval = setInterval(() => {
      setProgress(prev => Math.min(prev + 1, 95));
    }, 3000);

    try {
      const response = await analyzeVideo(url);
      setResult(response);
      setProgress(100);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setProgress(0);
    } finally {
      clearInterval(progressInterval);
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setIsLoading(false);
    setResult(null);
    setError(null);
    setProgress(0);
  }, []);

  return { analyze, isLoading, result, error, reset, progress };
}
```

## 3. Main Analysis Component

Create `components/VideoAnalysis.tsx`:

```tsx
'use client';

import { useState } from 'react';
import { useVideoAnalysis } from '@/hooks/useVideoAnalysis';
import { validateTwitchUrl, extractVideoId, formatDuration } from '@/lib/klipstream-api';

export default function VideoAnalysis() {
  const [url, setUrl] = useState('');
  const { analyze, isLoading, result, error, reset, progress } = useVideoAnalysis();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateTwitchUrl(url)) {
      alert('Please enter a valid Twitch VOD URL (e.g., https://www.twitch.tv/videos/123456789)');
      return;
    }

    await analyze(url);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          KlipStream Video Analysis
        </h1>
        <p className="text-gray-600">
          Analyze Twitch VODs for transcripts, sentiment, and highlights
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
              Twitch VOD URL
            </label>
            <input
              id="url"
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.twitch.tv/videos/123456789"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
              required
            />
          </div>
          
          <div className="flex gap-4">
            <button
              type="submit"
              disabled={isLoading || !url}
              className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {isLoading ? 'Analyzing...' : 'Start Analysis'}
            </button>
            
            {(result || error) && (
              <button
                type="button"
                onClick={reset}
                className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-medium"
              >
                Reset
              </button>
            )}
          </div>
        </form>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-center mb-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-4"></div>
            <div>
              <h3 className="font-semibold text-blue-800">Processing Video</h3>
              <p className="text-blue-600">Video ID: {extractVideoId(url)}</p>
            </div>
          </div>
          
          <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          
          <p className="text-sm text-blue-600">
            This typically takes 5-10 minutes. Please keep this page open.
          </p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 mb-2">❌ Analysis Failed</h3>
          <p className="text-red-600 mb-4">{error}</p>
          
          <div className="text-sm text-red-600">
            <p className="font-medium mb-1">Common solutions:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>Verify the video is public and accessible</li>
              <li>Check that the URL format is correct</li>
              <li>Try again in a few minutes</li>
            </ul>
          </div>
        </div>
      )}

      {/* Success State */}
      {result && result.status === 'success' && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <h3 className="font-semibold text-green-800 mb-4">
            ✅ Analysis Complete!
          </h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Video Information */}
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">Video Information</h4>
                <div className="bg-white rounded p-4 space-y-2">
                  <p><span className="font-medium">Video ID:</span> {result.video_id}</p>
                  {result.execution_time && (
                    <>
                      <p><span className="font-medium">Total Time:</span> {formatDuration(result.execution_time.total)}</p>
                      <div className="text-sm text-gray-600">
                        <p>Raw Processing: {formatDuration(result.execution_time.raw_pipeline)}</p>
                        <p>Analysis: {formatDuration(result.execution_time.analysis_pipeline)}</p>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Generated Files */}
            {result.files && (
              <div className="space-y-4">
                <h4 className="font-semibold text-gray-800 mb-2">Generated Files</h4>
                
                <div className="bg-white rounded p-4 space-y-3">
                  <FileSection 
                    title="Media Files" 
                    files={[
                      { name: "Video (MP4)", url: result.files.video_file },
                      { name: "Audio (WAV)", url: result.files.audio_file },
                      { name: "Waveform (PNG)", url: result.files.waveform_file }
                    ]} 
                  />
                  
                  <FileSection 
                    title="Transcripts" 
                    files={[
                      { name: "Full Transcript (JSON)", url: result.files.transcript_files.json },
                      { name: "Word Timestamps (CSV)", url: result.files.transcript_files.words },
                      { name: "Paragraphs (CSV)", url: result.files.transcript_files.paragraphs }
                    ]} 
                  />
                  
                  <FileSection 
                    title="Analysis Results" 
                    files={[
                      { name: "Integrated Analysis (JSON)", url: result.files.analysis_files.integrated },
                      { name: "Sentiment Analysis (CSV)", url: result.files.analysis_files.sentiment },
                      { name: "Highlights (JSON)", url: result.files.analysis_files.highlights }
                    ]} 
                  />
                  
                  <FileSection 
                    title="Chat Data" 
                    files={[
                      { name: "Raw Chat (JSON)", url: result.files.chat_files.raw },
                      { name: "Processed Chat (CSV)", url: result.files.chat_files.processed }
                    ]} 
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Helper component for file sections
function FileSection({ title, files }: { 
  title: string; 
  files: { name: string; url: string }[] 
}) {
  return (
    <div>
      <h5 className="font-medium text-gray-700 mb-2">{title}</h5>
      <ul className="space-y-1">
        {files.map((file, index) => (
          <li key={index}>
            <a 
              href={file.url} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 hover:underline text-sm"
            >
              📄 {file.name}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

## 4. Usage in App Router

Create `app/analysis/page.tsx`:

```tsx
import VideoAnalysis from '@/components/VideoAnalysis';

export default function AnalysisPage() {
  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <VideoAnalysis />
    </main>
  );
}
```

## 5. Error Boundary (Recommended)

Create `components/ErrorBoundary.tsx`:

```tsx
'use client';

import { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-md p-6 max-w-md w-full">
            <h2 className="text-xl font-semibold text-red-600 mb-4">
              Something went wrong
            </h2>
            <p className="text-gray-600 mb-4">
              An unexpected error occurred. Please refresh the page and try again.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
```

## 6. Environment Configuration

Create `.env.local`:

```env
# Optional: Override API URL for development
NEXT_PUBLIC_KLIPSTREAM_API_URL=https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app

# Optional: Enable debug logging
NEXT_PUBLIC_DEBUG=false
```

## Best Practices

### 1. User Experience
- Show clear progress indicators
- Provide estimated completion times
- Prevent accidental page navigation during processing
- Display helpful error messages

### 2. Error Handling
- Validate URLs before submission
- Handle network timeouts gracefully
- Provide retry mechanisms
- Log errors for debugging

### 3. Performance
- Use React.memo for expensive components
- Implement proper loading states
- Consider implementing request cancellation
- Cache results when appropriate

### 4. Security
- Validate all user inputs
- Sanitize URLs before processing
- Implement rate limiting on client side
- Use HTTPS for all requests

## Testing

Create `__tests__/klipstream-api.test.ts`:

```typescript
import { validateTwitchUrl, extractVideoId, formatDuration } from '@/lib/klipstream-api';

describe('KlipStream API Utils', () => {
  test('validates Twitch URLs correctly', () => {
    expect(validateTwitchUrl('https://www.twitch.tv/videos/123456789')).toBe(true);
    expect(validateTwitchUrl('https://twitch.tv/videos/123456789')).toBe(false);
    expect(validateTwitchUrl('https://www.twitch.tv/user123')).toBe(false);
  });

  test('extracts video ID correctly', () => {
    expect(extractVideoId('https://www.twitch.tv/videos/123456789')).toBe('123456789');
    expect(extractVideoId('invalid-url')).toBe(null);
  });

  test('formats duration correctly', () => {
    expect(formatDuration(65)).toBe('1m 5s');
    expect(formatDuration(3665)).toBe('1h 1m 5s');
  });
});
```

## Deployment Considerations

### Vercel
- No additional configuration needed
- API routes work out of the box
- Consider timeout limits for long requests

### Netlify
- May need to configure timeout settings
- Consider using Netlify Functions for proxy

### Self-hosted
- Ensure proper timeout configurations
- Configure reverse proxy if needed
- Monitor memory usage for long requests

This integration provides a complete, production-ready solution for using the KlipStream Analysis API in Next.js applications!
