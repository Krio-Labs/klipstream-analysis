# Deployment & Troubleshooting Guide

## 🚀 Frontend Deployment Considerations

### Environment Configuration

#### Production Environment Variables
```env
# .env.production
NEXT_PUBLIC_KLIPSTREAM_API_URL=https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app
NEXT_PUBLIC_KLIPSTREAM_API_VERSION=v1
NEXT_PUBLIC_ENVIRONMENT=production

# Optional: Analytics and monitoring
NEXT_PUBLIC_ANALYTICS_ID=your-analytics-id
NEXT_PUBLIC_SENTRY_DSN=your-sentry-dsn
```

#### Development Environment Variables
```env
# .env.local
NEXT_PUBLIC_KLIPSTREAM_API_URL=http://localhost:3000
NEXT_PUBLIC_KLIPSTREAM_API_VERSION=v1
NEXT_PUBLIC_ENVIRONMENT=development
NEXT_PUBLIC_DEBUG=true
```

### CORS Configuration

The Cloud Run API is configured to accept requests from any origin. For production, you may want to restrict this:

```typescript
// If you need to handle CORS issues
const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_KLIPSTREAM_API_URL,
  timeout: 60000, // 60 seconds for long-running requests
  headers: {
    'Content-Type': 'application/json',
  },
});
```

### Performance Optimization

#### API Response Caching
```typescript
// lib/api-cache.ts
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export function getCachedResponse<T>(key: string): T | null {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return cached.data;
  }
  return null;
}

export function setCachedResponse<T>(key: string, data: T): void {
  cache.set(key, { data, timestamp: Date.now() });
}

// Usage in API calls
export async function getAnalysisStatus(jobId: string): Promise<StatusResponse> {
  const cacheKey = `status-${jobId}`;
  const cached = getCachedResponse<StatusResponse>(cacheKey);
  
  if (cached && cached.status !== 'completed' && cached.status !== 'failed') {
    return cached;
  }

  const response = await fetch(`${API_BASE_URL}/analysis/${jobId}/status`);
  const data = await response.json();
  
  setCachedResponse(cacheKey, data);
  return data;
}
```

#### Lazy Loading Components
```typescript
// Lazy load heavy components
import dynamic from 'next/dynamic';

const AnalysisResults = dynamic(() => import('@/components/AnalysisResults'), {
  loading: () => <div>Loading results...</div>,
  ssr: false, // Disable SSR for client-heavy components
});

const VideoPlayer = dynamic(() => import('@/components/VideoPlayer'), {
  loading: () => <div>Loading video player...</div>,
  ssr: false,
});
```

## 🔧 Troubleshooting Common Issues

### 1. API Connection Issues

#### Problem: "Failed to fetch" or CORS errors
```typescript
// Solution: Add proper error handling and retry logic
async function apiRequest<T>(url: string, options: RequestInit = {}): Promise<T> {
  const maxRetries = 3;
  let lastError: Error;

  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      lastError = error as Error;
      
      // Don't retry on client errors (4xx)
      if (error instanceof Error && error.message.includes('4')) {
        throw error;
      }

      // Wait before retry (exponential backoff)
      if (i < maxRetries - 1) {
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
      }
    }
  }

  throw lastError!;
}
```

#### Problem: Request timeouts
```typescript
// Solution: Implement proper timeout handling
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

try {
  const response = await fetch(url, {
    signal: controller.signal,
    ...options,
  });
  clearTimeout(timeoutId);
  return response;
} catch (error) {
  clearTimeout(timeoutId);
  if (error.name === 'AbortError') {
    throw new Error('Request timed out');
  }
  throw error;
}
```

### 2. Status Polling Issues

#### Problem: Excessive API calls
```typescript
// Solution: Implement smart polling with backoff
class SmartPoller {
  private intervalId: NodeJS.Timeout | null = null;
  private pollCount = 0;

  start(jobId: string, callback: (status: StatusResponse) => void) {
    this.poll(jobId, callback);
  }

  private async poll(jobId: string, callback: (status: StatusResponse) => void) {
    try {
      const status = await klipStreamAPI.getAnalysisStatus(jobId);
      callback(status);

      if (['completed', 'failed'].includes(status.status)) {
        this.stop();
        return;
      }

      // Adaptive polling interval
      const interval = this.getPollingInterval();
      this.intervalId = setTimeout(() => this.poll(jobId, callback), interval);
      this.pollCount++;
    } catch (error) {
      console.error('Polling error:', error);
      // Retry with longer interval on error
      this.intervalId = setTimeout(() => this.poll(jobId, callback), 10000);
    }
  }

  private getPollingInterval(): number {
    // Start with 2 seconds, increase to max 30 seconds
    return Math.min(2000 + (this.pollCount * 1000), 30000);
  }

  stop() {
    if (this.intervalId) {
      clearTimeout(this.intervalId);
      this.intervalId = null;
    }
    this.pollCount = 0;
  }
}
```

### 3. Memory and Performance Issues

#### Problem: Memory leaks from polling
```typescript
// Solution: Proper cleanup in useEffect
useEffect(() => {
  if (!jobId) return;

  const poller = new SmartPoller();
  poller.start(jobId, setStatus);

  // Cleanup function
  return () => {
    poller.stop();
  };
}, [jobId]);
```

#### Problem: Large video files causing browser issues
```typescript
// Solution: Progressive loading and streaming
const VideoPlayer = ({ videoUrl }: { videoUrl: string }) => {
  const [isLoading, setIsLoading] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadStart = () => setIsLoading(true);
    const handleCanPlay = () => setIsLoading(false);

    video.addEventListener('loadstart', handleLoadStart);
    video.addEventListener('canplay', handleCanPlay);

    return () => {
      video.removeEventListener('loadstart', handleLoadStart);
      video.removeEventListener('canplay', handleCanPlay);
    };
  }, []);

  return (
    <div className="video-container">
      {isLoading && <div className="video-loading">Loading video...</div>}
      <video
        ref={videoRef}
        controls
        preload="metadata" // Only load metadata initially
        src={videoUrl}
        className="video-player"
      />
    </div>
  );
};
```

### 4. Error Handling Best Practices

#### Comprehensive Error Boundary
```typescript
// components/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
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

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    
    // Send to error reporting service
    if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
      // Sentry.captureException(error);
    }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <p>We're sorry, but something unexpected happened.</p>
          <button onClick={() => this.setState({ hasError: false })}>
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
```

#### API Error Handler
```typescript
// utils/error-handler.ts
export interface APIError {
  code: string;
  message: string;
  status: number;
  details?: any;
}

export function handleAPIError(error: any): APIError {
  // Network errors
  if (!error.response) {
    return {
      code: 'NETWORK_ERROR',
      message: 'Unable to connect to the server. Please check your internet connection.',
      status: 0,
    };
  }

  // HTTP errors
  const { status, data } = error.response;
  
  switch (status) {
    case 400:
      return {
        code: 'INVALID_REQUEST',
        message: data?.error?.message || 'Invalid request. Please check your input.',
        status,
        details: data?.error?.details,
      };
    
    case 404:
      return {
        code: 'NOT_FOUND',
        message: 'The requested resource was not found.',
        status,
      };
    
    case 429:
      return {
        code: 'RATE_LIMITED',
        message: 'Too many requests. Please wait a moment before trying again.',
        status,
      };
    
    case 500:
      return {
        code: 'SERVER_ERROR',
        message: 'Server error. Please try again later.',
        status,
      };
    
    default:
      return {
        code: 'UNKNOWN_ERROR',
        message: data?.error?.message || 'An unexpected error occurred.',
        status,
        details: data?.error?.details,
      };
  }
}
```

## 📊 Monitoring and Analytics

### Performance Monitoring
```typescript
// utils/performance.ts
export function trackAPICall(endpoint: string, duration: number, success: boolean) {
  // Send to analytics service
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', 'api_call', {
      endpoint,
      duration,
      success,
    });
  }
}

export function trackAnalysisFlow(step: string, jobId: string) {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', 'analysis_flow', {
      step,
      job_id: jobId,
    });
  }
}
```

### User Experience Metrics
```typescript
// hooks/useAnalytics.ts
export function useAnalytics() {
  const trackEvent = useCallback((event: string, properties: Record<string, any>) => {
    if (process.env.NEXT_PUBLIC_ENVIRONMENT === 'production') {
      // Send to analytics service
      console.log('Analytics event:', event, properties);
    }
  }, []);

  const trackAnalysisStart = useCallback((twitchUrl: string) => {
    trackEvent('analysis_started', { url: twitchUrl });
  }, [trackEvent]);

  const trackAnalysisComplete = useCallback((jobId: string, duration: number) => {
    trackEvent('analysis_completed', { job_id: jobId, duration });
  }, [trackEvent]);

  return {
    trackEvent,
    trackAnalysisStart,
    trackAnalysisComplete,
  };
}
```

## 🔍 Debugging Tools

### Debug Panel Component
```typescript
// components/DebugPanel.tsx (only in development)
export function DebugPanel({ status, jobId }: { status?: StatusResponse; jobId?: string }) {
  if (process.env.NEXT_PUBLIC_ENVIRONMENT !== 'development') {
    return null;
  }

  return (
    <div className="debug-panel">
      <h4>Debug Information</h4>
      <div className="debug-info">
        <p><strong>Job ID:</strong> {jobId || 'None'}</p>
        <p><strong>Status:</strong> {status?.status || 'None'}</p>
        <p><strong>Progress:</strong> {status?.progress.percentage || 0}%</p>
        <p><strong>Stage:</strong> {status?.progress.current_stage || 'None'}</p>
        <p><strong>API URL:</strong> {process.env.NEXT_PUBLIC_KLIPSTREAM_API_URL}</p>
      </div>
      
      {status && (
        <details>
          <summary>Full Status Object</summary>
          <pre>{JSON.stringify(status, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}
```

### API Health Check
```typescript
// utils/health-check.ts
export async function checkAPIHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_KLIPSTREAM_API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

// Use in app initialization
useEffect(() => {
  checkAPIHealth().then(isHealthy => {
    if (!isHealthy) {
      console.warn('API health check failed');
      // Show user notification
    }
  });
}, []);
```

This guide provides comprehensive solutions for common deployment and troubleshooting scenarios when integrating with the KlipStream Analysis API.
