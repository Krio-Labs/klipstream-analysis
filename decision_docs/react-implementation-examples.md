# React/Next.js Implementation Examples

## 🎯 Complete Implementation Examples

### 1. Custom Hook for Analysis Management

```typescript
// hooks/useKlipStreamAnalysis.ts
import { useState, useCallback, useEffect } from 'react';
import { klipStreamAPI } from '@/lib/klipstream-api';

interface UseAnalysisState {
  jobId: string | null;
  status: StatusResponse | null;
  results: ResultsResponse | null;
  loading: boolean;
  error: string | null;
}

export function useKlipStreamAnalysis() {
  const [state, setState] = useState<UseAnalysisState>({
    jobId: null,
    status: null,
    results: null,
    loading: false,
    error: null,
  });

  const startAnalysis = useCallback(async (twitchUrl: string) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await klipStreamAPI.startAnalysis(twitchUrl);
      setState(prev => ({
        ...prev,
        jobId: response.job_id,
        status: response,
        loading: false,
      }));
      return response.job_id;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to start analysis',
        loading: false,
      }));
      throw error;
    }
  }, []);

  const pollStatus = useCallback(async (jobId: string) => {
    try {
      const status = await klipStreamAPI.getAnalysisStatus(jobId);
      setState(prev => ({ ...prev, status, error: null }));

      if (status.status === 'completed') {
        const results = await klipStreamAPI.getAnalysisResults(jobId);
        setState(prev => ({ ...prev, results }));
        return results;
      }

      return status;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to check status',
      }));
      throw error;
    }
  }, []);

  const reset = useCallback(() => {
    setState({
      jobId: null,
      status: null,
      results: null,
      loading: false,
      error: null,
    });
  }, []);

  // Auto-polling effect
  useEffect(() => {
    if (!state.jobId || state.status?.status === 'completed' || state.status?.status === 'failed') {
      return;
    }

    const interval = setInterval(() => {
      pollStatus(state.jobId!);
    }, 5000);

    return () => clearInterval(interval);
  }, [state.jobId, state.status?.status, pollStatus]);

  return {
    ...state,
    startAnalysis,
    pollStatus,
    reset,
    isProcessing: state.status?.status && !['completed', 'failed'].includes(state.status.status),
  };
}
```

### 2. Analysis Form Component

```tsx
// components/AnalysisForm.tsx
import { useState } from 'react';
import { useKlipStreamAnalysis } from '@/hooks/useKlipStreamAnalysis';

interface AnalysisFormProps {
  onAnalysisStart?: (jobId: string) => void;
  onAnalysisComplete?: (results: ResultsResponse) => void;
}

export function AnalysisForm({ onAnalysisStart, onAnalysisComplete }: AnalysisFormProps) {
  const [url, setUrl] = useState('');
  const { startAnalysis, loading, error } = useKlipStreamAnalysis();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isValidTwitchUrl(url)) {
      alert('Please enter a valid Twitch VOD URL');
      return;
    }

    try {
      const jobId = await startAnalysis(url);
      onAnalysisStart?.(jobId);
    } catch (error) {
      console.error('Failed to start analysis:', error);
    }
  };

  const isValidTwitchUrl = (url: string): boolean => {
    const twitchRegex = /^https:\/\/(www\.)?twitch\.tv\/videos\/\d+/;
    return twitchRegex.test(url);
  };

  return (
    <form onSubmit={handleSubmit} className="analysis-form">
      <div className="form-group">
        <label htmlFor="twitch-url">Twitch VOD URL</label>
        <input
          id="twitch-url"
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://www.twitch.tv/videos/1234567890"
          className="url-input"
          disabled={loading}
          required
        />
        <small className="help-text">
          Enter a Twitch VOD URL to analyze sentiment and find highlights
        </small>
      </div>

      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}

      <button 
        type="submit" 
        disabled={loading || !url || !isValidTwitchUrl(url)}
        className="submit-button"
      >
        {loading ? (
          <>
            <span className="spinner" />
            Starting Analysis...
          </>
        ) : (
          'Start Analysis'
        )}
      </button>
    </form>
  );
}
```

### 3. Progress Indicator Component

```tsx
// components/AnalysisProgress.tsx
import { useEffect, useState } from 'react';

interface AnalysisProgressProps {
  status: StatusResponse;
  className?: string;
}

const STAGE_DESCRIPTIONS = {
  queued: 'Preparing analysis...',
  downloading: 'Downloading video and chat data...',
  transcribing: 'Converting audio to text...',
  analyzing: 'Analyzing sentiment and emotions...',
  completed: 'Analysis complete!',
  failed: 'Analysis failed',
};

const STAGE_ICONS = {
  queued: '⏳',
  downloading: '⬇️',
  transcribing: '🎤',
  analyzing: '🧠',
  completed: '✅',
  failed: '❌',
};

export function AnalysisProgress({ status, className = '' }: AnalysisProgressProps) {
  const [timeElapsed, setTimeElapsed] = useState(0);

  useEffect(() => {
    const startTime = new Date(status.created_at).getTime();
    
    const interval = setInterval(() => {
      const now = Date.now();
      setTimeElapsed(Math.floor((now - startTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [status.created_at]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getEstimatedTimeRemaining = (): string => {
    if (!status.progress.estimated_completion_seconds) return 'Calculating...';
    
    const remaining = Math.max(0, status.progress.estimated_completion_seconds - timeElapsed);
    return formatTime(remaining);
  };

  return (
    <div className={`analysis-progress ${className}`}>
      <div className="progress-header">
        <div className="stage-info">
          <span className="stage-icon">
            {STAGE_ICONS[status.status as keyof typeof STAGE_ICONS]}
          </span>
          <div>
            <h3 className="stage-title">{status.progress.current_stage}</h3>
            <p className="stage-description">
              {STAGE_DESCRIPTIONS[status.status as keyof typeof STAGE_DESCRIPTIONS]}
            </p>
          </div>
        </div>
        
        <div className="progress-stats">
          <div className="stat">
            <span className="stat-label">Progress</span>
            <span className="stat-value">{status.progress.percentage.toFixed(1)}%</span>
          </div>
          <div className="stat">
            <span className="stat-label">Elapsed</span>
            <span className="stat-value">{formatTime(timeElapsed)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Remaining</span>
            <span className="stat-value">{getEstimatedTimeRemaining()}</span>
          </div>
        </div>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{ 
              width: `${status.progress.percentage}%`,
              transition: 'width 0.3s ease-in-out'
            }}
          />
        </div>
        <span className="progress-text">
          {status.progress.percentage.toFixed(1)}%
        </span>
      </div>

      {status.progress.message && (
        <div className="progress-message">
          <span className="message-icon">ℹ️</span>
          {status.progress.message}
        </div>
      )}

      <div className="progress-stages">
        {Object.keys(STAGE_DESCRIPTIONS).slice(0, -2).map((stage, index) => (
          <div 
            key={stage}
            className={`stage ${getStageStatus(stage, status.status, index)}`}
          >
            <div className="stage-dot" />
            <span className="stage-name">{stage}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function getStageStatus(stage: string, currentStatus: string, index: number): string {
  const stages = ['queued', 'downloading', 'transcribing', 'analyzing'];
  const currentIndex = stages.indexOf(currentStatus);
  
  if (currentStatus === 'completed') return 'completed';
  if (currentStatus === 'failed') return index <= currentIndex ? 'failed' : 'pending';
  if (index < currentIndex) return 'completed';
  if (index === currentIndex) return 'active';
  return 'pending';
}
```

### 4. Results Display Component

```tsx
// components/AnalysisResults.tsx
import { useState } from 'react';

interface AnalysisResultsProps {
  results: ResultsResponse;
  onNewAnalysis?: () => void;
}

export function AnalysisResults({ results, onNewAnalysis }: AnalysisResultsProps) {
  const [activeTab, setActiveTab] = useState<'highlights' | 'sentiment' | 'transcript'>('highlights');
  const [selectedHighlight, setSelectedHighlight] = useState<number | null>(null);

  const downloadFile = async (url: string, filename: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  return (
    <div className="analysis-results">
      <div className="results-header">
        <h2>Analysis Complete! 🎉</h2>
        <div className="header-actions">
          <button onClick={onNewAnalysis} className="new-analysis-btn">
            New Analysis
          </button>
          <div className="download-menu">
            <button className="download-btn">Download Files ⬇️</button>
            <div className="download-dropdown">
              <button onClick={() => downloadFile(results.results.video_url, 'video.mp4')}>
                Video (MP4)
              </button>
              <button onClick={() => downloadFile(results.results.audio_url, 'audio.wav')}>
                Audio (WAV)
              </button>
              <button onClick={() => downloadFile(results.results.transcript_url, 'transcript.json')}>
                Transcript (JSON)
              </button>
              <button onClick={() => downloadFile(results.results.analysis_url, 'analysis.json')}>
                Full Analysis (JSON)
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="results-summary">
        <div className="summary-card">
          <h3>Overall Sentiment</h3>
          <div className="sentiment-badge">
            {results.results.sentiment_summary.overall_sentiment}
          </div>
        </div>
        <div className="summary-card">
          <h3>Highlights Found</h3>
          <div className="highlight-count">
            {results.results.highlights.length}
          </div>
        </div>
        <div className="summary-card">
          <h3>Duration</h3>
          <div className="duration">
            {formatDuration(results.results.sentiment_summary.total_duration)}
          </div>
        </div>
      </div>

      <div className="results-tabs">
        <button 
          className={`tab ${activeTab === 'highlights' ? 'active' : ''}`}
          onClick={() => setActiveTab('highlights')}
        >
          Highlights ({results.results.highlights.length})
        </button>
        <button 
          className={`tab ${activeTab === 'sentiment' ? 'active' : ''}`}
          onClick={() => setActiveTab('sentiment')}
        >
          Sentiment Analysis
        </button>
        <button 
          className={`tab ${activeTab === 'transcript' ? 'active' : ''}`}
          onClick={() => setActiveTab('transcript')}
        >
          Transcript
        </button>
      </div>

      <div className="results-content">
        {activeTab === 'highlights' && (
          <div className="highlights-section">
            <div className="video-player">
              <video 
                controls 
                src={results.results.video_url}
                className="main-video"
              />
            </div>
            
            <div className="highlights-list">
              {results.results.highlights.map((highlight, index) => (
                <div 
                  key={index}
                  className={`highlight-item ${selectedHighlight === index ? 'selected' : ''}`}
                  onClick={() => setSelectedHighlight(index)}
                >
                  <div className="highlight-emotion">
                    <span className={`emotion-badge ${highlight.emotion}`}>
                      {highlight.emotion}
                    </span>
                    <span className="confidence">
                      {(highlight.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="highlight-time">
                    {formatTime(highlight.start_time)} - {formatTime(highlight.end_time)}
                  </div>
                  
                  <div className="highlight-description">
                    {highlight.description}
                  </div>
                  
                  <button 
                    className="play-highlight"
                    onClick={(e) => {
                      e.stopPropagation();
                      // Jump to highlight time in video
                      const video = document.querySelector('.main-video') as HTMLVideoElement;
                      if (video) {
                        video.currentTime = highlight.start_time;
                        video.play();
                      }
                    }}
                  >
                    ▶️ Play
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'sentiment' && (
          <div className="sentiment-section">
            <div className="emotion-breakdown">
              <h3>Emotion Breakdown</h3>
              {Object.entries(results.results.sentiment_summary.emotion_breakdown)
                .sort(([,a], [,b]) => b - a)
                .map(([emotion, score]) => (
                  <div key={emotion} className="emotion-bar">
                    <div className="emotion-label">
                      <span className="emotion-name">{emotion}</span>
                      <span className="emotion-percentage">
                        {(score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="emotion-progress">
                      <div 
                        className={`emotion-fill ${emotion}`}
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
            </div>
            
            <div className="sentiment-chart">
              {/* Add chart component here */}
              <p>Sentiment timeline chart would go here</p>
            </div>
          </div>
        )}

        {activeTab === 'transcript' && (
          <div className="transcript-section">
            <div className="transcript-controls">
              <button onClick={() => downloadFile(results.results.transcript_url, 'transcript.json')}>
                Download Full Transcript
              </button>
            </div>
            <div className="transcript-preview">
              <p>Transcript preview would be loaded here from the transcript URL</p>
              <a href={results.results.transcript_url} target="_blank" rel="noopener noreferrer">
                View Full Transcript
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}
```

### 5. Main Analysis Page

```tsx
// pages/analysis.tsx or app/analysis/page.tsx
import { useState } from 'react';
import { AnalysisForm } from '@/components/AnalysisForm';
import { AnalysisProgress } from '@/components/AnalysisProgress';
import { AnalysisResults } from '@/components/AnalysisResults';
import { useKlipStreamAnalysis } from '@/hooks/useKlipStreamAnalysis';

export default function AnalysisPage() {
  const { jobId, status, results, error, reset } = useKlipStreamAnalysis();

  const handleAnalysisStart = (newJobId: string) => {
    // Analysis started, jobId is automatically set by the hook
    console.log('Analysis started:', newJobId);
  };

  const handleNewAnalysis = () => {
    reset();
  };

  return (
    <div className="analysis-page">
      <div className="container">
        <header className="page-header">
          <h1>KlipStream Analysis</h1>
          <p>Analyze Twitch VODs for sentiment and highlights</p>
        </header>

        <main className="page-content">
          {!jobId && !results && (
            <AnalysisForm onAnalysisStart={handleAnalysisStart} />
          )}

          {jobId && status && !results && (
            <AnalysisProgress status={status} />
          )}

          {results && (
            <AnalysisResults 
              results={results} 
              onNewAnalysis={handleNewAnalysis}
            />
          )}

          {error && (
            <div className="error-container">
              <h3>Analysis Failed</h3>
              <p>{error}</p>
              <button onClick={handleNewAnalysis} className="retry-button">
                Try Again
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
```

This comprehensive set of React/Next.js components provides a complete implementation for integrating with the KlipStream Analysis API, including form handling, real-time progress tracking, and results display.
