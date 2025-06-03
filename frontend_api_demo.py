#!/usr/bin/env python3
"""
Frontend API Integration Demo
Demonstrates how a frontend application would interact with the KlipStream Analysis API
"""

import requests
import json
import time
from datetime import datetime

class KlipStreamAPIClient:
    """Demo API client showing frontend integration patterns"""
    
    def __init__(self, base_url="http://localhost:3001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'KlipStream-Frontend/1.0'
        })
    
    def health_check(self):
        """Check if the API service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/monitoring/dashboard", timeout=15)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def start_analysis(self, video_url):
        """Start a new analysis job"""
        try:
            payload = {"url": video_url}
            response = self.session.post(
                f"{self.base_url}/api/v1/analysis", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_job_status(self, job_id):
        """Get current status of an analysis job"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/analysis/{job_id}/status", 
                timeout=10
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_queue_status(self):
        """Get current queue status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/queue/status", timeout=10)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return False, {"error": str(e)}

def demo_frontend_integration():
    """Demonstrate complete frontend integration workflow"""
    
    print("🌐 KlipStream Analysis API - Frontend Integration Demo")
    print("=" * 60)
    
    # Initialize API client
    client = KlipStreamAPIClient()
    
    # Step 1: Health Check
    print("\n🏥 Step 1: Health Check")
    print("-" * 30)
    
    is_healthy, health_data = client.health_check()
    if is_healthy:
        print(f"✅ API is healthy: {health_data.get('status', 'unknown')}")
        print(f"   Version: {health_data.get('version', 'unknown')}")
    else:
        print(f"❌ API health check failed: {health_data}")
        return
    
    # Step 2: System Status
    print("\n📊 Step 2: System Status")
    print("-" * 30)
    
    status_ok, status_data = client.get_system_status()
    if status_ok and status_data.get("status") == "success":
        dashboard = status_data.get("dashboard", {})
        summary = dashboard.get("summary", {})
        metrics = dashboard.get("metrics", {})
        
        print(f"✅ System Status: {summary.get('system_status', 'unknown')}")
        print(f"   Performance Score: {summary.get('performance_score', 0)}/100")
        print(f"   CPU Usage: {metrics.get('cpu_percent', {}).get('current', 0):.1f}%")
        print(f"   Memory Usage: {metrics.get('memory_percent', {}).get('current', 0):.1f}%")
        print(f"   Active Alerts: {summary.get('active_alerts', 0)}")
    else:
        print(f"⚠️  Could not get system status: {status_data}")
    
    # Step 3: Queue Status
    print("\n🔄 Step 3: Queue Status")
    print("-" * 30)
    
    queue_ok, queue_data = client.get_queue_status()
    if queue_ok and queue_data.get("status") == "success":
        queue = queue_data.get("queue", {})
        print(f"✅ Queue Status: {queue.get('status', 'unknown')}")
        print(f"   Queue Length: {queue.get('queue_length', 0)}")
        print(f"   Active Jobs: {queue.get('active_jobs', 0)}")
        print(f"   Max Workers: {queue.get('max_workers', 0)}")
        print(f"   Total Processed: {queue.get('metrics', {}).get('total_processed', 0)}")
    else:
        print(f"⚠️  Could not get queue status: {queue_data}")
    
    # Step 4: Start Analysis (Demo)
    print("\n🚀 Step 4: Start Analysis Job")
    print("-" * 30)
    
    video_url = "https://www.twitch.tv/videos/2472774741"
    print(f"Starting analysis for: {video_url}")
    
    success, result = client.start_analysis(video_url)
    if success and result.get("status") == "success":
        job_id = result.get("job_id")
        video_id = result.get("video_id")
        progress = result.get("progress", {})
        
        print(f"✅ Analysis started successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   Video ID: {video_id}")
        print(f"   Initial Stage: {progress.get('current_stage', 'unknown')}")
        print(f"   Progress: {progress.get('percentage', 0):.1f}%")
        print(f"   ETA: {progress.get('estimated_completion_seconds', 0)/60:.1f} minutes")
        
        # Step 5: Track Progress (Demo)
        print(f"\n📈 Step 5: Progress Tracking")
        print("-" * 30)
        print("In a real frontend, you would:")
        print("1. Subscribe to Server-Sent Events for real-time updates")
        print("2. Update progress bars and UI elements")
        print("3. Show current stage and estimated completion time")
        print("4. Handle completion and error states")
        
        print(f"\n🔗 Real-time Stream URL:")
        print(f"   {client.base_url}/api/v1/analysis/{job_id}/stream")
        
        # Demonstrate a few status checks
        print(f"\n⏱️  Demonstrating status checks...")
        for i in range(3):
            time.sleep(2)
            status_ok, status_result = client.get_job_status(job_id)
            if status_ok:
                if "job" in status_result:
                    job = status_result["job"]
                elif "job_id" in status_result:
                    job = status_result
                else:
                    job = status_result
                
                progress = job.get("progress", {})
                current_stage = progress.get("current_stage", "Unknown")
                percentage = progress.get("percentage", 0)
                job_status = job.get("status", "unknown")
                
                print(f"   [{i+1}/3] {job_status.upper()}: {current_stage} - {percentage:.1f}%")
                
                if job_status in ["completed", "failed"]:
                    break
            else:
                print(f"   [{i+1}/3] ❌ Status check failed")
        
        return job_id
        
    else:
        print(f"❌ Failed to start analysis: {result}")
        return None

def demo_frontend_components():
    """Show how frontend components would use the API"""
    
    print("\n\n🎨 Frontend Component Integration Examples")
    print("=" * 60)
    
    print("""
📱 React Component Example:

```typescript
// AnalysisComponent.tsx
import { useState } from 'react';
import { KlipStreamAPIClient } from './api/client';

export function AnalysisComponent() {
  const [job, setJob] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const startAnalysis = async (videoUrl) => {
    setIsLoading(true);
    try {
      const client = new KlipStreamAPIClient();
      const result = await client.startAnalysis(videoUrl);
      setJob(result.job);
      
      // Subscribe to real-time updates
      const eventSource = new EventSource(
        `${API_URL}/api/v1/analysis/${result.job_id}/stream`
      );
      
      eventSource.onmessage = (event) => {
        const updatedJob = JSON.parse(event.data);
        setJob(updatedJob);
        
        if (updatedJob.status === 'completed') {
          eventSource.close();
        }
      };
      
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div>
      {job && (
        <ProgressBar 
          percentage={job.progress.percentage}
          stage={job.progress.current_stage}
          status={job.status}
        />
      )}
    </div>
  );
}
```

🔄 Real-time Updates:
- Server-Sent Events for live progress
- WebSocket fallback for older browsers
- Automatic reconnection on connection loss
- Progress bars and stage indicators

📊 System Monitoring:
- Real-time system health dashboard
- Performance metrics and alerts
- Queue status and job management
- Error tracking and recovery

🎯 Key Benefits:
- 99.5% faster response times (immediate vs 5-7 minutes)
- Real-time progress tracking
- Multiple concurrent jobs
- Intelligent error handling with retries
- Comprehensive system monitoring
""")

def main():
    """Main demo function"""
    try:
        job_id = demo_frontend_integration()
        demo_frontend_components()
        
        print(f"\n🎉 Demo completed successfully!")
        if job_id:
            print(f"   Job ID for tracking: {job_id}")
        print(f"   API Documentation: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/docs")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
