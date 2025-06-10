#!/usr/bin/env python3
"""
Test script for KlipStream Analysis Cloud API
Tests the API with a specific Twitch video URL and monitors progress
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# API Configuration
API_BASE_URL = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"
TEST_VIDEO_URL = "https://www.twitch.tv/videos/2479611486"
USER_ID = "jh7en1zt0460hfzf2qa470p1k17epm40"
TEAM_ID = "js7bj9zgdkyj9ykvr4m6jarxkh7ep9fa"

def print_status(message: str, status: str = "INFO"):
    """Print formatted status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {status}: {message}")

def test_health_check() -> bool:
    """Test API health check endpoint"""
    print_status("Testing API health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_status(f"API is healthy - Version: {data.get('version', 'unknown')}", "SUCCESS")
            return True
        else:
            print_status(f"Health check failed with status {response.status_code}", "ERROR")
            return False
    except Exception as e:
        print_status(f"Health check failed: {str(e)}", "ERROR")
        return False

def test_transcription_methods() -> bool:
    """Test transcription methods endpoint"""
    print_status("Checking available transcription methods...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/transcription/methods", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_status("Available transcription methods:", "SUCCESS")
            for method, details in data.get('methods', {}).items():
                print(f"  - {method}: {details.get('name', 'Unknown')}")
                print(f"    Cost: {details.get('cost_per_hour', 'Unknown')}")
                print(f"    GPU Required: {details.get('gpu_required', 'Unknown')}")
            return True
        else:
            print_status(f"Transcription methods check failed with status {response.status_code}", "ERROR")
            return False
    except Exception as e:
        print_status(f"Transcription methods check failed: {str(e)}", "ERROR")
        return False

def start_analysis() -> Optional[str]:
    """Start video analysis and return job ID"""
    print_status(f"Starting analysis for video: {TEST_VIDEO_URL}")
    
    payload = {
        "url": TEST_VIDEO_URL,
        "user_id": USER_ID,
        "team_id": TEAM_ID,
        "options": {
            "transcription_method": "auto",
            "enable_highlights": True,
            "enable_sentiment": True,
            "chunk_duration": 10
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/analysis",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                job_id = data.get('job_id')
                print_status(f"Analysis started successfully! Job ID: {job_id}", "SUCCESS")
                print_status(f"Estimated duration: {data.get('estimated_duration', 'unknown')}")
                return job_id
            else:
                print_status(f"Analysis failed to start: {data.get('error', 'unknown error')}", "ERROR")
                return None
        else:
            print_status(f"Analysis request failed with status {response.status_code}", "ERROR")
            try:
                error_data = response.json()
                print_status(f"Error details: {error_data.get('error', 'No details available')}", "ERROR")
            except:
                print_status(f"Response text: {response.text}", "ERROR")
            return None
            
    except Exception as e:
        print_status(f"Analysis request failed: {str(e)}", "ERROR")
        return None

def get_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get current status of analysis job"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print_status(f"Status check failed with status {response.status_code}", "ERROR")
            return None
    except Exception as e:
        print_status(f"Status check failed: {str(e)}", "ERROR")
        return None

def monitor_progress(job_id: str, max_wait_minutes: int = 60) -> bool:
    """Monitor analysis progress with polling"""
    print_status(f"Monitoring progress for job {job_id}...")
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    last_progress = -1
    last_stage = ""
    
    while time.time() - start_time < max_wait_seconds:
        status_data = get_status(job_id)
        if not status_data:
            time.sleep(10)
            continue
            
        current_status = status_data.get('status')
        progress = status_data.get('progress', 0)
        current_stage = status_data.get('current_stage', 'unknown')
        
        # Print progress updates
        if progress != last_progress or current_stage != last_stage:
            print_status(f"Progress: {progress}% - Stage: {current_stage}")
            last_progress = progress
            last_stage = current_stage
            
            # Print stage details
            stages = status_data.get('stages', {})
            stage_summary = []
            for stage, stage_status in stages.items():
                if stage_status == 'completed':
                    stage_summary.append(f"✓ {stage}")
                elif stage_status == 'processing':
                    stage_summary.append(f"⏳ {stage}")
                elif stage_status == 'failed':
                    stage_summary.append(f"✗ {stage}")
                else:
                    stage_summary.append(f"⏸ {stage}")
            
            if stage_summary:
                print(f"    Stages: {' | '.join(stage_summary)}")
        
        # Check if completed
        if current_status == 'completed':
            print_status("Analysis completed successfully!", "SUCCESS")
            
            # Print results
            results = status_data.get('results', {})
            if results:
                print_status("Results available:")
                for key, url in results.items():
                    if url and key.endswith('_url'):
                        print(f"  - {key}: {url}")
                
                # Print highlights summary
                highlights = results.get('highlights', [])
                if highlights:
                    print_status(f"Found {len(highlights)} highlights:")
                    for i, highlight in enumerate(highlights[:5]):  # Show first 5
                        start_min = int(highlight['start_time'] // 60)
                        start_sec = int(highlight['start_time'] % 60)
                        print(f"  {i+1}. {highlight['type']} at {start_min}:{start_sec:02d} (score: {highlight['score']:.2f})")
                    if len(highlights) > 5:
                        print(f"  ... and {len(highlights) - 5} more")
            
            return True
            
        elif current_status == 'failed':
            error = status_data.get('error', 'Unknown error')
            print_status(f"Analysis failed: {error}", "ERROR")
            return False
            
        # Wait before next check
        time.sleep(10)
    
    print_status(f"Analysis timed out after {max_wait_minutes} minutes", "ERROR")
    return False

def main():
    """Main test function"""
    print_status("Starting KlipStream Analysis API Test", "INFO")
    print_status(f"Testing with video: {TEST_VIDEO_URL}")
    print("=" * 80)
    
    # Test health check
    if not test_health_check():
        print_status("Health check failed, aborting test", "ERROR")
        sys.exit(1)
    
    print()
    
    # Test transcription methods
    if not test_transcription_methods():
        print_status("Transcription methods check failed, continuing anyway", "WARNING")
    
    print()
    
    # Start analysis
    job_id = start_analysis()
    if not job_id:
        print_status("Failed to start analysis, aborting test", "ERROR")
        sys.exit(1)
    
    print()
    
    # Monitor progress
    success = monitor_progress(job_id, max_wait_minutes=60)
    
    print()
    print("=" * 80)
    if success:
        print_status("API test completed successfully!", "SUCCESS")
        sys.exit(0)
    else:
        print_status("API test failed", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
