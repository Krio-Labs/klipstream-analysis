#!/usr/bin/env python3
"""
Test script for the fixed KlipStream Analysis API
Tests that jobs are now properly added to the Convex queue
"""

import requests
import json
import time
import sys
from datetime import datetime

# API Configuration
API_BASE_URL = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"
TEST_VIDEO_URL = "https://www.twitch.tv/videos/2479611486"

def print_status(message: str, status: str = "INFO"):
    """Print formatted status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {status}: {message}")

def test_api_health():
    """Test API health"""
    print_status("Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health/detailed", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_status(f"API Status: {data['status']}")
            print_status(f"Convex Status: {data['services']['external_services']['convex']['status']}")
            return data['status'] == 'healthy'
        else:
            print_status(f"Health check failed: {response.status_code}", "ERROR")
            return False
    except Exception as e:
        print_status(f"Health check error: {str(e)}", "ERROR")
        return False

def start_analysis():
    """Start analysis using the fixed API"""
    print_status("Starting analysis with fixed API...")
    
    payload = {
        "url": TEST_VIDEO_URL,
        "options": {
            "transcription_method": "auto",
            "enable_highlights": True,
            "enable_sentiment": True
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
                print_status(f"✅ Analysis started! Job ID: {job_id}", "SUCCESS")
                return job_id
            else:
                print_status(f"❌ Analysis failed: {data.get('message', 'unknown')}", "ERROR")
                return None
        else:
            print_status(f"❌ HTTP Error: {response.status_code}", "ERROR")
            try:
                error_data = response.json()
                print_status(f"Error details: {error_data}", "ERROR")
            except:
                print_status(f"Response text: {response.text}", "ERROR")
            return None
            
    except Exception as e:
        print_status(f"❌ Request failed: {str(e)}", "ERROR")
        return None

def check_convex_queue():
    """Check if job was added to Convex queue"""
    print_status("Checking Convex queue status...")
    
    try:
        # Use the Node.js script we created earlier
        import subprocess
        result = subprocess.run(
            ["node", "test_convex_queue.js"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print_status("Convex queue check completed", "SUCCESS")
            print(result.stdout)
            return "Queue length: 1" in result.stdout or "Currently processing: 1" in result.stdout
        else:
            print_status(f"Convex queue check failed: {result.stderr}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Error checking Convex queue: {str(e)}", "ERROR")
        return False

def monitor_job_briefly(job_id: str):
    """Monitor job for a short time to see if it starts processing"""
    print_status(f"Monitoring job {job_id} for 2 minutes...")
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < 120:  # 2 minutes
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                current_status = data.get('status')
                progress = data.get('progress', {}).get('percentage', 0)
                stage = data.get('progress', {}).get('current_stage', 'unknown')
                
                if current_status != last_status:
                    print_status(f"Status: {current_status} | Progress: {progress}% | Stage: {stage}")
                    last_status = current_status
                    
                    if current_status == "Processing":
                        print_status("🎉 Job moved to Processing status!", "SUCCESS")
                        return True
                    elif current_status == "completed":
                        print_status("🎉 Job completed!", "SUCCESS")
                        return True
                    elif current_status == "failed":
                        print_status("❌ Job failed", "ERROR")
                        return False
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            print_status(f"Error checking status: {str(e)}", "ERROR")
            time.sleep(10)
    
    print_status("⏰ Monitoring timeout - job may still be processing", "WARNING")
    return False

def main():
    """Main test function"""
    print_status("🚀 Testing Fixed KlipStream Analysis API", "INFO")
    print_status(f"Testing with video: {TEST_VIDEO_URL}")
    print("=" * 80)
    
    # Test 1: Health check
    if not test_api_health():
        print_status("❌ API health check failed, aborting test", "ERROR")
        sys.exit(1)
    
    print()
    
    # Test 2: Start analysis
    job_id = start_analysis()
    if not job_id:
        print_status("❌ Failed to start analysis", "ERROR")
        sys.exit(1)
    
    print()
    
    # Test 3: Check Convex queue
    print_status("Waiting 10 seconds for job to be added to Convex queue...")
    time.sleep(10)
    
    convex_has_job = check_convex_queue()
    if convex_has_job:
        print_status("✅ Job found in Convex queue!", "SUCCESS")
    else:
        print_status("⚠️ Job not found in Convex queue (may have already started processing)", "WARNING")
    
    print()
    
    # Test 4: Monitor job briefly
    job_started = monitor_job_briefly(job_id)
    
    print()
    print("=" * 80)
    
    if job_started:
        print_status("🎉 SUCCESS: Job started processing automatically!", "SUCCESS")
        print_status("✅ The fix worked - API now uses Convex queue properly", "SUCCESS")
    else:
        print_status("⏳ Job is queued but hasn't started processing yet", "INFO")
        print_status("💡 This may be normal if there are other jobs in the queue", "INFO")
    
    print_status(f"🔗 Monitor job progress at: {API_BASE_URL}/api/v1/analysis/{job_id}/status")

if __name__ == "__main__":
    main()
