#!/usr/bin/env python3
"""
Debug script to understand why the queue isn't processing
"""

import requests
import json
from datetime import datetime

def check_api_endpoints():
    """Check various API endpoints to understand the system state"""
    
    print("🔍 DEBUGGING QUEUE PROCESSING ISSUE")
    print("=" * 60)
    print(f"⏰ Time: {datetime.now().isoformat()}")
    print()
    
    base_url = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"
    
    # 1. Check API health
    print("1️⃣ API Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ API is healthy")
        else:
            print(f"   ❌ API health issue: {response.text}")
    except Exception as e:
        print(f"   ❌ Health check failed: {str(e)}")
    
    print()
    
    # 2. Check transcription methods
    print("2️⃣ Transcription Methods")
    try:
        response = requests.get(f"{base_url}/api/v1/transcription/methods", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            methods = response.json()
            print(f"   ✅ Available methods: {methods}")
        else:
            print(f"   ❌ Methods check failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Methods check error: {str(e)}")
    
    print()
    
    # 3. Check our specific job status
    print("3️⃣ Our Job Status")
    job_id = "59c38070-f870-48bf-b07b-2cdde41a25f3"
    try:
        response = requests.get(f"{base_url}/api/v1/analysis/{job_id}/status", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            status = response.json()
            print(f"   📊 Job Status: {status.get('status')}")
            progress = status.get('progress', {})
            if isinstance(progress, dict):
                print(f"   📈 Progress: {progress.get('percentage', 0)}%")
                print(f"   🎯 Stage: {progress.get('current_stage', 'unknown')}")
            print(f"   🕐 Created: {status.get('created_at')}")
            print(f"   🕐 Updated: {status.get('updated_at')}")
        else:
            print(f"   ❌ Status check failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Status check error: {str(e)}")
    
    print()
    
    # 4. Try to submit a new job to see if it triggers processing
    print("4️⃣ Test New Job Submission")
    try:
        test_payload = {"url": "https://www.twitch.tv/videos/2480161276"}
        response = requests.post(
            f"{base_url}/api/v1/analysis",
            json=test_payload,
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ New job created: {data.get('job_id')}")
            print(f"   📊 Status: {data.get('status')}")
            print(f"   💬 Message: {data.get('message')}")
        else:
            print(f"   ❌ Job submission failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Job submission error: {str(e)}")
    
    print()

def analyze_issue():
    """Analyze the potential root causes"""
    
    print("🔬 ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    print("Based on the codebase analysis, here are the potential issues:")
    print()
    
    print("📋 QUEUE PROCESSING FLOW:")
    print("1. Job submitted → addVideoToQueue mutation")
    print("2. If queuePosition === 0 → scheduler.runAfter(2000ms, processQueuedVideo)")
    print("3. processQueuedVideo action → processNextVideo mutation")
    print("4. processNextVideo → updates job status to 'processing'")
    print("5. processNextVideo → calls Cloud Run API")
    print()
    
    print("🚨 POTENTIAL ISSUES:")
    print()
    
    print("A. CONVEX SCHEDULER ISSUE (Most Likely)")
    print("   - The scheduler.runAfter() call might be failing silently")
    print("   - Convex scheduled functions might not be executing")
    print("   - The 2-second delay might not be triggering")
    print("   - Likelihood: 70%")
    print()
    
    print("B. PROCESSING JOBS BLOCKING QUEUE")
    print("   - Another job might be stuck in 'processing' status")
    print("   - processNextVideo checks for existing processing jobs")
    print("   - If found, it skips queue processing")
    print("   - Likelihood: 20%")
    print()
    
    print("C. DATABASE CONNECTION ISSUES")
    print("   - Convex database queries might be failing")
    print("   - Job might not be properly saved to database")
    print("   - Likelihood: 5%")
    print()
    
    print("D. CRON JOBS NOT RUNNING")
    print("   - maintainQueue cron (every 5 min) should catch stuck jobs")
    print("   - checkStuckVideos cron (every 15 min) should help")
    print("   - But these are disabled/not working")
    print("   - Likelihood: 5%")
    print()
    
    print("🛠️ SOLUTIONS TO TRY:")
    print()
    print("1. MANUAL TRIGGER (Immediate)")
    print("   - Call processQueuedVideo action directly via Convex")
    print("   - Submit another job to trigger queue processing")
    print()
    
    print("2. CHECK CONVEX LOGS (Diagnostic)")
    print("   - Look at Convex dashboard for scheduled function logs")
    print("   - Check if processQueuedVideo action is being called")
    print()
    
    print("3. ENABLE CRON FALLBACK (Preventive)")
    print("   - Re-enable the processQueuedVideo cron job")
    print("   - Set it to run every 30 seconds as backup")
    print()
    
    print("4. ADD MANUAL TRIGGER ENDPOINT (Workaround)")
    print("   - Create API endpoint to manually trigger queue processing")
    print("   - Allow manual intervention when scheduler fails")

if __name__ == "__main__":
    check_api_endpoints()
    analyze_issue()
