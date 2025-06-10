#!/usr/bin/env python3
"""
Quick test to see if our fix is deployed
"""

import requests
import json
import time
from datetime import datetime

def test_new_job():
    """Test creating a new job to see if it goes to Convex queue"""
    print("🧪 Testing if API fix is deployed")
    print("=" * 50)
    
    api_url = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis"
    
    payload = {
        "url": "https://www.twitch.tv/videos/2479611486",
        "options": {
            "transcription_method": "auto"
        }
    }
    
    try:
        print("🚀 Creating new analysis job...")
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            job_id = data.get('job_id')
            print(f"✅ Job created: {job_id}")
            
            # Wait a moment then check status
            time.sleep(5)
            
            status_url = f"https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis/{job_id}/status"
            status_response = requests.get(status_url, timeout=10)
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"📊 Status: {status_data.get('status')}")
                print(f"📈 Progress: {status_data.get('progress', {}).get('percentage', 0)}%")
                
                # If it's still queued, the fix might not be deployed yet
                if status_data.get('status') == 'Queued':
                    print("⚠️ Job still in 'Queued' status - fix may not be deployed yet")
                    return False
                else:
                    print("✅ Job status changed - fix appears to be working!")
                    return True
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                return False
        else:
            print(f"❌ Job creation failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data}")
            except:
                print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def check_convex_queue_simple():
    """Simple check of Convex queue using Node.js"""
    print("\n🔍 Checking Convex queue...")
    
    try:
        import subprocess
        result = subprocess.run(
            ["node", "-e", """
const { ConvexHttpClient } = require("convex/browser");
const client = new ConvexHttpClient("https://laudable-horse-446.convex.cloud");

client.query("queueManager:getQueueStatus", {})
  .then(status => {
    console.log(`Queue length: ${status.queueLength}`);
    console.log(`Processing: ${status.processingCount}`);
    if (status.queueLength > 0) {
      console.log("✅ Jobs found in Convex queue!");
    } else {
      console.log("📭 No jobs in Convex queue");
    }
  })
  .catch(err => console.log(`Error: ${err.message}`));
            """],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return "Jobs found in Convex queue" in result.stdout
        else:
            print(f"❌ Queue check failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking queue: {str(e)}")
        return False

def main():
    """Main test"""
    print("🚀 Quick Fix Deployment Test")
    print("Testing if our Convex integration fix is deployed")
    print("=" * 60)
    
    # Test 1: Create new job
    job_working = test_new_job()
    
    # Test 2: Check Convex queue
    queue_has_jobs = check_convex_queue_simple()
    
    print("\n" + "=" * 60)
    
    if job_working or queue_has_jobs:
        print("🎉 SUCCESS: Fix appears to be working!")
        if queue_has_jobs:
            print("✅ Jobs found in Convex queue")
        if job_working:
            print("✅ Job status changed from Queued")
    else:
        print("⏳ Fix not yet deployed or not working")
        print("💡 Wait for deployment to complete")
    
    return job_working or queue_has_jobs

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
