#!/usr/bin/env python3
"""
Quick fix to test API integration with Convex queue
This script demonstrates how the API should work with the Convex queue
"""

import requests
import json
from utils.convex_client_updated import ConvexManager

def test_convex_integration():
    """Test creating a video and adding it to Convex queue"""
    
    # Initialize Convex manager
    convex_manager = ConvexManager()
    
    if not convex_manager.convex:
        print("❌ Convex client not initialized")
        return False
    
    # Test video details
    twitch_id = "2479611486"
    team_id = "js7bj9zgdkyj9ykvr4m6jarxkh7ep9fa"
    user_id = "jh7en1zt0460hfzf2qa470p1k17epm40"
    
    print(f"🧪 Testing Convex integration for video {twitch_id}")
    
    try:
        # Step 1: Check if video exists
        print("1️⃣ Checking if video exists in Convex...")
        existing_video = convex_manager.convex.get_video(twitch_id)
        
        if existing_video:
            print(f"✅ Video exists: {existing_video.get('_id')} - {existing_video.get('title', 'No title')}")
            video_id = existing_video['_id']
        else:
            print("❌ Video not found in Convex")
            print("💡 The API should create the video first, then add to queue")
            return False
        
        # Step 2: Try to add to Convex queue using the API
        print("\n2️⃣ Adding job to Convex queue via API...")
        
        # Use the Convex mutation directly (simulating what the API should do)
        job_id = f"api-fix-{int(time.time())}-test"
        
        # This is what the API should do instead of using internal queue
        result = convex_manager.convex.client.mutation("queueManager:addVideoToQueue", {
            "videoId": video_id,
            "teamId": team_id,
            "jobId": job_id,
            "priority": 0
        })
        
        print(f"✅ Added to Convex queue: {result}")
        
        # Step 3: Check queue status
        print("\n3️⃣ Checking Convex queue status...")
        queue_status = convex_manager.convex.client.query("queueManager:getQueueStatus", {})
        print(f"📊 Queue length: {queue_status['queueLength']}")
        print(f"📊 Processing count: {queue_status['processingCount']}")
        
        if queue_status['queueLength'] > 0:
            print("✅ Job successfully added to Convex queue!")
            
            # Try to trigger processing
            print("\n4️⃣ Triggering queue processing...")
            try:
                process_result = convex_manager.convex.client.mutation("processor:processQueuedVideo", {})
                print(f"🚀 Process trigger result: {process_result}")
            except Exception as e:
                print(f"⚠️ Could not trigger processing: {str(e)}")
            
            return True
        else:
            print("❌ Job not found in queue")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def create_fixed_api_endpoint():
    """Show how the API endpoint should be modified"""
    
    print("\n" + "="*60)
    print("🔧 HOW TO FIX THE API ENDPOINT")
    print("="*60)
    
    print("""
The current API endpoint in api/routes/analysis.py should be modified:

CURRENT (BROKEN):
```python
# Create analysis job in internal queue
analysis_job = AnalysisJob(...)
await job_manager.create_job(analysis_job)  # ❌ Internal queue
```

FIXED (WORKING):
```python
# 1. Create/find video in Convex
convex_manager = ConvexManager()
video = convex_manager.convex.get_video(video_id)
if not video:
    # Create video entry first
    video = convex_manager.convex.create_video_minimal(video_id, "Queued")

# 2. Add to Convex queue (not internal queue)
result = convex_manager.convex.client.mutation("queueManager:addVideoToQueue", {
    "videoId": video['_id'],
    "teamId": team_id,
    "jobId": job_id,
    "priority": 0
})

# 3. Return response (Convex scheduler will handle processing)
```

This way:
✅ Jobs go to Convex queue (where frontend expects them)
✅ Convex scheduler automatically triggers processing
✅ No need for internal queue manager
✅ Frontend and API use same queue system
""")

if __name__ == "__main__":
    import time
    
    print("🚀 Testing API-Convex Queue Integration Fix")
    print("="*60)
    
    success = test_convex_integration()
    
    if success:
        print("\n🎉 SUCCESS: Convex queue integration working!")
        print("💡 The API should be modified to use this approach")
    else:
        print("\n❌ FAILED: Issues with Convex queue integration")
    
    create_fixed_api_endpoint()
