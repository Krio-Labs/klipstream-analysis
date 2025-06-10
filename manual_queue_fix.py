#!/usr/bin/env python3
"""
Manually add the stuck job to Convex queue as a workaround
"""

import sys
import os
sys.path.append('.')

def manual_fix_stuck_job():
    """Manually add the stuck job to Convex queue"""
    print("🔧 Manual Fix: Adding Stuck Job to Convex Queue")
    print("=" * 60)
    
    try:
        # Initialize Convex
        from utils.convex_client_updated import ConvexManager
        convex_manager = ConvexManager()
        
        if not convex_manager.convex:
            print("❌ ConvexManager not initialized")
            return False
        
        # Job details from the test
        stuck_job_id = "1407db01-b05a-4e7d-992f-ea129a4a8f1b"
        video_id = "2479611486"
        team_id = "js7bj9zgdkyj9ykvr4m6jarxkh7ep9fa"
        
        print(f"🎯 Target job: {stuck_job_id}")
        print(f"📹 Video ID: {video_id}")
        print(f"👥 Team ID: {team_id}")
        
        # Step 1: Find or create video in Convex
        print("\n1️⃣ Looking for video in Convex...")
        existing_video = convex_manager.convex.client.query("video:getAnyByTwitchId", {
            "twitch_id": video_id
        })
        
        if existing_video:
            print(f"✅ Found video in Convex")
            print(f"🔍 Raw response: {existing_video}")

            # Handle Convex response structure
            video_data = existing_video
            if 'value' in existing_video:
                video_data = existing_video['value']

            # Get the actual video ID (handle different field names)
            convex_video_id = video_data.get('_id') or video_data.get('id')
            if not convex_video_id:
                print("❌ Video found but no ID field")
                print(f"Available fields in video_data: {list(video_data.keys()) if isinstance(video_data, dict) else 'Not a dict'}")
                return False

            print(f"📄 Video ID: {convex_video_id}")
            print(f"📝 Title: {video_data.get('title', 'No title')}")

            # Use the video's team ID if available
            video_team_id = video_data.get('team')
            if video_team_id:
                team_id = video_team_id
                print(f"👥 Using video's team ID: {team_id}")
        else:
            print("❌ Video not found in Convex")
            print("💡 Need to create video first")
            return False
        
        # Step 2: Add job to Convex queue
        print(f"\n2️⃣ Adding job {stuck_job_id} to Convex queue...")
        
        try:
            queue_result = convex_manager.convex.client.mutation("queueManager:addVideoToQueue", {
                "videoId": convex_video_id,
                "teamId": team_id,
                "jobId": stuck_job_id,
                "priority": 1  # High priority for manual fix
            })
            
            if queue_result.get('success'):
                print(f"✅ Successfully added job to Convex queue!")
                print(f"📊 Queue position: {queue_result.get('queuePosition', 'unknown')}")
                print(f"⏱️ Estimated wait: {queue_result.get('estimatedWaitTime', 'unknown')} ms")
                
                # Step 3: Check queue status
                print(f"\n3️⃣ Checking queue status...")
                queue_status = convex_manager.convex.client.query("queueManager:getQueueStatus", {})
                print(f"📊 Queue length: {queue_status.get('queueLength', 0)}")
                print(f"⚡ Processing count: {queue_status.get('processingCount', 0)}")
                
                return True
            else:
                print(f"❌ Failed to add job to queue: {queue_result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Error adding job to queue: {str(e)}")
            return False
        
    except Exception as e:
        print(f"❌ Error in manual fix: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_api_job_status():
    """Check the status of the stuck job in the API"""
    print("\n🔍 Checking API Job Status")
    print("=" * 40)
    
    import requests
    
    stuck_job_id = "1407db01-b05a-4e7d-992f-ea129a4a8f1b"
    api_url = f"https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis/{stuck_job_id}/status"
    
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Status: {data.get('status', 'unknown')}")
            print(f"📈 Progress: {data.get('progress', {}).get('percentage', 0)}%")
            print(f"🎯 Stage: {data.get('progress', {}).get('current_stage', 'unknown')}")
            return data.get('status') != 'Queued'
        else:
            print(f"❌ API request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking API status: {str(e)}")
        return False

def main():
    """Run manual fix"""
    print("🚀 Manual Queue Fix for Stuck Job")
    print("This will manually add the stuck job to Convex queue")
    print("=" * 60)
    
    # Check current API status
    api_changed = check_api_job_status()
    if api_changed:
        print("✅ Job is no longer stuck in API - fix may have worked!")
        return True
    
    # Try manual fix
    success = manual_fix_stuck_job()
    
    if success:
        print("\n🎉 Manual fix completed!")
        print("💡 The job should start processing automatically")
        print("🔗 Monitor at: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis/1407db01-b05a-4e7d-992f-ea129a4a8f1b/status")
    else:
        print("\n❌ Manual fix failed")
        print("💡 Wait for deployment to complete")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
