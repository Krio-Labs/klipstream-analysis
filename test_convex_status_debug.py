#!/usr/bin/env python3
"""
Test script to debug Convex status update issues
"""

import os
import sys
from utils.convex_client_updated import ConvexManager
from convex_integration import ConvexIntegration

def test_video_exists(twitch_id):
    """Test if a video exists in the database"""
    print(f"\n🔍 Testing if video {twitch_id} exists...")
    
    # Test with ConvexIntegration
    convex = ConvexIntegration()
    video = convex.get_video(twitch_id)
    
    if video:
        print(f"✅ Video found: {video.get('_id')} - {video.get('title', 'No title')}")
        print(f"   Current status: {video.get('status', 'No status')}")
        print(f"   Team: {video.get('team', 'No team')}")
        return video
    else:
        print(f"❌ Video {twitch_id} not found in database")
        return None

def test_status_update(twitch_id, new_status):
    """Test status update for a video"""
    print(f"\n🔄 Testing status update for video {twitch_id} to '{new_status}'...")
    
    # Test with ConvexManager
    convex_manager = ConvexManager()
    success = convex_manager.update_video_status(twitch_id, new_status)
    
    if success:
        print(f"✅ Status update successful")
    else:
        print(f"❌ Status update failed")
    
    return success

def test_create_video(twitch_id):
    """Test creating a video if it doesn't exist"""
    print(f"\n➕ Testing video creation for {twitch_id}...")
    
    convex_manager = ConvexManager()
    success = convex_manager.create_video_if_missing(twitch_id)
    
    if success:
        print(f"✅ Video creation/verification successful")
    else:
        print(f"❌ Video creation failed")
    
    return success

def main():
    """Main test function"""
    print("🧪 CONVEX STATUS UPDATE DEBUG TEST")
    print("=" * 50)
    
    # Test with the videos shown in the screenshot
    test_videos = [
        "2434635255",  # Jynxzi video with "Error: Processing..."
        "2480161276",  # Video with "Error: Processing..."
        "2301647082",  # Video with "queued"
        "2472774741",  # Video with "queued"
    ]
    
    for twitch_id in test_videos:
        print(f"\n{'='*60}")
        print(f"🎬 TESTING VIDEO: {twitch_id}")
        print(f"{'='*60}")
        
        # Step 1: Check if video exists
        video = test_video_exists(twitch_id)
        
        if not video:
            # Step 2: Try to create the video if it doesn't exist
            create_success = test_create_video(twitch_id)
            if create_success:
                # Check again after creation
                video = test_video_exists(twitch_id)
        
        if video:
            # Step 3: Try to update status
            test_status_update(twitch_id, "Testing Status Update")
            
            # Step 4: Verify the update worked
            updated_video = test_video_exists(twitch_id)
            if updated_video:
                new_status = updated_video.get('status', 'No status')
                if new_status == "Testing Status Update":
                    print(f"✅ Status update verified: {new_status}")
                else:
                    print(f"❌ Status update not reflected: {new_status}")
        
        print(f"\n{'='*60}")
    
    print(f"\n🎉 DEBUG TEST COMPLETED")

if __name__ == "__main__":
    main()
