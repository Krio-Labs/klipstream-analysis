#!/usr/bin/env python3
"""
Fix the videos that are stuck in "queued" and "Error: Processing..." states
"""

import os
import sys
from utils.convex_client_updated import ConvexManager
from convex_integration import ConvexIntegration

def fix_stuck_video(twitch_id, new_status="Ready for retry"):
    """Fix a stuck video by updating its status"""
    print(f"\n🔧 Fixing stuck video {twitch_id}...")
    
    # Get current status first
    convex = ConvexIntegration()
    video = convex.get_video(twitch_id)
    
    if video:
        current_status = video.get('status', 'No status')
        print(f"   Current status: {current_status}")
        
        # Update to new status
        convex_manager = ConvexManager()
        success = convex_manager.update_video_status(twitch_id, new_status)
        
        if success:
            print(f"✅ Successfully updated status to '{new_status}'")
            
            # Verify the change
            updated_video = convex.get_video(twitch_id)
            if updated_video:
                updated_status = updated_video.get('status', 'No status')
                print(f"   Verified new status: {updated_status}")
                return True
        else:
            print(f"❌ Failed to update status")
            return False
    else:
        print(f"❌ Video not found")
        return False

def main():
    """Main function to fix all stuck videos"""
    print("🚀 FIXING STUCK VIDEOS IN CONVEX DATABASE")
    print("=" * 50)
    
    # Videos from the screenshot that are stuck
    stuck_videos = [
        {
            "twitch_id": "2434635255",
            "title": "Jynxzi - Spending $10K on R6 Skins",
            "current_status": "Error: Processing timeout"
        },
        {
            "twitch_id": "2480161276", 
            "title": "Twitch VOD 2480161276",
            "current_status": "Error: Processing timeout"
        },
        {
            "twitch_id": "2301647082",
            "title": "ADAPT ! overcome ! ONLY way peaker place all time",
            "current_status": "queued"
        },
        {
            "twitch_id": "2472774741",
            "title": "GAMING TIME TO GET CONTENT",
            "current_status": "queued"
        }
    ]
    
    print(f"\n📋 Found {len(stuck_videos)} stuck videos to fix:")
    for video in stuck_videos:
        print(f"   • {video['twitch_id']}: {video['current_status']}")
    
    print(f"\n🔧 FIXING VIDEOS...")
    print("=" * 40)
    
    fixed_count = 0
    for video in stuck_videos:
        twitch_id = video['twitch_id']
        title = video['title'][:50] + "..." if len(video['title']) > 50 else video['title']
        
        print(f"\n🎬 FIXING: {twitch_id}")
        print(f"   Title: {title}")
        
        success = fix_stuck_video(twitch_id, "Ready for retry")
        if success:
            fixed_count += 1
    
    print(f"\n🎉 SUMMARY")
    print("=" * 20)
    print(f"Videos processed: {len(stuck_videos)}")
    print(f"Videos fixed: {fixed_count}")
    print(f"Success rate: {(fixed_count/len(stuck_videos)*100):.1f}%")
    
    if fixed_count == len(stuck_videos):
        print(f"✅ ALL VIDEOS SUCCESSFULLY FIXED!")
        print(f"💡 Videos are now set to 'Ready for retry' and can be reprocessed")
    else:
        print(f"⚠️  Some videos could not be fixed")
    
    print(f"\n🔍 FINAL STATUS CHECK")
    print("=" * 25)
    convex = ConvexIntegration()
    for video in stuck_videos:
        twitch_id = video['twitch_id']
        current_video = convex.get_video(twitch_id)
        if current_video:
            final_status = current_video.get('status', 'No status')
            print(f"   {twitch_id}: {final_status}")
        else:
            print(f"   {twitch_id}: Video not found")

if __name__ == "__main__":
    main()
