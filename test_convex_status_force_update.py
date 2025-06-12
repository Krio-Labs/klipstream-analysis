#!/usr/bin/env python3
"""
Test script to force Convex status updates by bypassing local mode detection
"""

import os
import sys
from utils.convex_client_updated import ConvexManager
from convex_integration import ConvexIntegration

def force_convex_update(twitch_id, new_status):
    """Force a Convex status update by bypassing local mode detection"""
    print(f"\n🔧 FORCING status update for video {twitch_id} to '{new_status}'...")
    
    # Temporarily set Cloud Run environment variable to bypass local mode
    original_cloud_run = os.environ.get('CLOUD_RUN_SERVICE')
    os.environ['CLOUD_RUN_SERVICE'] = 'klipstream-analysis'  # Fake Cloud Run service name
    
    try:
        # Create ConvexManager with forced cloud mode
        convex_manager = ConvexManager()
        
        # Force the update
        success = convex_manager.update_video_status(twitch_id, new_status)
        
        if success:
            print(f"✅ FORCED status update successful")
        else:
            print(f"❌ FORCED status update failed")
        
        return success
        
    finally:
        # Restore original environment variable
        if original_cloud_run is None:
            if 'CLOUD_RUN_SERVICE' in os.environ:
                del os.environ['CLOUD_RUN_SERVICE']
        else:
            os.environ['CLOUD_RUN_SERVICE'] = original_cloud_run

def test_direct_convex_integration(twitch_id, new_status):
    """Test direct ConvexIntegration without ConvexManager wrapper"""
    print(f"\n🎯 DIRECT ConvexIntegration test for video {twitch_id} to '{new_status}'...")
    
    try:
        # Use ConvexIntegration directly
        convex = ConvexIntegration()
        success = convex.update_status_by_twitch_id(twitch_id, new_status)
        
        if success:
            print(f"✅ DIRECT status update successful")
        else:
            print(f"❌ DIRECT status update failed")
        
        return success
        
    except Exception as e:
        print(f"❌ DIRECT status update error: {str(e)}")
        return False

def verify_status_change(twitch_id, expected_status):
    """Verify that the status actually changed"""
    print(f"\n🔍 Verifying status change for video {twitch_id}...")
    
    convex = ConvexIntegration()
    video = convex.get_video(twitch_id)
    
    if video:
        current_status = video.get('status', 'No status')
        print(f"   Current status: {current_status}")
        
        if current_status == expected_status:
            print(f"✅ Status change verified!")
            return True
        else:
            print(f"❌ Status change NOT verified (expected: {expected_status})")
            return False
    else:
        print(f"❌ Video not found for verification")
        return False

def main():
    """Main test function"""
    print("🚀 CONVEX STATUS UPDATE FORCE TEST")
    print("=" * 50)
    
    # Test with one video from the screenshot
    test_video = "2434635255"  # Jynxzi video
    test_status = "Testing Force Update"
    
    print(f"\n🎬 TESTING FORCED UPDATE FOR VIDEO: {test_video}")
    print(f"🎯 TARGET STATUS: {test_status}")
    print("=" * 60)
    
    # Method 1: Force update through ConvexManager
    print(f"\n📋 METHOD 1: ConvexManager with forced cloud mode")
    success1 = force_convex_update(test_video, test_status)
    
    if success1:
        verify_status_change(test_video, test_status)
    
    # Method 2: Direct ConvexIntegration
    print(f"\n📋 METHOD 2: Direct ConvexIntegration")
    test_status_2 = "Testing Direct Update"
    success2 = test_direct_convex_integration(test_video, test_status_2)
    
    if success2:
        verify_status_change(test_video, test_status_2)
    
    # Final verification
    print(f"\n🎉 FINAL VERIFICATION")
    print("=" * 30)
    convex = ConvexIntegration()
    video = convex.get_video(test_video)
    if video:
        final_status = video.get('status', 'No status')
        print(f"Final status: {final_status}")
        
        if final_status in [test_status, test_status_2]:
            print(f"✅ SUCCESS: Status was successfully updated!")
        else:
            print(f"❌ FAILURE: Status was not updated")
    
    print(f"\n🎉 FORCE UPDATE TEST COMPLETED")

if __name__ == "__main__":
    main()
