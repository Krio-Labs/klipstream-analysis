#!/usr/bin/env python3
"""
Test script to verify that Convex status updates now work correctly
"""

import os
import sys
from utils.convex_client_updated import ConvexManager
from convex_integration import ConvexIntegration

def test_status_update_fixed(twitch_id, new_status):
    """Test that status updates now work with the fixed ConvexManager"""
    print(f"\n🔧 Testing FIXED ConvexManager status update for video {twitch_id} to '{new_status}'...")
    
    # Create ConvexManager (should now always make real API calls)
    convex_manager = ConvexManager()
    
    # Test the update
    success = convex_manager.update_video_status(twitch_id, new_status)
    
    if success:
        print(f"✅ Status update successful")
    else:
        print(f"❌ Status update failed")
    
    return success

def verify_status_change(twitch_id, expected_status):
    """Verify that the status actually changed in the database"""
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

def test_url_update_fixed(twitch_id):
    """Test that URL updates now work with the fixed ConvexManager"""
    print(f"\n🔗 Testing FIXED ConvexManager URL update for video {twitch_id}...")
    
    # Create ConvexManager
    convex_manager = ConvexManager()
    
    # Test URL update
    test_urls = {
        'analysis_url': f'gs://klipstream-analysis/{twitch_id}/test_analysis_fixed.csv'
    }
    
    success = convex_manager.update_video_urls(twitch_id, test_urls)
    
    if success:
        print(f"✅ URL update successful")
    else:
        print(f"❌ URL update failed")
    
    return success

def verify_url_change(twitch_id, expected_url):
    """Verify that the URL actually changed in the database"""
    print(f"\n🔍 Verifying URL change for video {twitch_id}...")
    
    convex = ConvexIntegration()
    video = convex.get_video(twitch_id)
    
    if video:
        current_url = video.get('analysis_url', 'No URL')
        print(f"   Current analysis_url: {current_url}")
        
        if current_url == expected_url:
            print(f"✅ URL change verified!")
            return True
        else:
            print(f"❌ URL change NOT verified (expected: {expected_url})")
            return False
    else:
        print(f"❌ Video not found for verification")
        return False

def main():
    """Main test function"""
    print("🚀 CONVEX STATUS UPDATE FIXED TEST")
    print("=" * 50)
    
    # Test with one video from the screenshot
    test_video = "2434635255"  # Jynxzi video
    
    print(f"\n🎬 TESTING FIXED UPDATES FOR VIDEO: {test_video}")
    print("=" * 60)
    
    # Test 1: Status Update
    print(f"\n📋 TEST 1: Status Update")
    test_status = "Downloading"  # Use a valid status
    success1 = test_status_update_fixed(test_video, test_status)
    
    if success1:
        verify_status_change(test_video, test_status)
    
    # Test 2: URL Update
    print(f"\n📋 TEST 2: URL Update")
    expected_url = f'gs://klipstream-analysis/{test_video}/test_analysis_fixed.csv'
    success2 = test_url_update_fixed(test_video)
    
    if success2:
        verify_url_change(test_video, expected_url)
    
    # Test 3: Combined Update
    print(f"\n📋 TEST 3: Combined Status + URL Update")
    combined_status = "Transcribing"
    combined_url = f'gs://klipstream-analysis/{test_video}/test_combined_fixed.csv'
    
    convex_manager = ConvexManager()
    success3 = convex_manager.update_pipeline_progress(
        test_video, 
        status=combined_status,
        urls={'analysis_url': combined_url}
    )
    
    if success3:
        print(f"✅ Combined update successful")
        verify_status_change(test_video, combined_status)
        verify_url_change(test_video, combined_url)
    else:
        print(f"❌ Combined update failed")
    
    # Final verification
    print(f"\n🎉 FINAL VERIFICATION")
    print("=" * 30)
    convex = ConvexIntegration()
    video = convex.get_video(test_video)
    if video:
        final_status = video.get('status', 'No status')
        final_url = video.get('analysis_url', 'No URL')
        print(f"Final status: {final_status}")
        print(f"Final analysis_url: {final_url}")
        
        if final_status in [test_status, combined_status] and 'test_' in final_url:
            print(f"✅ SUCCESS: Both status and URL were successfully updated!")
        else:
            print(f"❌ FAILURE: Updates were not reflected in database")
    
    print(f"\n🎉 FIXED UPDATE TEST COMPLETED")

if __name__ == "__main__":
    main()
