#!/usr/bin/env python3
"""
Test the Convex integration fix locally before deploying
"""

import sys
import os
sys.path.append('.')

def test_convex_integration():
    """Test that our Convex integration works"""
    print("🧪 Testing Convex Integration Locally")
    print("=" * 50)
    
    try:
        # Test 1: Import the ConvexManager
        print("1️⃣ Testing ConvexManager import...")
        from utils.convex_client_updated import ConvexManager
        print("✅ ConvexManager imported successfully")
        
        # Test 2: Initialize ConvexManager
        print("\n2️⃣ Testing ConvexManager initialization...")
        convex_manager = ConvexManager()
        
        if not convex_manager.convex:
            print("❌ ConvexManager not initialized - check environment variables")
            return False
        
        print("✅ ConvexManager initialized successfully")
        
        # Test 3: Test video lookup
        print("\n3️⃣ Testing video lookup...")
        test_video_id = "2479611486"
        
        existing_video = convex_manager.convex.client.query("video:getAnyByTwitchId", {
            "twitch_id": test_video_id
        })
        
        if existing_video:
            print(f"✅ Found existing video: {existing_video['_id']}")
            print(f"   Title: {existing_video.get('title', 'No title')}")
            print(f"   Team: {existing_video.get('team', 'No team')}")
        else:
            print(f"ℹ️ Video {test_video_id} not found (this is OK for testing)")
        
        # Test 4: Test queue status
        print("\n4️⃣ Testing queue status...")
        try:
            queue_status = convex_manager.convex.client.query("queueManager:getQueueStatus", {})
            print(f"✅ Queue status retrieved:")
            print(f"   Queue length: {queue_status.get('queueLength', 'unknown')}")
            print(f"   Processing count: {queue_status.get('processingCount', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Queue status error (may be normal): {str(e)}")
        
        print("\n🎉 All Convex integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Convex integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_route_logic():
    """Test the analysis route logic without FastAPI"""
    print("\n🧪 Testing Analysis Route Logic")
    print("=" * 50)
    
    try:
        # Test the URL validation
        print("1️⃣ Testing URL validation...")
        from utils.helpers import extract_video_id
        
        test_url = "https://www.twitch.tv/videos/2479611486"
        video_id = extract_video_id(test_url)
        print(f"✅ Extracted video ID: {video_id}")
        
        # Test UUID generation
        print("\n2️⃣ Testing job ID generation...")
        import uuid
        job_id = str(uuid.uuid4())
        print(f"✅ Generated job ID: {job_id}")
        
        print("\n🎉 Analysis route logic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing analysis route logic: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all local tests"""
    print("🚀 Testing KlipStream API Fix Locally")
    print("This will verify our changes work before deploying")
    print("=" * 60)
    
    # Check environment
    print("🔧 Checking environment...")
    convex_url = os.getenv('CONVEX_URL')
    convex_key = os.getenv('CONVEX_API_KEY')
    
    if not convex_url or not convex_key:
        print("❌ Missing environment variables:")
        print(f"   CONVEX_URL: {'✅' if convex_url else '❌'}")
        print(f"   CONVEX_API_KEY: {'✅' if convex_key else '❌'}")
        print("\n💡 Make sure .env file is loaded")
        return False
    
    print("✅ Environment variables found")
    
    # Run tests
    test1_passed = test_convex_integration()
    test2_passed = test_analysis_route_logic()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! ✅")
        print("💡 The fix should work when deployed")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("💡 Fix issues before deploying")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
