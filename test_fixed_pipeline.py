#!/usr/bin/env python3
"""
Test script to verify the fixed pipeline works correctly
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for testing
os.environ['ENVIRONMENT'] = 'development'
os.environ['USE_GCS'] = 'true'  # Enable GCS but expect it to fail gracefully

from main import run_integrated_pipeline

async def test_fixed_pipeline():
    """Test the fixed pipeline with a real Twitch URL"""
    
    print("🧪 Testing Fixed Pipeline")
    print("=" * 50)
    
    # Test URL
    test_url = "https://www.twitch.tv/videos/2472774741"
    
    print(f"Testing with URL: {test_url}")
    print(f"Environment: {os.environ.get('ENVIRONMENT', 'production')}")
    print(f"GCS Enabled: {os.environ.get('USE_GCS', 'false')}")
    print()
    
    start_time = time.time()
    
    try:
        print("🚀 Starting pipeline...")
        result = await run_integrated_pipeline(test_url)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 50)
        print("📊 Pipeline Results:")
        print("=" * 50)
        
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Video ID: {result.get('video_id', 'unknown')}")
        print(f"Total Duration: {duration:.2f} seconds")
        
        if result.get('stage_times'):
            print("\n⏱️  Stage Times:")
            for stage, stage_time in result['stage_times'].items():
                print(f"  {stage}: {stage_time:.2f}s")
        
        if result.get('status') == 'completed':
            print("\n✅ Pipeline completed successfully!")
            print("🎉 The fix worked - pipeline no longer fails on GCS upload issues!")
            
            if result.get('files'):
                print(f"\n📁 Generated files: {len(result['files'])}")
                for file_type, file_path in result['files'].items():
                    print(f"  {file_type}: {file_path}")
        else:
            print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n❌ Pipeline failed with exception: {str(e)}")
        print(f"Duration before failure: {duration:.2f} seconds")
        return False
    
    return True

def main():
    """Main test function"""
    print("🔧 Testing Fixed Pipeline for GCS Upload Issues")
    print("This test verifies that the pipeline completes successfully")
    print("even when GCS uploads fail in development mode.")
    print()
    
    try:
        success = asyncio.run(test_fixed_pipeline())
        
        if success:
            print("\n🎉 Test PASSED - Pipeline fix is working correctly!")
            print("The pipeline now handles GCS upload failures gracefully.")
        else:
            print("\n❌ Test FAILED - Pipeline still has issues.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
