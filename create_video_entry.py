#!/usr/bin/env python3
"""
Script to create a video entry in Convex database for pipeline testing
"""

import os
import sys
from dotenv import load_dotenv
from convex_api import ConvexAPIClient

# Load environment variables
load_dotenv()

def create_video_entry(twitch_id: str):
    """Create a video entry in Convex database"""
    
    print(f"Creating video entry for Twitch ID: {twitch_id}")
    
    # Initialize Convex client
    client = ConvexAPIClient()
    
    # Check if video already exists
    existing_video = client.get_video_by_twitch_id(twitch_id)
    if existing_video:
        print(f"Video {twitch_id} already exists in Convex database")
        print(f"Video ID: {existing_video.get('_id')}")
        print(f"Title: {existing_video.get('title')}")
        print(f"Status: {existing_video.get('status')}")
        return True
    
    # Try to create the video using the minimal creation function
    print(f"Video {twitch_id} not found, attempting to create...")
    
    try:
        success = client.create_video_minimal(twitch_id, "Queued")
        if success:
            print(f"✅ Successfully created video entry for {twitch_id}")
            
            # Verify it was created
            created_video = client.get_video_by_twitch_id(twitch_id)
            if created_video:
                print(f"✅ Verified: Video ID {created_video.get('_id')} created")
                return True
            else:
                print("❌ Error: Video was created but cannot be retrieved")
                return False
        else:
            print(f"❌ Failed to create video entry for {twitch_id}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating video entry: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_video_entry.py <twitch_video_id>")
        print("Example: python create_video_entry.py 2479611486")
        sys.exit(1)
    
    twitch_id = sys.argv[1]
    
    # Validate twitch_id is numeric
    if not twitch_id.isdigit():
        print(f"Error: Twitch video ID must be numeric, got: {twitch_id}")
        sys.exit(1)
    
    success = create_video_entry(twitch_id)
    
    if success:
        print(f"\n🎉 Video entry ready! You can now run the pipeline:")
        print(f"python main.py https://www.twitch.tv/videos/{twitch_id}")
        sys.exit(0)
    else:
        print(f"\n❌ Failed to create video entry for {twitch_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()
