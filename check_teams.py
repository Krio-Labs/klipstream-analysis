#!/usr/bin/env python3
"""
Script to check available teams in Convex database
"""

import os
import sys
from dotenv import load_dotenv
from convex_api import ConvexAPIClient

# Load environment variables
load_dotenv()

def check_teams():
    """Check what teams are available in the database"""
    
    print("Checking available teams in Convex database...")
    
    # Initialize Convex client
    client = ConvexAPIClient()
    
    try:
        # Try to get some existing videos to see what team IDs are being used
        print("Attempting to find existing videos to see team IDs...")
        
        # We can't directly query teams, but we can look at existing videos
        # Let's try a few different approaches
        
        # Try to call a query that might give us team info
        result = client.query("video:list", {
            "paginationOpts": {"numItems": 5, "cursor": None},
            "team": ""  # Try with empty team
        })
        
        if result and result.get("page"):
            videos = result["page"]
            print(f"Found {len(videos)} videos:")
            for video in videos:
                print(f"  Video ID: {video.get('_id')}")
                print(f"  Twitch ID: {video.get('twitch_id')}")
                print(f"  Team ID: {video.get('team')}")
                print(f"  Title: {video.get('title')}")
                print("  ---")
        else:
            print("No videos found or query failed")
            
    except Exception as e:
        print(f"Error checking teams: {str(e)}")
        
        # Try a different approach - check if we can create a team
        try:
            print("Attempting to create a system user with team...")
            result = client.mutation("user:createSystemUser", {})
            if result:
                print(f"System user result: {result}")
                return result.get("teamId")
        except Exception as e2:
            print(f"Error creating system user: {str(e2)}")
    
    return None

def main():
    team_id = check_teams()
    if team_id:
        print(f"\n✅ Found/created team ID: {team_id}")
    else:
        print("\n❌ Could not find or create a team ID")

if __name__ == "__main__":
    main()
