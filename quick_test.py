#!/usr/bin/env python3
"""
Quick test to start analysis and show how to check completion
"""

import requests
import json
import time

def start_analysis():
    """Start analysis and return job ID"""
    url = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis"
    
    payload = {
        "url": "https://www.twitch.tv/videos/2479611486",
        "user_id": "jh7en1zt0460hfzf2qa470p1k17epm40",
        "team_id": "js7bj9zgdkyj9ykvr4m6jarxkh7ep9fa",
        "options": {
            "transcription_method": "auto",
            "enable_highlights": True,
            "enable_sentiment": True,
            "chunk_duration": 10
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                job_id = data.get('job_id')
                print(f"✅ Analysis started! Job ID: {job_id}")
                print(f"📊 Estimated duration: {data.get('estimated_duration', 'unknown')}")
                return job_id
            else:
                print(f"❌ Failed to start: {data.get('error', 'unknown')}")
                return None
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def check_status(job_id):
    """Check current status"""
    url = f"https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis/{job_id}/status"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Status check error: {str(e)}")
        return None

def main():
    print("🚀 Starting KlipStream Analysis Test")
    print("=" * 50)
    
    # Start analysis
    job_id = start_analysis()
    if not job_id:
        print("Failed to start analysis")
        return
    
    print(f"\n📋 To check completion status, run:")
    print(f"python check_completion.py {job_id}")
    
    print(f"\n🔄 Or check manually at:")
    print(f"https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis/{job_id}/status")
    
    # Show initial status
    print(f"\n📊 Initial status check:")
    status_data = check_status(job_id)
    if status_data:
        print(f"Status: {status_data.get('status')}")
        print(f"Progress: {status_data.get('progress', 0)}%")
        print(f"Current Stage: {status_data.get('current_stage', 'unknown')}")
        
        # Show how to detect completion
        print(f"\n🎯 How to know when complete:")
        print(f"1. status == 'completed' ✅")
        print(f"2. progress == 100 📊")
        print(f"3. results object is present 📄")
        
        if status_data.get('status') == 'completed':
            print(f"\n🎉 Already complete!")
        elif status_data.get('status') == 'failed':
            print(f"\n❌ Failed: {status_data.get('error', 'unknown')}")
        else:
            print(f"\n⏳ Still processing... Check again in a few minutes")

if __name__ == "__main__":
    main()
