#!/usr/bin/env python3
"""
Simple script to check if a KlipStream analysis job is complete
"""

import requests
import json
import sys

def is_processing_complete(job_id: str) -> tuple[bool, str, dict]:
    """
    Check if processing is complete for a given job ID
    
    Returns:
        tuple: (is_complete, status, full_response)
    """
    api_url = f"https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/api/v1/analysis/{job_id}/status"
    
    try:
        response = requests.get(api_url, timeout=10)
        
        if response.status_code != 200:
            return False, f"API Error: {response.status_code}", {}
        
        data = response.json()
        status = data.get('status', 'unknown')
        
        # Check if complete
        if status == 'completed':
            return True, 'completed', data
        elif status == 'failed':
            return True, 'failed', data
        else:
            return False, status, data
            
    except Exception as e:
        return False, f"Error: {str(e)}", {}

def print_status_summary(data: dict):
    """Print a summary of the current status"""
    print(f"Job ID: {data.get('job_id', 'unknown')}")
    print(f"Status: {data.get('status', 'unknown')}")
    print(f"Progress: {data.get('progress', 0)}%")
    print(f"Current Stage: {data.get('current_stage', 'unknown')}")
    
    # Print stage breakdown
    stages = data.get('stages', {})
    if stages:
        print("\nStage Status:")
        for stage, stage_status in stages.items():
            icon = "✅" if stage_status == "completed" else "⏳" if stage_status == "processing" else "❌" if stage_status == "failed" else "⏸️"
            print(f"  {icon} {stage}: {stage_status}")
    
    # Print error if failed
    if data.get('status') == 'failed' and data.get('error'):
        print(f"\nError: {data.get('error')}")
    
    # Print results if completed
    if data.get('status') == 'completed' and data.get('results'):
        results = data.get('results', {})
        print(f"\n🎉 Analysis Complete! Results available:")
        
        for key, url in results.items():
            if url and key.endswith('_url'):
                print(f"  📄 {key}: {url}")
        
        # Show highlights summary
        highlights = results.get('highlights', [])
        if highlights:
            print(f"\n🎬 Found {len(highlights)} highlights:")
            for i, highlight in enumerate(highlights[:3]):  # Show first 3
                start_min = int(highlight['start_time'] // 60)
                start_sec = int(highlight['start_time'] % 60)
                print(f"  {i+1}. {highlight['type']} at {start_min}:{start_sec:02d} (score: {highlight['score']:.2f})")
            if len(highlights) > 3:
                print(f"  ... and {len(highlights) - 3} more")

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_completion.py <job_id>")
        print("Example: python check_completion.py analysis_20241210_103045_abc123")
        sys.exit(1)
    
    job_id = sys.argv[1]
    print(f"Checking status for job: {job_id}")
    print("=" * 50)
    
    is_complete, status, data = is_processing_complete(job_id)
    
    if data:
        print_status_summary(data)
    else:
        print(f"Failed to get status: {status}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    if is_complete:
        if status == 'completed':
            print("🎉 PROCESSING COMPLETE! ✅")
        elif status == 'failed':
            print("❌ PROCESSING FAILED!")
        sys.exit(0)
    else:
        print(f"⏳ Still processing... (Status: {status})")
        print("Run this script again to check progress.")
        sys.exit(1)

if __name__ == "__main__":
    main()
