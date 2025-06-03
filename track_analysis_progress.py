#!/usr/bin/env python3
"""
Real-time Analysis Progress Tracker
Tracks the progress of a Twitch VOD analysis job from start to completion.
"""

import requests
import time
import json
from datetime import datetime
import sys

def track_analysis_progress(job_id, base_url="http://localhost:3001"):
    """Track analysis progress until completion"""
    
    status_url = f"{base_url}/api/v1/analysis/{job_id}/status"
    
    print(f"🎯 Tracking Analysis Progress")
    print(f"Job ID: {job_id}")
    print(f"Status URL: {status_url}")
    print("=" * 60)
    
    last_stage = ""
    last_percentage = -1
    start_time = time.time()
    
    while True:
        try:
            # Get current status
            response = requests.get(status_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()

                # Handle both response formats
                if "job" in data:
                    job = data["job"]
                elif "job_id" in data:
                    job = data  # Direct job object
                else:
                    job = None

                if job:
                    progress = job.get("progress", {})

                    current_stage = progress.get("current_stage", "Unknown")
                    current_percentage = progress.get("percentage", 0)
                    job_status = job.get("status", "unknown")
                    
                    # Only print updates when something changes
                    if current_stage != last_stage or abs(current_percentage - last_percentage) >= 1:
                        elapsed = time.time() - start_time
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        print(f"[{timestamp}] {job_status.upper()}: {current_stage} - {current_percentage:.1f}% (Elapsed: {elapsed:.0f}s)")
                        
                        # Show estimated completion if available
                        if progress.get("estimated_completion_seconds"):
                            eta_minutes = progress["estimated_completion_seconds"] / 60
                            print(f"           ETA: {eta_minutes:.1f} minutes")
                        
                        last_stage = current_stage
                        last_percentage = current_percentage
                    
                    # Check if job is complete
                    if job_status in ["completed", "failed"]:
                        print("\n" + "=" * 60)
                        if job_status == "completed":
                            print("🎉 Analysis COMPLETED successfully!")
                            
                            # Show results if available
                            if job.get("results"):
                                print("\n📊 Results:")
                                results = job["results"]
                                for key, value in results.items():
                                    if value:
                                        print(f"   {key}: {value}")
                        else:
                            print("❌ Analysis FAILED")
                            if job.get("error"):
                                error = job["error"]
                                print(f"   Error: {error.get('message', 'Unknown error')}")
                                print(f"   Type: {error.get('type', 'Unknown')}")
                                if error.get("is_retryable"):
                                    print("   ✅ This error is retryable")
                        
                        total_time = time.time() - start_time
                        print(f"\n⏱️  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
                        break
                        
                else:
                    print(f"❌ Unexpected response: {data}")
                    break
                    
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except KeyboardInterrupt:
            print("\n\n⏹️  Tracking stopped by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
        
        # Wait before next check
        time.sleep(2)

def main():
    if len(sys.argv) != 2:
        print("Usage: python track_analysis_progress.py <job_id>")
        print("Example: python track_analysis_progress.py f09aa3b2-4049-442e-be01-a58a27fd9935")
        sys.exit(1)
    
    job_id = sys.argv[1]
    track_analysis_progress(job_id)

if __name__ == "__main__":
    main()
