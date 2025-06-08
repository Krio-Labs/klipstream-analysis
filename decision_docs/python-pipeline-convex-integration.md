# Python Pipeline Convex Integration Guide

## Overview

The Convex `video:updateUrls` function is now fully implemented and ready for Python analysis pipeline integration. This guide provides everything needed to update your Python code to work with the new Convex backend.

## Quick Start

### 1. Update Your Python Code

Replace any existing Convex URL update calls with the new `video:updateUrls` function:

```python
# OLD CODE (if you had any working version)
# convex_client.mutation("video:updateVideoUrls", {...})

# NEW CODE - Use this format
result = convex_client.mutation("video:updateUrls", {
    "twitchId": video_id,  # String: Twitch video ID (e.g., "2434635255")
    "urls": {              # Object: URL fields to update
        "transcript_url": "https://storage.googleapis.com/bucket/transcript.json",
        "chat_url": "https://storage.googleapis.com/bucket/chat.csv",
        "waveform_url": "https://storage.googleapis.com/bucket/waveform.json",
        "analysis_url": "https://storage.googleapis.com/bucket/analysis.json",
        "transcriptWords_url": "https://storage.googleapis.com/bucket/words.csv"
    }
})

# Expected response: {"status": "success", "videoId": "convex_video_id"}
```

### 2. Supported URL Fields

Use these **standardized field names** (recommended):

```python
url_updates = {
    "video_url": "...",           # Processed video file URL
    "audio_url": "...",           # Audio file URL  
    "transcript_url": "...",      # Transcript segments JSON/CSV URL
    "analysis_url": "...",        # Full analysis results JSON URL
    "waveform_url": "...",        # Audio waveform data URL
    "transcriptWords_url": "...", # Word-level transcript URL
    "chat_url": "...",            # Chat log URL
}
```

**Legacy field names** (still supported for backward compatibility):
```python
legacy_urls = {
    "transcriptUrl": "...",       # Legacy transcript URL
    "transcriptWordUrl": "...",   # Legacy word-level transcript URL
    "chatUrl": "...",             # Legacy chat URL
    "audiowaveUrl": "...",        # Legacy audio waveform URL
    "transcriptAnalysisUrl": "...", # Legacy analysis URL
}
```

## Implementation Examples

### Example 1: Using ConvexManager (Recommended)

If you're using the existing `ConvexManager` class:

```python
from utils.convex_client_updated import ConvexManager

# Initialize manager
convex_manager = ConvexManager()

# Update URLs after analysis completion
url_updates = {
    "transcript_url": f"https://storage.googleapis.com/klipstream-transcripts/{video_id}/segments.csv",
    "transcriptWords_url": f"https://storage.googleapis.com/klipstream-transcripts/{video_id}/words.csv", 
    "chat_url": f"https://storage.googleapis.com/klipstream-chatlogs/{video_id}/chat.csv",
    "waveform_url": f"https://storage.googleapis.com/klipstream-vods-raw/{video_id}/waveform.json",
    "analysis_url": f"https://storage.googleapis.com/klipstream-analysis/{video_id}/analysis.json"
}

success = convex_manager.update_video_urls(video_id, url_updates)
if success:
    print(f"Successfully updated URLs for video {video_id}")
    # Mark video as completed
    convex_manager.update_video_status(video_id, "completed")
else:
    print(f"Failed to update URLs for video {video_id}")
```

### Example 2: Direct Convex Client Usage

If you're using the Convex client directly:

```python
import os
from convex import ConvexClient

# Initialize client
client = ConvexClient(os.environ["CONVEX_URL"])

# Update URLs
try:
    result = client.mutation("video:updateUrls", {
        "twitchId": video_id,
        "urls": {
            "analysis_url": analysis_file_url,
            "transcript_url": transcript_file_url,
            "chat_url": chat_file_url,
            "waveform_url": waveform_file_url,
            "transcriptWords_url": words_file_url
        }
    })
    
    if result.get("status") == "success":
        print(f"URLs updated successfully for video {video_id}")
        video_convex_id = result.get("videoId")
    else:
        print(f"URL update failed for video {video_id}")
        
except Exception as e:
    print(f"Error updating URLs: {str(e)}")
```

### Example 3: Integration in Analysis Pipeline

Here's how to integrate it into your analysis workflow:

```python
async def process_analysis(video_id):
    """Process analysis and update Convex with results"""
    
    convex_manager = ConvexManager()
    
    try:
        # Update status to analyzing
        convex_manager.update_video_status(video_id, "analyzing")
        
        # Run your analysis steps here
        # ... analysis code ...
        
        # After successful analysis, update URLs
        if analysis_successful:
            # Upload files to GCS and get URLs
            uploaded_files = upload_analysis_results_to_gcs(video_id)
            
            # Prepare URL updates
            url_updates = {}
            
            if uploaded_files.get("transcript_segments"):
                url_updates["transcript_url"] = uploaded_files["transcript_segments"]["gcs_uri"]
                
            if uploaded_files.get("transcript_words"):
                url_updates["transcriptWords_url"] = uploaded_files["transcript_words"]["gcs_uri"]
                
            if uploaded_files.get("chat"):
                url_updates["chat_url"] = uploaded_files["chat"]["gcs_uri"]
                
            if uploaded_files.get("waveform"):
                url_updates["waveform_url"] = uploaded_files["waveform"]["gcs_uri"]
                
            if uploaded_files.get("analysis"):
                url_updates["analysis_url"] = uploaded_files["analysis"]["gcs_uri"]
            
            # Update Convex with URLs
            if url_updates:
                success = convex_manager.update_video_urls(video_id, url_updates)
                if success:
                    # Mark as completed
                    convex_manager.update_video_status(video_id, "completed")
                    return {"status": "completed", "urls": url_updates}
                else:
                    convex_manager.update_video_status(video_id, "failed")
                    return {"status": "failed", "error": "Failed to update URLs"}
            else:
                convex_manager.update_video_status(video_id, "completed")
                return {"status": "completed", "message": "No URLs to update"}
        
    except Exception as e:
        # Update status to failed
        convex_manager.update_video_status(video_id, "failed")
        return {"status": "failed", "error": str(e)}
```

## Error Handling

### Common Errors and Solutions

1. **"Video with Twitch ID {id} not found"**
   ```python
   # Solution: Ensure video exists in database before updating URLs
   video = convex_manager.get_video_by_twitch_id(video_id)
   if not video:
       print(f"Video {video_id} not found in database")
       return False
   ```

2. **"Object contains extra field that is not in the validator"**
   ```python
   # Solution: Only use supported URL field names
   # Use standardized names: transcript_url, chat_url, etc.
   # Avoid custom field names like "custom_url" or "my_analysis_url"
   ```

3. **Network/Connection Errors**
   ```python
   # Solution: Implement retry logic
   def update_urls_with_retry(convex_manager, video_id, urls, max_retries=3):
       for attempt in range(max_retries):
           try:
               success = convex_manager.update_video_urls(video_id, urls)
               if success:
                   return True
           except Exception as e:
               if attempt < max_retries - 1:
                   time.sleep(2 ** attempt)  # Exponential backoff
               else:
                   raise e
       return False
   ```

## Environment Configuration

Ensure your environment variables are set correctly:

```bash
# Production
CONVEX_URL=https://benevolent-gnat-59.convex.cloud

# Development  
CONVEX_URL=https://laudable-horse-446.convex.cloud
```

## Testing Your Implementation

### Test with a Real Video

```python
# Test script
def test_url_update():
    convex_manager = ConvexManager()
    
    # Use a real video ID from your database
    test_video_id = "2479611486"  # Replace with actual video ID
    
    # Test URL updates
    test_urls = {
        "transcript_url": "https://example.com/test-transcript.json",
        "chat_url": "https://example.com/test-chat.csv"
    }
    
    success = convex_manager.update_video_urls(test_video_id, test_urls)
    print(f"Test result: {'SUCCESS' if success else 'FAILED'}")
    
    # Verify the update
    video = convex_manager.get_video_by_twitch_id(test_video_id)
    if video:
        print(f"Updated transcript_url: {video.get('transcript_url')}")
        print(f"Updated chat_url: {video.get('chat_url')}")

if __name__ == "__main__":
    test_url_update()
```

## Migration Checklist

- [ ] Update all `video:updateUrls` function calls to use correct format
- [ ] Replace legacy field names with standardized names (optional but recommended)
- [ ] Add proper error handling for URL updates
- [ ] Test with development environment first
- [ ] Verify URLs are being updated correctly in Convex dashboard
- [ ] Deploy to production and monitor for errors

## Support

If you encounter issues:

1. Check the Convex dashboard for function execution logs
2. Verify your video IDs exist in the database
3. Ensure URL field names match the supported schema
4. Test with the development environment first

The function is now fully deployed and ready for integration! 🚀
