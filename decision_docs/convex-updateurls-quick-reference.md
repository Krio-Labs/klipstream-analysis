# Convex updateUrls Quick Reference

## ✅ Function Ready: `video:updateUrls`

The Convex function is **deployed and working**. Use this format:

```python
# CORRECT FORMAT
result = convex_client.mutation("video:updateUrls", {
    "twitchId": "2434635255",  # String: Your video ID
    "urls": {                  # Object: URL fields to update
        "transcript_url": "https://your-bucket/transcript.json",
        "chat_url": "https://your-bucket/chat.csv",
        "waveform_url": "https://your-bucket/waveform.json", 
        "analysis_url": "https://your-bucket/analysis.json",
        "transcriptWords_url": "https://your-bucket/words.csv"
    }
})
# Returns: {"status": "success", "videoId": "convex_id"}
```

## 🔧 Supported URL Fields

**Use these standardized names:**
- `transcript_url` - Transcript segments file
- `transcriptWords_url` - Word-level transcript file  
- `chat_url` - Chat log file
- `waveform_url` - Audio waveform data
- `analysis_url` - Analysis results file
- `video_url` - Processed video file
- `audio_url` - Audio file

**Legacy names (still work):**
- `transcriptUrl`, `transcriptWordUrl`, `chatUrl`, `audiowaveUrl`, `transcriptAnalysisUrl`

## 🚨 Common Mistakes to Avoid

❌ **DON'T** use custom field names:
```python
# WRONG - will cause validation error
"urls": {"my_custom_url": "https://..."}
```

❌ **DON'T** use old function names:
```python
# WRONG - function doesn't exist
convex_client.mutation("video:updateVideoUrls", {...})
```

✅ **DO** use exact function name and supported fields:
```python
# CORRECT
convex_client.mutation("video:updateUrls", {
    "twitchId": video_id,
    "urls": {"transcript_url": "https://..."}
})
```

## 🔄 Replace Your Existing Code

**If you have this pattern:**
```python
# OLD/BROKEN CODE
convex_manager.update_video_urls(video_id, {
    "transcriptAnalysisUrl": analysis_url
})
```

**Replace with:**
```python
# NEW WORKING CODE  
convex_manager.update_video_urls(video_id, {
    "analysis_url": analysis_url  # Use standardized name
})
```

## 🧪 Test It Works

```python
# Quick test
convex_manager = ConvexManager()
success = convex_manager.update_video_urls("your_video_id", {
    "transcript_url": "https://example.com/test.json"
})
print("SUCCESS!" if success else "FAILED!")
```

## 🌐 Environment URLs

- **Production**: `https://benevolent-gnat-59.convex.cloud`
- **Development**: `https://laudable-horse-446.convex.cloud`

## ⚡ Ready to Use!

The function is deployed and tested. Just update your Python code with the correct format above and it will work immediately! 🚀
