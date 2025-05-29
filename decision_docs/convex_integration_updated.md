# Convex Database Integration (Updated)

This document describes the integration between the KlipStream Analysis pipeline and the Convex database, which is used to track the status and output files of video processing.

## Overview

The KlipStream Analysis pipeline needs to update the Convex database with the status of the analysis and the URLs of the generated files. This integration is implemented using the Convex HTTP API.

## Implementation

The integration consists of three main components in a layered architecture:

1. `convex_api.py`: A low-level client for the Convex API
2. `convex_integration.py`: A high-level integration module that provides a more user-friendly interface
3. `utils/convex_client_updated.py`: A singleton wrapper around the `convex_integration.py` module for use in the pipeline

### Environment Variables

The integration requires the following environment variables:

- `CONVEX_URL`: The URL of the Convex deployment (e.g., `https://laudable-horse-446.convex.cloud`)
- `CONVEX_API_KEY`: The API key for the Convex deployment

These variables can be set in the `.env` file, `.env.yaml` file, or in the Cloud Run environment variables.

### Convex API Client

The `ConvexAPIClient` class in `convex_api.py` provides methods for interacting with the Convex API:

- `query`: Execute a Convex query function
- `mutation`: Execute a Convex mutation function
- `get_video_by_twitch_id`: Get a video by its Twitch ID
- `update_video_status`: Update the status of a video using its Convex ID
- `update_status_by_twitch_id`: Update the status of a video using its Twitch ID
- `update_video_urls`: Update the URLs of a video

### Convex Integration

The `ConvexIntegration` class in `convex_integration.py` provides a higher-level interface for integrating with the Convex database:

- `get_video`: Get a video by its Twitch ID
- `update_status`: Update the status of a video using its Convex ID
- `update_status_by_twitch_id`: Update the status of a video using its Twitch ID
- `update_urls`: Update the URLs of a video
- `update_transcript_url`: Update the transcript URL of a video
- `update_chat_url`: Update the chat URL of a video
- `update_audiowave_url`: Update the audiowave URL of a video
- `update_transcript_analysis_url`: Update the transcript analysis URL of a video
- `update_transcript_word_url`: Update the transcript word URL of a video
- `update_all_urls`: Update all URLs of a video

### ConvexManager

The `ConvexManager` class in `utils/convex_client_updated.py` provides a singleton instance of the `ConvexIntegration` class for use in the pipeline:

- `check_video_exists`: Check if a video exists in the database
- `update_video_status`: Update the status of a video
- `update_video_urls`: Update the URLs of a video
- `update_pipeline_progress`: Update both status and URLs in one operation

The singleton pattern ensures that only one instance of the `ConvexManager` is created throughout the application, which helps manage resources and maintain consistent state.

### Video Status Constants

The `convex_integration.py` module defines constants for the different status values:

- `STATUS_QUEUED`: "Queued"
- `STATUS_DOWNLOADING`: "Downloading"
- `STATUS_FETCHING_CHAT`: "Fetching chat"
- `STATUS_TRANSCRIBING`: "Transcribing"
- `STATUS_ANALYZING`: "Analyzing"
- `STATUS_FINDING_HIGHLIGHTS`: "Finding highlights"
- `STATUS_COMPLETED`: "Completed"
- `STATUS_FAILED`: "Failed"

These constants are imported and used throughout the codebase to ensure consistency in status values.

## Current Usage in the Pipeline

The `ConvexManager` is used extensively throughout the pipeline to update the status and URLs of videos as they progress through the analysis stages:

### In Raw Pipeline (raw_pipeline/processor.py)

```python
# Initialize Convex client
convex_manager = ConvexManager()

# Update status when starting download
convex_manager.update_video_status(video_id, STATUS_DOWNLOADING)

# Update status when fetching chat
convex_manager.update_video_status(video_id, STATUS_FETCHING_CHAT)

# Update status when starting transcription
convex_manager.update_video_status(video_id, STATUS_TRANSCRIBING)
```

### In Analysis Pipeline (analysis_pipeline/processor.py)

```python
# Initialize Convex client
convex_manager = ConvexManager()

# Update status when starting analysis
convex_manager.update_video_status(video_id, STATUS_ANALYZING)

# Update status when finding highlights
convex_manager.update_video_status(video_id, STATUS_FINDING_HIGHLIGHTS)
```

### URL Updates

After generating output files, the pipeline updates the Convex database with the URLs of these files:

```python
# Update URLs with GCS paths
convex_manager.update_video_urls(
    video_id,
    {
        "transcriptUrl": transcript_url,
        "chatUrl": chat_url,
        "audiowaveUrl": audiowave_url,
        "transcriptAnalysisUrl": analysis_url,
        "transcriptWordUrl": word_url
    }
)
```

## Error Handling and Retry Logic

The `ConvexManager` includes robust error handling and retry logic to ensure that transient errors don't cause the pipeline to fail:

```python
# Try to update with retries
for attempt in range(max_retries):
    try:
        # Call the method to update the status
        success = self.convex.update_status_by_twitch_id(twitch_id, status)

        if success:
            return True
        else:
            # If unsuccessful but no exception, retry with exponential backoff
            if attempt < max_retries - 1:
                retry_delay = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                time.sleep(retry_delay)
    except Exception as e:
        # If an exception occurs, log it and retry
        if attempt < max_retries - 1:
            retry_delay = 2 ** attempt
            time.sleep(retry_delay)
```

This retry logic uses exponential backoff to handle transient network issues or API rate limits.

## Test Mode

The `ConvexManager` includes a test mode that can be enabled for development and testing:

```python
# Set to False to make actual API calls
self.test_mode = False
logger.info("Running in LIVE MODE - actual Convex API calls will be made")
```

When test mode is enabled, the manager logs what actions would be taken but doesn't make actual API calls.

## Convex Functions

The integration uses the following Convex functions:

- `video:getByTwitchIdPublic`: Get a video by its Twitch ID
- `video:updateStatus`: Update the status of a video using its Convex ID
- `video:updateStatusByTwitchId`: Update the status of a video using its Twitch ID
- `video:updateUrls`: Update the URLs of a video

These functions must be defined in the Convex backend project. An example implementation is provided in `convex_schema_example.js`.

## Current Status and Limitations

The Convex integration is fully implemented and actively used in the pipeline. However, there are a few limitations:

1. **No Authentication**: The current implementation doesn't use authentication for the Convex API, which could be a security concern.

2. **Limited Error Recovery**: While there is retry logic, there's no mechanism to recover from a failed update if all retries fail.

3. **No Test Script**: The documentation mentions a `test_convex_manager.py` script, but this file doesn't exist in the current codebase.

## Future Improvements

Based on the current implementation, here are some recommended improvements:

1. **Add Authentication**: Implement proper authentication for the Convex API using JWT tokens.

2. **Create Test Script**: Develop the mentioned `test_convex_manager.py` script to facilitate testing.

3. **Implement Offline Mode**: Add an offline mode that stores updates locally when the Convex API is unavailable.

4. **Add More Comprehensive Logging**: Enhance logging to provide more detailed information about API calls and responses.

5. **Add Unit Tests**: Develop unit tests for the Convex integration components.

6. **Implement Status Rollback**: Add functionality to roll back to a previous status if a pipeline stage fails.
