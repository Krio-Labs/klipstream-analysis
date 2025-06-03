# Convex Database Schema Updates for Async API

## Overview

This document outlines the required database schema changes to support the new asynchronous API with job tracking and real-time progress updates.

## Current Schema

The current `videos` table in Convex has these fields:
- `_id` (Convex ID)
- `twitchId` (string) - Twitch video ID
- `title` (string) - Video title
- `status` (string) - Processing status
- `transcriptUrl` (optional string) - URL to transcript file
- `chatUrl` (optional string) - URL to chat file
- `audiowaveUrl` (optional string) - URL to audiowave file
- `transcriptAnalysisUrl` (optional string) - URL to analysis file
- `transcriptWordUrl` (optional string) - URL to word-level transcript

## Required Schema Updates

### New Fields for Job Tracking

Add the following optional fields to the `videos` table:

```typescript
// In your Convex schema definition
export default defineSchema({
  videos: defineTable({
    // Existing fields...
    twitchId: v.string(),
    title: v.string(),
    status: v.string(),
    transcriptUrl: v.optional(v.string()),
    chatUrl: v.optional(v.string()),
    audiowaveUrl: v.optional(v.string()),
    transcriptAnalysisUrl: v.optional(v.string()),
    transcriptWordUrl: v.optional(v.string()),
    
    // NEW FIELDS for async processing
    jobId: v.optional(v.string()),                    // Unique job identifier
    progressPercentage: v.optional(v.number()),       // Progress 0-100
    currentStage: v.optional(v.string()),             // Current processing stage
    estimatedCompletionSeconds: v.optional(v.number()), // ETA in seconds
    
    // Error handling
    errorType: v.optional(v.string()),                // Type of error if failed
    errorMessage: v.optional(v.string()),             // Human-readable error message
    errorDetails: v.optional(v.string()),             // Detailed error information
    isRetryable: v.optional(v.boolean()),             // Whether operation can be retried
    retryCount: v.optional(v.number()),               // Number of retry attempts
    
    // Timestamps
    createdAt: v.optional(v.number()),                // Job creation timestamp
    updatedAt: v.optional(v.number()),                // Last update timestamp
    processingStartedAt: v.optional(v.number()),      // When processing began
    processingCompletedAt: v.optional(v.number()),    // When processing finished
    
    // Additional metadata
    callbackUrl: v.optional(v.string()),              // Webhook callback URL
    stageProgress: v.optional(v.object({})),          // Progress for each stage
  })
    .index("by_twitch_id", ["twitchId"])
    .index("by_job_id", ["jobId"])
    .index("by_status", ["status"])
    .index("by_created_at", ["createdAt"])
});
```

### New Convex Functions Required

The backend will need these Convex functions to be implemented:

#### 1. Job Progress Update Function
```typescript
// convex/videos.ts
export const updateJobProgress = mutation({
  args: {
    twitchId: v.string(),
    jobId: v.string(),
    progressPercentage: v.optional(v.number()),
    currentStage: v.optional(v.string()),
    estimatedCompletionSeconds: v.optional(v.number()),
    stageProgress: v.optional(v.object({})),
  },
  handler: async (ctx, args) => {
    const video = await ctx.db
      .query("videos")
      .withIndex("by_twitch_id", (q) => q.eq("twitchId", args.twitchId))
      .first();
    
    if (!video) {
      throw new Error(`Video not found: ${args.twitchId}`);
    }
    
    await ctx.db.patch(video._id, {
      jobId: args.jobId,
      progressPercentage: args.progressPercentage,
      currentStage: args.currentStage,
      estimatedCompletionSeconds: args.estimatedCompletionSeconds,
      stageProgress: args.stageProgress,
      updatedAt: Date.now(),
    });
  },
});
```

#### 2. Error Update Function
```typescript
export const updateJobError = mutation({
  args: {
    twitchId: v.string(),
    jobId: v.string(),
    errorType: v.string(),
    errorMessage: v.string(),
    errorDetails: v.optional(v.string()),
    isRetryable: v.boolean(),
  },
  handler: async (ctx, args) => {
    const video = await ctx.db
      .query("videos")
      .withIndex("by_twitch_id", (q) => q.eq("twitchId", args.twitchId))
      .first();
    
    if (!video) {
      throw new Error(`Video not found: ${args.twitchId}`);
    }
    
    await ctx.db.patch(video._id, {
      jobId: args.jobId,
      status: "Failed",
      errorType: args.errorType,
      errorMessage: args.errorMessage,
      errorDetails: args.errorDetails,
      isRetryable: args.isRetryable,
      processingCompletedAt: Date.now(),
      updatedAt: Date.now(),
    });
  },
});
```

#### 3. Job Completion Function
```typescript
export const completeJob = mutation({
  args: {
    twitchId: v.string(),
    jobId: v.string(),
    urls: v.optional(v.object({})),
  },
  handler: async (ctx, args) => {
    const video = await ctx.db
      .query("videos")
      .withIndex("by_twitch_id", (q) => q.eq("twitchId", args.twitchId))
      .first();
    
    if (!video) {
      throw new Error(`Video not found: ${args.twitchId}`);
    }
    
    await ctx.db.patch(video._id, {
      jobId: args.jobId,
      status: "Completed",
      progressPercentage: 100,
      currentStage: "Completed",
      estimatedCompletionSeconds: 0,
      processingCompletedAt: Date.now(),
      updatedAt: Date.now(),
      ...args.urls, // Spread any URL updates
    });
  },
});
```

#### 4. Get Job by ID Function
```typescript
export const getJobById = query({
  args: { jobId: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("videos")
      .withIndex("by_job_id", (q) => q.eq("jobId", args.jobId))
      .first();
  },
});
```

## Migration Strategy

### Phase 1: Add New Fields (Non-breaking)
1. Add all new optional fields to the schema
2. Deploy schema changes
3. Existing records will have `undefined` for new fields

### Phase 2: Update Backend Integration
1. Update Python backend to use new progress update functions
2. Maintain backward compatibility with existing status updates
3. Test with new async API endpoints

### Phase 3: Frontend Updates
1. Update frontend to consume new progress fields
2. Implement real-time progress UI components
3. Add error handling for new error types

### Phase 4: Cleanup (Optional)
1. Remove old status update patterns if no longer needed
2. Add data validation rules
3. Optimize indexes based on usage patterns

## Backward Compatibility

All new fields are optional, ensuring backward compatibility with:
- Existing frontend code
- Current API endpoints
- Existing database records

The old synchronous API can continue to work alongside the new async API during the transition period.

## Status Values

The `status` field will continue to use these values:
- `"Queued"` - Job queued for processing
- `"Downloading"` - Video download in progress
- `"Fetching chat"` - Chat data download
- `"Transcribing"` - Audio transcription
- `"Analyzing"` - Sentiment analysis
- `"Finding highlights"` - Highlight detection
- `"Completed"` - Processing finished successfully
- `"Failed"` - Processing failed

## Error Types

The `errorType` field will use these standardized values:
- `"network_error"` - Network connectivity issues
- `"invalid_video_url"` - Invalid or inaccessible video URL
- `"video_too_large"` - Video exceeds size limits
- `"processing_timeout"` - Processing took too long
- `"insufficient_resources"` - System resource constraints
- `"external_service_error"` - Third-party service failures
- `"unknown_error"` - Unexpected errors

---

**Next Steps:**
1. Implement schema changes in Convex
2. Deploy new Convex functions
3. Test with async API endpoints
4. Update frontend to use new fields
