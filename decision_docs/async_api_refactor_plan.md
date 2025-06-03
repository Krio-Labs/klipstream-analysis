# 🚀 KlipStream Analysis API - Async Refactor Implementation Plan

## 📊 Current Codebase Analysis

### 🔧 Technology Stack
- **Framework**: Google Cloud Functions with `functions-framework`
- **Language**: Python 3.10 with async/await support
- **Database**: Convex (real-time database)
- **Storage**: Google Cloud Storage
- **Deployment**: Google Cloud Run
- **Current API**: Single synchronous endpoint

### 🏗️ Current API Structure
The system currently has **3 endpoints** using `@functions_framework.http`:

1. **`run_pipeline`** (main endpoint) - Processes entire video analysis synchronously
2. **`list_files`** - Debug utility for listing files
3. **`list_output_files`** - Debug utility for output files

### 📊 Current Data Flow
```
Frontend Request → Cloud Run → run_integrated_pipeline() → 
Raw Pipeline (2-4 min) → Analysis Pipeline (1-3 min) → Response
```

### 🗄️ Database Schema (Convex)
Current `videos` table fields:
- `_id` (Convex ID)
- `twitchId` (Twitch video ID)
- `title` (video title)
- `status` (processing status)
- `transcriptUrl`, `chatUrl`, `audiowaveUrl`, `transcriptAnalysisUrl`, `transcriptWordUrl`

### 📈 Status Tracking System
Current status flow: `Queued` → `Downloading` → `Fetching chat` → `Transcribing` → `Analyzing` → `Finding highlights` → `Completed`/`Failed`

## 🎯 Detailed Implementation Plan

## Phase 1: Foundation & Async Job System (Week 1-2)

### Task 1.1: Add FastAPI Dependencies
```bash
# Add to requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
```

### Task 1.2: Create New API Structure
**Files to create:**
- `api/` (new directory)
  - `__init__.py`
  - `main.py` (FastAPI app)
  - `models.py` (Pydantic models)
  - `routes/` 
    - `__init__.py`
    - `analysis.py` (analysis endpoints)
    - `status.py` (status endpoints)
  - `services/`
    - `__init__.py`
    - `job_manager.py` (background job management)
    - `status_manager.py` (status tracking)

### Task 1.3: Database Schema Extension
**New fields to add to Convex `videos` table:**
```typescript
// In Convex schema
export default defineSchema({
  videos: defineTable({
    // Existing fields...
    twitchId: v.string(),
    status: v.string(),
    
    // New fields for async processing
    jobId: v.optional(v.string()),
    progressPercentage: v.optional(v.number()),
    currentStage: v.optional(v.string()),
    estimatedCompletionSeconds: v.optional(v.number()),
    errorType: v.optional(v.string()),
    errorMessage: v.optional(v.string()),
    isRetryable: v.optional(v.boolean()),
    retryCount: v.optional(v.number()),
    createdAt: v.optional(v.number()),
    updatedAt: v.optional(v.number()),
    processingStartedAt: v.optional(v.number()),
    processingCompletedAt: v.optional(v.number()),
  })
});
```

### Task 1.4: Create Job Management System
**File: `api/services/job_manager.py`**
- `JobManager` class for background task management
- `AnalysisJob` dataclass for job state
- Integration with existing pipeline functions

### Task 1.5: Update Convex Integration
**File: `utils/convex_client_updated.py`**
- Add methods for job tracking
- Add progress update methods
- Add error handling methods

## Phase 2: Core API Endpoints (Week 2-3)

### Task 2.1: Implement Standardized Response Models
**File: `api/models.py`**
```python
class AnalysisRequest(BaseModel):
    url: str
    callback_url: Optional[str] = None

class ProgressInfo(BaseModel):
    percentage: float
    current_stage: str
    estimated_completion_seconds: Optional[int]

class AnalysisResponse(BaseModel):
    status: str
    message: str
    job_id: str
    video_id: str
    progress: ProgressInfo
    # ... other fields
```

### Task 2.2: Create Main Analysis Endpoint
**File: `api/routes/analysis.py`**
```python
@router.post("/analysis")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    # Generate job ID
    # Create analysis record
    # Start background processing
    # Return immediate response
```

### Task 2.3: Create Status Endpoints
**File: `api/routes/status.py`**
```python
@router.get("/analysis/{job_id}/status")
async def get_analysis_status(job_id: str):
    # Return current status

@router.get("/analysis/{job_id}/stream")
async def stream_analysis_status(job_id: str):
    # Server-Sent Events implementation
```

### Task 2.4: Background Task Integration
**Modify existing pipeline functions:**
- `raw_pipeline/processor.py` - Add progress callbacks
- `analysis_pipeline/processor.py` - Add progress callbacks
- `main.py` - Refactor `run_integrated_pipeline` for background use

## Phase 3: Real-time Updates & Error Handling (Week 3-4)

### Task 3.1: Implement Server-Sent Events
**File: `api/routes/status.py`**
```python
@router.get("/analysis/{job_id}/stream")
async def stream_analysis_status(job_id: str):
    async def event_stream():
        # Stream real-time updates
        yield f"data: {json.dumps(status_data)}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/plain")
```

### Task 3.2: Enhanced Error Classification
**File: `api/services/error_handler.py`**
```python
class ErrorType(Enum):
    NETWORK_ERROR = "network_error"
    INVALID_VIDEO_URL = "invalid_video_url"
    # ... other error types

class ErrorHandler:
    @staticmethod
    def classify_error(exception: Exception) -> AnalysisError:
        # Comprehensive error classification
```

### Task 3.3: Retry Mechanism
**File: `api/services/retry_manager.py`**
```python
class RetryManager:
    @staticmethod
    async def retry_with_backoff(func, max_retries=3):
        # Exponential backoff retry logic
```

### Task 3.4: Progress Tracking Integration
**Modify pipeline processors to send progress updates:**
- Add progress callbacks to download functions
- Add progress callbacks to transcription
- Add progress callbacks to analysis steps

## Phase 4: Deployment & Migration (Week 4-5)

### Task 4.1: Update Dockerfile
```dockerfile
# Add FastAPI dependencies
# Update CMD to use uvicorn instead of functions-framework
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Task 4.2: Create Migration Strategy
**Approach: Gradual Migration**
1. Deploy new FastAPI endpoints alongside existing functions-framework
2. Update frontend to use new endpoints
3. Keep old endpoint for backward compatibility
4. Remove old endpoint after migration complete

### Task 4.3: Update Cloud Run Configuration
**File: `deploy_cloud_run_simple.sh`**
- Update deployment script for FastAPI
- Add health check endpoints
- Update environment variables

### Task 4.4: Testing & Validation
- Create test scripts for new endpoints
- Validate backward compatibility
- Performance testing
- Load testing

## Phase 5: Advanced Features (Week 5-6)

### Task 5.1: Webhook System (Optional)
**File: `api/services/webhook_manager.py`**
```python
class WebhookManager:
    async def register_webhook(self, video_id: str, callback_url: str):
        # Register webhook for notifications
    
    async def send_status_update(self, status: StatusUpdate):
        # Send webhook notifications
```

### Task 5.2: Queue Management
**File: `api/services/queue_manager.py`**
- Implement job queuing
- Resource management
- Concurrent job limits

### Task 5.3: Monitoring & Analytics
- Add performance metrics
- Add error tracking
- Add usage analytics

## 📋 Implementation Priority & Dependencies

### Critical Path:
1. **Phase 1** → **Phase 2** → **Phase 4** (Core functionality)
2. **Phase 3** (Enhanced features)
3. **Phase 5** (Advanced features)

### Key Dependencies:
- FastAPI setup must be complete before endpoint creation
- Database schema changes must be deployed before job management
- Background task system must work before real-time updates

### Risk Mitigation:
- Keep existing endpoint functional during migration
- Implement feature flags for gradual rollout
- Comprehensive testing at each phase

## 📊 Expected Outcomes

### Performance Improvements:
- **Response Time**: < 2 seconds (vs current 5-7 minutes)
- **User Experience**: Real-time progress updates
- **Scalability**: Multiple concurrent jobs
- **Reliability**: Comprehensive error handling and retry logic

### New Capabilities:
- Immediate job submission response
- Real-time progress tracking via SSE
- Detailed error classification and recovery
- Optional webhook notifications
- Job queuing and resource management

---

**Status**: Implementation in progress
**Created**: 2024-01-15
**Last Updated**: 2024-01-15
