"""
Queue Management System for KlipStream Analysis API

This module provides queue management functionality for handling multiple
concurrent analysis jobs with priority queuing and resource management.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import uuid4

from utils.logging_setup import setup_logger

logger = setup_logger(__name__)


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class QueueStatus(Enum):
    """Queue status types"""
    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"  # No new jobs, finish existing ones
    STOPPED = "stopped"


@dataclass
class QueuedJob:
    """Represents a job in the queue"""
    job_id: str
    video_id: str
    video_url: str
    priority: JobPriority
    queued_at: datetime
    estimated_duration: int = 3600  # seconds
    retry_count: int = 0
    max_retries: int = 3
    callback_url: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class QueueMetrics:
    """Queue performance metrics"""
    total_jobs_processed: int = 0
    total_jobs_failed: int = 0
    total_jobs_retried: int = 0
    average_processing_time: float = 0.0
    queue_length: int = 0
    active_workers: int = 0
    max_workers: int = 0
    uptime_seconds: float = 0.0


class QueueManager:
    """
    Advanced queue management system for analysis jobs
    
    Features:
    - Priority-based job queuing
    - Concurrent job processing with configurable limits
    - Automatic retry mechanism
    - Queue metrics and monitoring
    - Resource management and throttling
    """
    
    def __init__(self, max_concurrent_jobs: int = 3, max_queue_size: int = 100):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queue_size = max_queue_size
        
        # Queue storage (priority queue)
        self._queue: List[QueuedJob] = []
        self._active_jobs: Dict[str, QueuedJob] = {}
        self._completed_jobs: Set[str] = set()
        self._failed_jobs: Set[str] = set()
        
        # Queue management
        self._status = QueueStatus.ACTIVE
        self._workers: Set[asyncio.Task] = set()
        self._queue_lock = asyncio.Lock()
        self._processing_lock = asyncio.Lock()
        
        # Metrics
        self._metrics = QueueMetrics(max_workers=max_concurrent_jobs)
        self._start_time = time.time()
        
        # Job processing callback
        self._job_processor = None
        
        logger.info(f"QueueManager initialized with max_concurrent_jobs={max_concurrent_jobs}, max_queue_size={max_queue_size}")
    
    def set_job_processor(self, processor_func):
        """Set the function to process jobs"""
        self._job_processor = processor_func
        logger.info("Job processor function set")
    
    async def enqueue_job(self, job: QueuedJob) -> bool:
        """
        Add a job to the queue
        
        Args:
            job: QueuedJob instance to add to queue
            
        Returns:
            bool: True if job was queued successfully, False otherwise
        """
        async with self._queue_lock:
            # Check queue size limit
            if len(self._queue) >= self.max_queue_size:
                logger.warning(f"Queue is full ({self.max_queue_size} jobs). Cannot enqueue job {job.job_id}")
                return False
            
            # Check if job already exists
            if job.job_id in [j.job_id for j in self._queue] or job.job_id in self._active_jobs:
                logger.warning(f"Job {job.job_id} already exists in queue or is being processed")
                return False
            
            # Add job to queue and sort by priority
            self._queue.append(job)
            self._queue.sort(key=lambda x: (x.priority.value, x.queued_at), reverse=True)
            
            self._metrics.queue_length = len(self._queue)
            
            logger.info(f"Job {job.job_id} queued with priority {job.priority.name}. Queue length: {len(self._queue)}")
            
            # Start processing if we have available workers
            await self._maybe_start_worker()
            
            return True
    
    async def _maybe_start_worker(self):
        """Start a new worker if we have capacity and jobs to process"""
        if (len(self._workers) < self.max_concurrent_jobs and 
            len(self._queue) > 0 and 
            self._status == QueueStatus.ACTIVE):
            
            worker_task = asyncio.create_task(self._worker())
            self._workers.add(worker_task)
            worker_task.add_done_callback(self._workers.discard)
            
            self._metrics.active_workers = len(self._workers)
            logger.info(f"Started new worker. Active workers: {len(self._workers)}")
    
    async def _worker(self):
        """Worker coroutine that processes jobs from the queue"""
        worker_id = str(uuid4())[:8]
        logger.info(f"Worker {worker_id} started")
        
        try:
            while self._status in [QueueStatus.ACTIVE, QueueStatus.DRAINING]:
                # Get next job from queue
                job = await self._get_next_job()
                if not job:
                    # No jobs available, exit worker
                    break
                
                # Process the job
                await self._process_job(job, worker_id)
                
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered error: {str(e)}")
        finally:
            self._metrics.active_workers = len(self._workers) - 1
            logger.info(f"Worker {worker_id} finished")
    
    async def _get_next_job(self) -> Optional[QueuedJob]:
        """Get the next job from the queue"""
        async with self._queue_lock:
            if not self._queue:
                return None
            
            job = self._queue.pop(0)  # Get highest priority job
            self._active_jobs[job.job_id] = job
            self._metrics.queue_length = len(self._queue)
            
            return job
    
    async def _process_job(self, job: QueuedJob, worker_id: str):
        """Process a single job"""
        start_time = time.time()
        logger.info(f"Worker {worker_id} processing job {job.job_id}")
        
        try:
            if self._job_processor:
                # Call the job processor function
                await self._job_processor(job)
                
                # Job completed successfully
                self._completed_jobs.add(job.job_id)
                self._metrics.total_jobs_processed += 1
                
                processing_time = time.time() - start_time
                self._update_average_processing_time(processing_time)
                
                logger.info(f"Job {job.job_id} completed successfully in {processing_time:.2f}s")
            else:
                logger.error(f"No job processor set. Cannot process job {job.job_id}")
                
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {str(e)}")
            
            # Handle job failure and retry logic
            await self._handle_job_failure(job, str(e))
            
        finally:
            # Remove from active jobs
            async with self._processing_lock:
                self._active_jobs.pop(job.job_id, None)
    
    async def _handle_job_failure(self, job: QueuedJob, error_message: str):
        """Handle job failure and retry logic"""
        job.retry_count += 1
        
        if job.retry_count <= job.max_retries:
            # Retry the job
            job.queued_at = datetime.utcnow()
            await self.enqueue_job(job)
            self._metrics.total_jobs_retried += 1
            logger.info(f"Job {job.job_id} queued for retry ({job.retry_count}/{job.max_retries})")
        else:
            # Job failed permanently
            self._failed_jobs.add(job.job_id)
            self._metrics.total_jobs_failed += 1
            logger.error(f"Job {job.job_id} failed permanently after {job.max_retries} retries")
    
    def _update_average_processing_time(self, processing_time: float):
        """Update the average processing time metric"""
        total_processed = self._metrics.total_jobs_processed
        if total_processed == 1:
            self._metrics.average_processing_time = processing_time
        else:
            # Calculate running average
            current_avg = self._metrics.average_processing_time
            self._metrics.average_processing_time = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )
    
    async def get_queue_status(self) -> Dict:
        """Get current queue status and metrics"""
        async with self._queue_lock:
            self._metrics.uptime_seconds = time.time() - self._start_time
            
            return {
                "status": self._status.value,
                "queue_length": len(self._queue),
                "active_jobs": len(self._active_jobs),
                "active_workers": len(self._workers),
                "max_workers": self.max_concurrent_jobs,
                "max_queue_size": self.max_queue_size,
                "metrics": {
                    "total_processed": self._metrics.total_jobs_processed,
                    "total_failed": self._metrics.total_jobs_failed,
                    "total_retried": self._metrics.total_jobs_retried,
                    "average_processing_time": round(self._metrics.average_processing_time, 2),
                    "uptime_seconds": round(self._metrics.uptime_seconds, 2),
                },
                "queue_jobs": [
                    {
                        "job_id": job.job_id,
                        "video_id": job.video_id,
                        "priority": job.priority.name,
                        "queued_at": job.queued_at.isoformat(),
                        "retry_count": job.retry_count
                    }
                    for job in self._queue
                ],
                "active_job_ids": list(self._active_jobs.keys())
            }
    
    async def pause_queue(self):
        """Pause the queue (stop processing new jobs)"""
        self._status = QueueStatus.PAUSED
        logger.info("Queue paused")
    
    async def resume_queue(self):
        """Resume the queue"""
        self._status = QueueStatus.ACTIVE
        logger.info("Queue resumed")
        
        # Start workers if needed
        for _ in range(min(len(self._queue), self.max_concurrent_jobs - len(self._workers))):
            await self._maybe_start_worker()
    
    async def drain_queue(self):
        """Drain the queue (finish existing jobs, don't accept new ones)"""
        self._status = QueueStatus.DRAINING
        logger.info("Queue draining - finishing existing jobs")
    
    async def stop_queue(self):
        """Stop the queue and cancel all workers"""
        self._status = QueueStatus.STOPPED
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        logger.info("Queue stopped")
    
    async def remove_job(self, job_id: str) -> bool:
        """Remove a job from the queue"""
        async with self._queue_lock:
            for i, job in enumerate(self._queue):
                if job.job_id == job_id:
                    self._queue.pop(i)
                    self._metrics.queue_length = len(self._queue)
                    logger.info(f"Removed job {job_id} from queue")
                    return True
            
            logger.warning(f"Job {job_id} not found in queue")
            return False
    
    def get_job_position(self, job_id: str) -> Optional[int]:
        """Get the position of a job in the queue (0-indexed)"""
        for i, job in enumerate(self._queue):
            if job.job_id == job_id:
                return i
        return None
