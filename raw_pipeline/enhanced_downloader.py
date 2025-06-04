"""
Enhanced Downloader Module with Comprehensive Error Handling and Monitoring

This module provides enhanced downloading capabilities with:
- Detailed process monitoring
- Comprehensive error capture
- Resource usage tracking
- Predictive failure detection
"""

import asyncio
import subprocess
import re
import time
import shutil
import psutil
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from utils.config import (
    RAW_VIDEOS_DIR,
    RAW_AUDIO_DIR,
    BINARY_PATHS,
    DOWNLOADS_DIR,
    TEMP_DIR
)
from utils.logging_setup import setup_logger
from .timeout_manager import AdaptiveTimeoutManager, TimeoutConfig, TimeoutAwareProcess, ProgressInfo as TimeoutProgressInfo

# Set up logger
logger = setup_logger("enhanced_downloader", "enhanced_downloader.log")

class ProcessState(Enum):
    """Process execution states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class FailureType(Enum):
    """Types of failures that can occur"""
    PROCESS_TIMEOUT = "process_timeout"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DISK_SPACE = "disk_space"
    NETWORK_ERROR = "network_error"
    PROCESS_CRASH = "process_crash"
    FILE_SYSTEM_ERROR = "file_system_error"
    UNKNOWN = "unknown"

@dataclass
class ProcessMetrics:
    """Process resource usage metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: int = 0
    disk_io_write: int = 0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

@dataclass
class DownloadProgress:
    """Download progress information"""
    percentage: float
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    speed_mbps: float = 0.0
    eta_seconds: Optional[int] = None
    current_segment: Optional[str] = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ProcessContext:
    """Enhanced process execution context"""
    job_id: str
    video_id: str
    start_time: datetime
    timeout_seconds: int
    max_memory_mb: int
    process_id: Optional[int] = None
    state: ProcessState = ProcessState.INITIALIZING
    metrics_history: List[ProcessMetrics] = field(default_factory=list)
    progress: Optional[DownloadProgress] = None
    
@dataclass
class ErrorContext:
    """Comprehensive error context for debugging"""
    error_id: str
    timestamp: datetime
    process_context: ProcessContext
    system_metrics: ProcessMetrics
    stdout_capture: str
    stderr_capture: str
    process_exit_code: Optional[int]
    execution_duration: float
    failure_stage: str
    resource_usage_at_failure: Dict[str, Any]

@dataclass
class DetailedErrorResponse:
    """Detailed error response with actionable information"""
    error_id: str
    error_type: FailureType
    error_message: str
    user_friendly_message: str
    technical_details: Dict[str, Any]
    suggested_actions: List[str]
    is_retryable: bool
    retry_delay_seconds: int
    support_reference: str
    context: ErrorContext

class ProcessMonitor:
    """Monitor process resource usage and health"""
    
    def __init__(self, context: ProcessContext):
        self.context = context
        self.monitoring = False
        self.alert_thresholds = {
            'memory_percent': 80.0,
            'cpu_percent': 90.0,
            'execution_time': context.timeout_seconds * 0.8
        }
    
    async def start_monitoring(self, process: asyncio.subprocess.Process):
        """Start monitoring the process"""
        self.monitoring = True
        self.context.process_id = process.pid
        
        try:
            psutil_process = psutil.Process(process.pid)
            
            while self.monitoring and process.returncode is None:
                try:
                    # Collect metrics
                    metrics = await self.collect_metrics(psutil_process)
                    self.context.metrics_history.append(metrics)
                    
                    # Check for alerts
                    await self.check_alert_conditions(metrics)
                    
                    # Predict potential failures
                    await self.predict_failures(metrics)
                    
                    await asyncio.sleep(5)  # Monitor every 5 seconds
                    
                except psutil.NoSuchProcess:
                    logger.warning(f"Process {process.pid} no longer exists")
                    break
                except Exception as e:
                    logger.error(f"Error monitoring process: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Failed to start process monitoring: {e}")
    
    async def collect_metrics(self, psutil_process) -> ProcessMetrics:
        """Collect comprehensive process metrics"""
        try:
            # Get process info
            memory_info = psutil_process.memory_info()
            cpu_percent = psutil_process.cpu_percent()
            
            # Get I/O stats if available
            try:
                io_counters = psutil_process.io_counters()
                disk_read = io_counters.read_bytes
                disk_write = io_counters.write_bytes
            except (psutil.AccessDenied, AttributeError):
                disk_read = disk_write = 0
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            memory_percent = (memory_info.rss / system_memory.total) * 100
            
            return ProcessMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_mb=memory_info.rss / (1024 * 1024),
                memory_percent=memory_percent,
                disk_io_read=disk_read,
                disk_io_write=disk_write
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return ProcessMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0
            )
    
    async def check_alert_conditions(self, metrics: ProcessMetrics):
        """Check for alert conditions"""
        alerts = []
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        execution_time = (datetime.now(timezone.utc) - self.context.start_time).total_seconds()
        if execution_time > self.alert_thresholds['execution_time']:
            alerts.append(f"Long execution time: {execution_time:.0f}s")
        
        if alerts:
            logger.warning(f"Process alerts for job {self.context.job_id}: {', '.join(alerts)}")
    
    async def predict_failures(self, metrics: ProcessMetrics):
        """Predict potential failures based on metrics trends"""
        if len(self.context.metrics_history) < 3:
            return
        
        # Check memory trend
        recent_metrics = self.context.metrics_history[-3:]
        memory_trend = [m.memory_percent for m in recent_metrics]
        
        if len(memory_trend) >= 2:
            memory_increase_rate = (memory_trend[-1] - memory_trend[0]) / len(memory_trend)
            if memory_increase_rate > 5.0:  # 5% increase per measurement
                logger.warning(f"Rapid memory increase detected: {memory_increase_rate:.1f}% per measurement")
    
    def stop_monitoring(self):
        """Stop monitoring the process"""
        self.monitoring = False

class ErrorAnalyzer:
    """Analyze errors and provide detailed diagnostics"""
    
    @staticmethod
    async def analyze_error(error: Exception, context: ProcessContext, 
                          stdout_capture: str, stderr_capture: str,
                          process_exit_code: Optional[int]) -> DetailedErrorResponse:
        """Analyze error and create detailed response"""
        
        error_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        execution_duration = (timestamp - context.start_time).total_seconds()
        
        # Determine error type and details
        error_type, failure_analysis = ErrorAnalyzer._classify_error(
            error, stdout_capture, stderr_capture, process_exit_code, context
        )
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=timestamp,
            process_context=context,
            system_metrics=context.metrics_history[-1] if context.metrics_history else None,
            stdout_capture=stdout_capture[-2000:],  # Last 2000 chars
            stderr_capture=stderr_capture[-2000:],   # Last 2000 chars
            process_exit_code=process_exit_code,
            execution_duration=execution_duration,
            failure_stage=context.state.value,
            resource_usage_at_failure=ErrorAnalyzer._get_resource_summary(context)
        )
        
        return DetailedErrorResponse(
            error_id=error_id,
            error_type=error_type,
            error_message=str(error),
            user_friendly_message=failure_analysis['user_message'],
            technical_details=failure_analysis['technical_details'],
            suggested_actions=failure_analysis['suggested_actions'],
            is_retryable=failure_analysis['is_retryable'],
            retry_delay_seconds=failure_analysis['retry_delay'],
            support_reference=f"DL-{timestamp.strftime('%Y%m%d-%H%M%S')}-{error_id[:8]}",
            context=error_context
        )
    
    @staticmethod
    def _classify_error(error: Exception, stdout: str, stderr: str, 
                       exit_code: Optional[int], context: ProcessContext) -> Tuple[FailureType, Dict]:
        """Classify the error and provide analysis"""
        
        # Check for timeout
        if "timed out" in str(error).lower() or isinstance(error, asyncio.TimeoutError):
            return FailureType.PROCESS_TIMEOUT, {
                'user_message': 'The download process took too long and was cancelled. This may be due to a large file or slow network.',
                'technical_details': {
                    'timeout_seconds': context.timeout_seconds,
                    'execution_duration': (datetime.utcnow() - context.start_time).total_seconds(),
                    'progress_at_failure': context.progress.percentage if context.progress else 0
                },
                'suggested_actions': [
                    'Retry with a longer timeout',
                    'Check network connectivity',
                    'Try downloading at a lower quality'
                ],
                'is_retryable': True,
                'retry_delay': 60
            }
        
        # Check for memory issues
        if any(keyword in stderr.lower() for keyword in ['memory', 'out of memory', 'oom', 'heap', 'allocation']):
            peak_memory = max([m.memory_mb for m in context.metrics_history]) if context.metrics_history else 0

            return FailureType.MEMORY_EXHAUSTION, {
                'user_message': f'The download process ran out of memory (peak: {peak_memory:.1f}MB). Try a lower quality setting.',
                'technical_details': {
                    'max_memory_mb': context.max_memory_mb,
                    'peak_memory_usage': peak_memory,
                    'memory_utilization': (peak_memory / context.max_memory_mb * 100) if context.max_memory_mb > 0 else 0,
                    'suggested_quality': 'Try 480p or 360p quality'
                },
                'suggested_actions': [
                    'Retry with lower quality (480p or 360p)',
                    'Use progressive quality fallback',
                    'Increase memory allocation if possible',
                    'Try downloading during off-peak hours'
                ],
                'is_retryable': True,
                'retry_delay': 60
            }
        
        # Check for network issues
        if any(keyword in stderr.lower() for keyword in ['network', 'connection', 'timeout', 'dns']):
            return FailureType.NETWORK_ERROR, {
                'user_message': 'Network connectivity issues prevented the download from completing.',
                'technical_details': {
                    'stderr_snippet': stderr[-500:],
                    'exit_code': exit_code
                },
                'suggested_actions': [
                    'Check internet connectivity',
                    'Retry the download',
                    'Contact support if the issue persists'
                ],
                'is_retryable': True,
                'retry_delay': 30
            }
        
        # Check for file system issues
        if any(keyword in stderr.lower() for keyword in ['disk', 'space', 'permission', 'file']):
            return FailureType.FILE_SYSTEM_ERROR, {
                'user_message': 'File system issues prevented the download from completing.',
                'technical_details': {
                    'stderr_snippet': stderr[-500:],
                    'exit_code': exit_code
                },
                'suggested_actions': [
                    'Check available disk space',
                    'Verify file permissions',
                    'Contact support'
                ],
                'is_retryable': False,
                'retry_delay': 0
            }
        
        # Default case - unknown error
        return FailureType.UNKNOWN, {
            'user_message': 'An unexpected error occurred during the download process.',
            'technical_details': {
                'error_message': str(error),
                'stdout_snippet': stdout[-500:],
                'stderr_snippet': stderr[-500:],
                'exit_code': exit_code
            },
            'suggested_actions': [
                'Retry the download',
                'Contact support with the error details'
            ],
            'is_retryable': True,
            'retry_delay': 60
        }
    
    @staticmethod
    def _get_resource_summary(context: ProcessContext) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not context.metrics_history:
            return {}
        
        metrics = context.metrics_history
        return {
            'peak_memory_mb': max([m.memory_mb for m in metrics]),
            'avg_cpu_percent': sum([m.cpu_percent for m in metrics]) / len(metrics),
            'total_disk_read_mb': max([m.disk_io_read for m in metrics]) / (1024 * 1024),
            'total_disk_write_mb': max([m.disk_io_write for m in metrics]) / (1024 * 1024),
            'execution_duration_seconds': (datetime.now(timezone.utc) - context.start_time).total_seconds()
        }


class EnhancedTwitchDownloader:
    """Enhanced Twitch downloader with comprehensive monitoring and error handling"""

    def __init__(self):
        """Initialize the enhanced downloader"""
        self.active_contexts = {}
        self.error_analyzer = ErrorAnalyzer()

        # Quality fallback configuration
        self.quality_levels = [
            {"quality": "720p", "max_memory_mb": 8192, "timeout_multiplier": 1.0},
            {"quality": "480p", "max_memory_mb": 4096, "timeout_multiplier": 0.8},
            {"quality": "360p", "max_memory_mb": 2048, "timeout_multiplier": 0.6},
            {"quality": "worst", "max_memory_mb": 1024, "timeout_multiplier": 0.5}
        ]

    async def download_video_with_progressive_fallback(self, video_id: str, job_id: str = None) -> Path:
        """Download video with progressive quality fallback on memory/timeout issues"""

        if job_id is None:
            job_id = str(uuid.uuid4())

        logger.info(f"Starting progressive quality fallback download for {video_id} (job: {job_id})")

        # Try each quality level in order
        for attempt, quality_config in enumerate(self.quality_levels):
            quality = quality_config["quality"]
            max_memory = quality_config["max_memory_mb"]
            timeout_multiplier = quality_config["timeout_multiplier"]

            logger.info(f"Attempt {attempt + 1}/{len(self.quality_levels)}: Trying quality {quality} (max memory: {max_memory}MB)")

            try:
                result = await self._download_with_quality(
                    video_id=video_id,
                    job_id=f"{job_id}_q{attempt}",
                    quality=quality,
                    max_memory_mb=max_memory,
                    timeout_multiplier=timeout_multiplier
                )

                logger.info(f"✅ Successfully downloaded {video_id} at {quality} quality")
                return result

            except Exception as e:
                error_msg = str(e).lower()

                # Check if this is a memory or timeout related error
                is_memory_error = any(keyword in error_msg for keyword in ['memory', 'oom', 'out of memory'])
                is_timeout_error = any(keyword in error_msg for keyword in ['timeout', 'timed out'])
                is_size_error = any(keyword in error_msg for keyword in ['too large', 'file size', 'disk space'])

                if is_memory_error or is_timeout_error or is_size_error:
                    logger.warning(f"❌ Quality {quality} failed with recoverable error: {e}")

                    if attempt < len(self.quality_levels) - 1:
                        next_quality = self.quality_levels[attempt + 1]["quality"]
                        logger.info(f"🔄 Falling back to {next_quality} quality...")
                        continue
                    else:
                        logger.error(f"💥 All quality levels failed. Last error: {e}")
                        raise RuntimeError(f"Progressive fallback failed - all quality levels exhausted. Final error: {e}")
                else:
                    # Non-recoverable error, don't try other qualities
                    logger.error(f"💥 Non-recoverable error at {quality} quality: {e}")
                    raise e

        # Should never reach here
        raise RuntimeError("Progressive fallback failed - no quality levels available")

    async def download_video_with_monitoring(self, video_id: str, job_id: str = None) -> Path:
        """Download video with comprehensive monitoring and error handling"""

        # Use progressive fallback by default
        return await self.download_video_with_progressive_fallback(video_id, job_id)

    async def _download_with_quality(self, video_id: str, job_id: str, quality: str,
                                   max_memory_mb: int, timeout_multiplier: float) -> Path:
        """Download video with specific quality and resource constraints"""

        # Create execution context with quality-specific settings
        base_timeout = 30 * 60  # 30 minutes base
        adjusted_timeout = int(base_timeout * timeout_multiplier)

        context = ProcessContext(
            job_id=job_id,
            video_id=video_id,
            start_time=datetime.now(timezone.utc),
            timeout_seconds=adjusted_timeout,
            max_memory_mb=max_memory_mb
        )

        # Create adaptive timeout manager with quality-specific settings
        timeout_config = TimeoutConfig(
            base_timeout_seconds=adjusted_timeout,
            max_timeout_seconds=int(adjusted_timeout * 1.5),  # 50% buffer
            progress_stall_timeout=min(5 * 60, adjusted_timeout // 6),  # 5 min or 1/6 of total
            adaptive_scaling=True
        )
        timeout_manager = AdaptiveTimeoutManager(timeout_config)

        # Add memory monitoring callback
        def memory_alert_callback(event):
            if event.reason.value == "memory_exhaustion":
                logger.warning(f"Memory exhaustion detected for quality {quality}: {event.message}")

        timeout_manager.register_timeout_callback(memory_alert_callback)

        self.active_contexts[job_id] = context

        try:
            logger.info(f"Starting enhanced video download for {video_id} (job: {job_id})")

            # Define output path
            video_file = RAW_VIDEOS_DIR / f"{video_id}.mp4"

            # Check if file already exists
            if video_file.exists():
                logger.info(f"Video file already exists: {video_file}")
                context.state = ProcessState.COMPLETED
                return video_file

            # Prepare download command with quality-specific settings
            threads = self._get_optimal_threads(quality, max_memory_mb)

            command = [
                BINARY_PATHS["twitch_downloader"],
                "videodownload",
                "--id", video_id,
                "-o", str(video_file),
                "--quality", quality,
                "--threads", str(threads),
                "--temp-path", str(TEMP_DIR)
            ]

            logger.info(f"Download command: {' '.join(command)}")

            # Execute download with monitoring and timeout management
            result = await self._execute_monitored_download(context, command, video_file, timeout_manager)

            context.state = ProcessState.COMPLETED
            logger.info(f"Successfully downloaded video {video_id}")
            return result

        except Exception as e:
            context.state = ProcessState.FAILED
            logger.error(f"Failed to download video {video_id}: {e}")

            # Analyze error and provide detailed response
            error_response = await self.error_analyzer.analyze_error(
                e, context, "", "", None
            )

            # Log detailed error information
            logger.error(f"Detailed error analysis: {error_response.support_reference}")
            logger.error(f"Error type: {error_response.error_type.value}")
            logger.error(f"User message: {error_response.user_friendly_message}")
            logger.error(f"Technical details: {json.dumps(error_response.technical_details, indent=2)}")
            logger.error(f"Suggested actions: {error_response.suggested_actions}")

            # Re-raise with enhanced error message
            raise RuntimeError(f"Enhanced download failed: {error_response.user_friendly_message} (Ref: {error_response.support_reference})")

        finally:
            # Cleanup
            if job_id in self.active_contexts:
                del self.active_contexts[job_id]

    async def _execute_monitored_download(self, context: ProcessContext, command: List[str], output_file: Path, timeout_manager: AdaptiveTimeoutManager) -> Path:
        """Execute download with comprehensive monitoring"""

        # Initialize progress tracking
        context.progress = DownloadProgress(percentage=0.0)
        context.state = ProcessState.RUNNING

        # Start the process with timeout management
        try:
            timeout_process = TimeoutAwareProcess(timeout_manager)
            process = await timeout_process.start_process(command)
        except Exception as e:
            raise RuntimeError(f"Failed to start download process: {e}")

        # Start monitoring
        monitor = ProcessMonitor(context)
        monitor_task = asyncio.create_task(monitor.start_monitoring(process))

        # Capture output
        stdout_capture = ""
        stderr_capture = ""

        try:
            # Process output with timeout
            while True:
                try:
                    # Read with timeout
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=30)

                    if not line:
                        # Check for stderr
                        try:
                            err_line = await asyncio.wait_for(process.stderr.readline(), timeout=5)
                            if err_line:
                                err_str = err_line.decode('utf-8', errors='replace').strip()
                                stderr_capture += err_str + "\n"
                                logger.info(f"Process stderr: {err_str}")
                        except asyncio.TimeoutError:
                            pass
                        break

                    line_str = line.decode('utf-8', errors='replace').strip()
                    stdout_capture += line_str + "\n"

                    # Update progress
                    await self._update_progress(context, line_str, timeout_manager)

                    # Log important output
                    if any(keyword in line_str.lower() for keyword in ['error', 'warning', 'failed', 'status']):
                        logger.info(f"Process output: {line_str}")

                except asyncio.TimeoutError:
                    # Check if process is still alive
                    if process.returncode is not None:
                        break

                    # Check for timeout using timeout manager
                    timeout_event = timeout_manager.check_timeouts(context.progress)
                    if timeout_event:
                        logger.error(f"Download timeout: {timeout_event.message}")
                        await timeout_process.cleanup()
                        raise RuntimeError(f"Download timeout: {timeout_event.message}")

                    logger.warning("No output for 30 seconds, checking process health...")
                    continue

            # Wait for process completion
            return_code = await process.wait()

            # Read any remaining stderr
            remaining_stderr = await process.stderr.read()
            if remaining_stderr:
                stderr_str = remaining_stderr.decode('utf-8', errors='replace')
                stderr_capture += stderr_str

            if return_code != 0:
                error_msg = f"Download process failed with exit code {return_code}"
                if stderr_capture.strip():
                    error_msg += f": {stderr_capture.strip()}"

                # Analyze the error
                error_response = await self.error_analyzer.analyze_error(
                    RuntimeError(error_msg), context, stdout_capture, stderr_capture, return_code
                )

                raise RuntimeError(f"Download failed: {error_response.user_friendly_message}")

            # Verify output file exists
            if not output_file.exists():
                raise RuntimeError(f"Download completed but output file not found: {output_file}")

            logger.info(f"Download completed successfully: {output_file}")
            return output_file

        finally:
            # Stop monitoring
            monitor.stop_monitoring()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Cleanup timeout process
            await timeout_process.cleanup()

    async def _update_progress(self, context: ProcessContext, line: str, timeout_manager: AdaptiveTimeoutManager):
        """Update download progress from process output"""

        # Look for various progress patterns
        progress_patterns = [
            r'Downloaded:\s+(\d+(?:\.\d+)?)%',
            r'Progress:\s+(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%\s+complete',
            r'Downloading:\s+(\d+(?:\.\d+)?)%'
        ]

        for pattern in progress_patterns:
            match = re.search(pattern, line)
            if match:
                progress = float(match.group(1))
                context.progress.percentage = progress
                context.progress.last_update = datetime.now(timezone.utc)

                # Update timeout manager with progress
                timeout_progress = TimeoutProgressInfo(
                    percentage=progress,
                    bytes_downloaded=context.progress.bytes_downloaded,
                    total_bytes=context.progress.total_bytes,
                    last_update=context.progress.last_update,
                    download_speed_mbps=context.progress.speed_mbps
                )
                timeout_manager.update_progress(timeout_progress)

                # Log progress every 10%
                if progress % 10 == 0 or progress > 95:
                    logger.info(f"Download progress: {progress}%")
                    # Also log timeout status at major milestones
                    timeout_status = timeout_manager.get_timeout_status()
                    logger.info(f"Timeout status: {timeout_status['remaining_seconds']}s remaining, risk: {timeout_status['timeout_risk_level']}")
                return

        # Look for segment information
        segment_match = re.search(r'Downloading segment (\d+)/(\d+)', line)
        if segment_match:
            current, total = map(int, segment_match.groups())
            progress = min(100, (current / total) * 100)
            context.progress.percentage = progress
            context.progress.current_segment = f"{current}/{total}"
            context.progress.last_update = datetime.now(timezone.utc)

            # Update timeout manager with segment progress
            timeout_progress = TimeoutProgressInfo(
                percentage=progress,
                bytes_downloaded=context.progress.bytes_downloaded,
                total_bytes=context.progress.total_bytes,
                last_update=context.progress.last_update,
                download_speed_mbps=context.progress.speed_mbps
            )
            timeout_manager.update_progress(timeout_progress)

            if current % 50 == 0:  # Log every 50 segments
                logger.info(f"Download progress: {progress}% (segment {current}/{total})")

    def get_active_downloads(self) -> Dict[str, ProcessContext]:
        """Get information about active downloads"""
        return self.active_contexts.copy()

    def get_download_status(self, job_id: str) -> Optional[ProcessContext]:
        """Get status of a specific download"""
        return self.active_contexts.get(job_id)

    def _get_optimal_threads(self, quality: str, max_memory_mb: int) -> int:
        """Calculate optimal thread count based on quality and memory constraints"""

        # Base thread counts by quality
        quality_threads = {
            "720p": 8,
            "480p": 6,
            "360p": 4,
            "worst": 2
        }

        base_threads = quality_threads.get(quality, 4)

        # Adjust based on memory constraints
        if max_memory_mb <= 1024:  # 1GB
            return min(base_threads, 2)
        elif max_memory_mb <= 2048:  # 2GB
            return min(base_threads, 4)
        elif max_memory_mb <= 4096:  # 4GB
            return min(base_threads, 6)
        else:  # 8GB+
            return base_threads

    def get_quality_recommendation(self, available_memory_mb: int, video_duration_minutes: int = None) -> str:
        """Get recommended quality based on available resources"""

        # Factor in video duration if available
        memory_factor = 1.0
        if video_duration_minutes:
            if video_duration_minutes > 240:  # 4+ hours
                memory_factor = 1.5
            elif video_duration_minutes > 120:  # 2+ hours
                memory_factor = 1.3  # Increased factor for 3+ hour videos

        adjusted_memory = available_memory_mb / memory_factor

        if adjusted_memory >= 6144:  # 6GB+
            return "720p"
        elif adjusted_memory >= 3072:  # 3GB+
            return "480p"
        elif adjusted_memory >= 1536:  # 1.5GB+
            return "360p"
        else:
            return "worst"
