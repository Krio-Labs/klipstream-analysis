"""
Dynamic Resource Detection and Optimization Utility

This module detects system resources and optimizes download parameters
for maximum performance across different machines.
"""

import os
import psutil
import platform
import multiprocessing
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ResourceOptimizer:
    """
    Detects system resources and provides optimized download parameters
    """
    
    def __init__(self):
        self.system_info = self._detect_system_resources()
        self.optimization_profile = self._calculate_optimization_profile()
        
    def _detect_system_resources(self) -> Dict:
        """
        Detect comprehensive system resources
        """
        try:
            # CPU Information
            cpu_count_logical = multiprocessing.cpu_count()
            cpu_count_physical = psutil.cpu_count(logical=False) or cpu_count_logical
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory Information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk Information
            disk_usage = psutil.disk_usage('/')
            disk_free_gb = disk_usage.free / (1024**3)
            
            # Network Information (basic)
            network_stats = psutil.net_io_counters()
            
            # Platform Information
            system_platform = platform.system()
            architecture = platform.architecture()[0]
            
            # Detect if running in cloud environment
            is_cloud = self._detect_cloud_environment()
            
            # Detect storage type (SSD vs HDD) - best effort
            storage_type = self._detect_storage_type()
            
            system_info = {
                'cpu': {
                    'logical_cores': cpu_count_logical,
                    'physical_cores': cpu_count_physical,
                    'frequency_mhz': cpu_freq.current if cpu_freq else None,
                    'current_usage_percent': cpu_percent
                },
                'memory': {
                    'total_gb': round(memory_gb, 2),
                    'available_gb': round(memory_available_gb, 2),
                    'usage_percent': memory.percent
                },
                'disk': {
                    'free_gb': round(disk_free_gb, 2),
                    'usage_percent': disk_usage.used / disk_usage.total * 100,
                    'storage_type': storage_type
                },
                'network': {
                    'bytes_sent': network_stats.bytes_sent,
                    'bytes_recv': network_stats.bytes_recv
                },
                'platform': {
                    'system': system_platform,
                    'architecture': architecture,
                    'is_cloud': is_cloud
                }
            }
            
            # Only log basic system info at debug level
            logger.debug(f"🔍 System Resources: {cpu_count_logical} cores, {memory_gb:.1f}GB RAM, {disk_free_gb:.1f}GB disk ({storage_type})")
            
            return system_info
            
        except Exception as e:
            logger.warning(f"Error detecting system resources: {e}")
            # Fallback to basic detection
            return {
                'cpu': {'logical_cores': multiprocessing.cpu_count(), 'physical_cores': multiprocessing.cpu_count()},
                'memory': {'total_gb': 8, 'available_gb': 4},
                'disk': {'free_gb': 50, 'storage_type': 'unknown'},
                'platform': {'system': platform.system(), 'is_cloud': False}
            }
    
    def _detect_cloud_environment(self) -> bool:
        """
        Detect if running in a cloud environment
        """
        cloud_indicators = [
            os.environ.get('CLOUD_RUN_SERVICE'),
            os.environ.get('AWS_EXECUTION_ENV'),
            os.environ.get('AZURE_FUNCTIONS_ENVIRONMENT'),
            os.environ.get('GOOGLE_CLOUD_PROJECT'),
            os.environ.get('KUBERNETES_SERVICE_HOST'),
            Path('/proc/version').exists() and 'Microsoft' in Path('/proc/version').read_text() if Path('/proc/version').exists() else False
        ]
        return any(cloud_indicators)
    
    def _detect_storage_type(self) -> str:
        """
        Attempt to detect storage type (SSD vs HDD)
        """
        try:
            if platform.system() == 'Linux':
                # Check for rotational storage
                for device in Path('/sys/block').glob('*'):
                    queue_path = device / 'queue' / 'rotational'
                    if queue_path.exists():
                        rotational = queue_path.read_text().strip()
                        return 'HDD' if rotational == '1' else 'SSD'
            elif platform.system() == 'Darwin':  # macOS
                # Most modern Macs have SSDs
                return 'SSD'
            elif platform.system() == 'Windows':
                # Would need additional logic for Windows
                return 'unknown'
        except Exception:
            pass
        return 'unknown'
    
    def _calculate_optimization_profile(self) -> Dict:
        """
        Calculate optimal download parameters based on system resources
        """
        cpu_cores = self.system_info['cpu']['logical_cores']
        memory_gb = self.system_info['memory']['available_gb']
        is_cloud = self.system_info['platform']['is_cloud']
        storage_type = self.system_info['disk']['storage_type']
        
        # Base thread calculation
        if cpu_cores <= 2:
            # Low-end systems
            base_threads = cpu_cores * 2
            max_threads = 8
        elif cpu_cores <= 4:
            # Mid-range systems
            base_threads = cpu_cores * 3
            max_threads = 16
        elif cpu_cores <= 8:
            # High-end consumer systems
            base_threads = cpu_cores * 4
            max_threads = 32
        else:
            # Server/workstation systems
            base_threads = cpu_cores * 3
            max_threads = min(cpu_cores * 4, 64)
        
        # Memory-based adjustments
        if memory_gb < 4:
            # Low memory - reduce threads
            base_threads = min(base_threads, 8)
            max_threads = min(max_threads, 16)
        elif memory_gb < 8:
            # Medium memory - moderate threads
            base_threads = min(base_threads, 16)
            max_threads = min(max_threads, 24)
        
        # Cloud environment adjustments
        if is_cloud:
            # Cloud environments often have shared resources
            base_threads = min(base_threads, cpu_cores * 2)
            max_threads = min(max_threads, 32)
        
        # Storage type adjustments
        buffer_size = 4096 if storage_type == 'SSD' else 2048
        
        # Quality strategy based on system capability
        if cpu_cores >= 8 and memory_gb >= 16:
            quality_strategies = ["480p", "360p", "720p", "480p"]
        elif cpu_cores >= 4 and memory_gb >= 8:
            quality_strategies = ["480p", "360p", "480p", "720p"]
        else:
            quality_strategies = ["360p", "480p", "360p", "720p"]
        
        profile = {
            'base_threads': base_threads,
            'max_threads': max_threads,
            'buffer_size': buffer_size,
            'quality_strategies': quality_strategies,
            'timeout_multiplier': 2.0 if is_cloud else 1.5,
            'parallel_downloads': min(2, cpu_cores // 4) if cpu_cores >= 8 else 1
        }
        
        # Log optimization profile at debug level only
        logger.debug(f"🚀 Optimization: {base_threads}-{max_threads} threads, {buffer_size}B buffer")
        
        return profile
    
    def get_download_strategies(self) -> list:
        """
        Get optimized download strategies for this system
        """
        profile = self.optimization_profile
        cpu_cores = self.system_info['cpu']['logical_cores']
        
        strategies = []
        
        # Strategy 1: Ultra High Performance
        strategies.append({
            "quality": profile['quality_strategies'][0],
            "threads": profile['max_threads'],
            "description": f"Ultra High Performance ({profile['max_threads']} threads)"
        })
        
        # Strategy 2: High Performance
        strategies.append({
            "quality": profile['quality_strategies'][1],
            "threads": profile['base_threads'],
            "description": f"High Performance ({profile['base_threads']} threads)"
        })
        
        # Strategy 3: Balanced Performance
        strategies.append({
            "quality": profile['quality_strategies'][2],
            "threads": max(cpu_cores, 8),
            "description": f"Balanced Performance ({max(cpu_cores, 8)} threads)"
        })
        
        # Strategy 4: Conservative Fallback
        strategies.append({
            "quality": profile['quality_strategies'][3],
            "threads": min(cpu_cores, 8),
            "description": f"Conservative Fallback ({min(cpu_cores, 8)} threads)"
        })
        
        return strategies
    
    def get_timeout_settings(self) -> Dict:
        """
        Get optimized timeout settings based on system resources
        """
        base_timeout = 3600  # 1 hour base
        multiplier = self.optimization_profile['timeout_multiplier']
        
        return {
            'download_timeout': int(base_timeout * multiplier),
            'audio_timeout': int(1800 * multiplier),  # 30 minutes base
            'process_timeout': int(300 * multiplier)   # 5 minutes base
        }
    
    def get_buffer_settings(self) -> Dict:
        """
        Get optimized buffer settings
        """
        return {
            'read_buffer_size': self.optimization_profile['buffer_size'],
            'read_timeout': 0.02 if self.system_info['disk']['storage_type'] == 'SSD' else 0.05
        }
    
    def log_system_summary(self):
        """
        Log a brief system summary
        """
        info = self.system_info
        profile = self.optimization_profile

        # Only log a brief summary
        logger.info(f"🚀 Optimized for {info['cpu']['logical_cores']} cores, {info['memory']['total_gb']:.0f}GB RAM ({profile['base_threads']}-{profile['max_threads']} threads)")


# Global instance
resource_optimizer = ResourceOptimizer()
