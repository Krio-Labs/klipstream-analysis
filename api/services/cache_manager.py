"""
Cache Manager Service

This module provides caching functionality for the KlipStream Analysis API,
including in-memory caching with TTL support and cache invalidation strategies.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Cache type categories"""
    JOB_STATUS = "job_status"
    JOB_RESULTS = "job_results"
    WEBHOOK_CONFIG = "webhook_config"
    ANALYTICS = "analytics"
    SYSTEM_METRICS = "system_metrics"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    cache_type: CacheType
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    def access(self) -> Any:
        """Access the cached value and update metadata"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        return self.value


class CacheManager:
    """
    In-memory cache manager with TTL support and intelligent invalidation
    """
    
    def __init__(self, default_ttl_seconds: int = 300):
        self.default_ttl_seconds = default_ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0
        }
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        logger.info(f"CacheManager initialized with default TTL: {default_ttl_seconds}s")
    
    async def start(self):
        """Start the cache manager and cleanup task"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Cache manager started")
    
    async def stop(self):
        """Stop the cache manager and cleanup task"""
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cache manager stopped")
    
    def set(
        self, 
        key: str, 
        value: Any, 
        cache_type: CacheType = CacheType.JOB_STATUS,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set a value in the cache
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache entry
            ttl_seconds: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        try:
            ttl = ttl_seconds or self.default_ttl_seconds
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                cache_type=cache_type,
                created_at=now,
                expires_at=expires_at
            )
            
            self.cache[key] = entry
            logger.debug(f"Cached {cache_type.value} key '{key}' with TTL {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key '{key}': {str(e)}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            entry = self.cache.get(key)
            
            if not entry:
                self.cache_stats["misses"] += 1
                logger.debug(f"Cache miss for key '{key}'")
                return None
            
            if entry.is_expired():
                # Remove expired entry
                del self.cache[key]
                self.cache_stats["misses"] += 1
                self.cache_stats["evictions"] += 1
                logger.debug(f"Cache expired for key '{key}'")
                return None
            
            # Cache hit
            self.cache_stats["hits"] += 1
            logger.debug(f"Cache hit for key '{key}'")
            return entry.access()
            
        except Exception as e:
            logger.error(f"Error getting cache key '{key}': {str(e)}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a specific cache entry
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was found and deleted
        """
        try:
            if key in self.cache:
                del self.cache[key]
                self.cache_stats["invalidations"] += 1
                logger.debug(f"Deleted cache key '{key}'")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting cache key '{key}': {str(e)}")
            return False
    
    def invalidate_by_type(self, cache_type: CacheType) -> int:
        """
        Invalidate all cache entries of a specific type
        
        Args:
            cache_type: Type of cache entries to invalidate
            
        Returns:
            Number of entries invalidated
        """
        try:
            keys_to_delete = [
                key for key, entry in self.cache.items()
                if entry.cache_type == cache_type
            ]
            
            for key in keys_to_delete:
                del self.cache[key]
            
            count = len(keys_to_delete)
            self.cache_stats["invalidations"] += count
            logger.info(f"Invalidated {count} cache entries of type {cache_type.value}")
            return count
            
        except Exception as e:
            logger.error(f"Error invalidating cache type {cache_type.value}: {str(e)}")
            return 0
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern
        
        Args:
            pattern: Pattern to match against cache keys
            
        Returns:
            Number of entries invalidated
        """
        try:
            keys_to_delete = [
                key for key in self.cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_delete:
                del self.cache[key]
            
            count = len(keys_to_delete)
            self.cache_stats["invalidations"] += count
            logger.info(f"Invalidated {count} cache entries matching pattern '{pattern}'")
            return count
            
        except Exception as e:
            logger.error(f"Error invalidating cache pattern '{pattern}': {str(e)}")
            return 0
    
    def clear(self) -> int:
        """
        Clear all cache entries
        
        Returns:
            Number of entries cleared
        """
        try:
            count = len(self.cache)
            self.cache.clear()
            self.cache_stats["invalidations"] += count
            logger.info(f"Cleared all {count} cache entries")
            return count
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        # Count entries by type
        type_counts = {}
        expired_count = 0
        
        for entry in self.cache.values():
            cache_type = entry.cache_type.value
            type_counts[cache_type] = type_counts.get(cache_type, 0) + 1
            
            if entry.is_expired():
                expired_count += 1
        
        return {
            "total_entries": len(self.cache),
            "expired_entries": expired_count,
            "entries_by_type": type_counts,
            "hit_rate_percentage": round(hit_rate, 2),
            "total_hits": self.cache_stats["hits"],
            "total_misses": self.cache_stats["misses"],
            "total_evictions": self.cache_stats["evictions"],
            "total_invalidations": self.cache_stats["invalidations"],
            "memory_usage_estimate": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage of cache"""
        try:
            import sys
            total_size = 0
            
            for entry in self.cache.values():
                total_size += sys.getsizeof(entry.key)
                total_size += sys.getsizeof(entry.value)
                total_size += sys.getsizeof(entry)
            
            # Convert to human readable format
            if total_size < 1024:
                return f"{total_size} B"
            elif total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            else:
                return f"{total_size / (1024 * 1024):.1f} MB"
                
        except Exception:
            return "Unknown"
    
    async def _cleanup_loop(self):
        """Background task to clean up expired entries"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run cleanup every minute
                
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if entry.is_expired()
                ]
                
                for key in expired_keys:
                    del self.cache[key]
                    self.cache_stats["evictions"] += 1
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {str(e)}")


# Global cache manager instance
cache_manager = CacheManager()


# Convenience functions for common caching patterns
def cache_job_status(job_id: str, status_data: Dict[str, Any], ttl_seconds: int = 60) -> bool:
    """Cache job status data"""
    return cache_manager.set(
        key=f"job_status:{job_id}",
        value=status_data,
        cache_type=CacheType.JOB_STATUS,
        ttl_seconds=ttl_seconds
    )


def get_cached_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get cached job status data"""
    return cache_manager.get(f"job_status:{job_id}")


def invalidate_job_cache(job_id: str) -> int:
    """Invalidate all cache entries for a specific job"""
    return cache_manager.invalidate_by_pattern(job_id)


def cache_job_results(job_id: str, results_data: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
    """Cache job results data"""
    return cache_manager.set(
        key=f"job_results:{job_id}",
        value=results_data,
        cache_type=CacheType.JOB_RESULTS,
        ttl_seconds=ttl_seconds
    )


def get_cached_job_results(job_id: str) -> Optional[Dict[str, Any]]:
    """Get cached job results data"""
    return cache_manager.get(f"job_results:{job_id}")


async def start_cache_manager():
    """Start the global cache manager"""
    await cache_manager.start()


async def stop_cache_manager():
    """Stop the global cache manager"""
    await cache_manager.stop()
