"""
High-Speed Feature Caching System
=================================

Redis-backed feature caching with sub-millisecond access times.
Implements multi-level caching, compression, and prefetching.
"""

import asyncio
import time
import logging
import pickle
import lz4.frame
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import msgpack
import msgpack_numpy as m

# Patch msgpack for numpy support
m.patch()

logger = logging.getLogger(__name__)

# Cache configuration
L1_CACHE_SIZE = 10000  # In-memory cache entries
L2_CACHE_SIZE = 100000  # Redis cache entries
CACHE_TTL = 300  # 5 minutes TTL
PREFETCH_SIZE = 100
COMPRESSION_THRESHOLD = 1024  # Compress if larger than 1KB


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    ttl: int = CACHE_TTL
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() - self.timestamp > self.ttl


@dataclass
class CacheStats:
    """Cache performance statistics"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    total_requests: int = 0
    avg_latency_us: float = 0.0
    compression_ratio: float = 1.0
    memory_usage_mb: float = 0.0
    evictions: int = 0
    
    @property
    def l1_hit_rate(self) -> float:
        """L1 cache hit rate"""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0
    
    @property
    def l2_hit_rate(self) -> float:
        """L2 cache hit rate"""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0
    
    @property
    def overall_hit_rate(self) -> float:
        """Overall cache hit rate"""
        hits = self.l1_hits + self.l2_hits
        return hits / self.total_requests if self.total_requests > 0 else 0.0


class LRUCache:
    """Fast LRU cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.size = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if self.size >= self.capacity:
                # Evict least recently used
                self.cache.popitem(last=False)
                self.size -= 1
            
            self.size += 1
        
        self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete from cache"""
        if key in self.cache:
            del self.cache[key]
            self.size -= 1
            return True
        return False
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.size = 0


class FeatureCache:
    """
    High-speed multi-level feature caching system.
    L1: In-memory LRU cache (microsecond access)
    L2: Redis cache (sub-millisecond access)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # L1 Cache (in-memory)
        self.l1_cache = LRUCache(self.config['l1_size'])
        
        # L2 Cache (Redis)
        self.redis_pool: Optional[ConnectionPool] = None
        self.redis: Optional[redis.Redis] = None
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Prefetch queue
        self.prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=PREFETCH_SIZE)
        self.prefetch_keys: Set[str] = set()
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        logger.info("Feature Cache initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'l1_size': L1_CACHE_SIZE,
            'l2_size': L2_CACHE_SIZE,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'redis_password': None,
            'pool_size': 50,
            'ttl': CACHE_TTL,
            'compression_enabled': True,
            'compression_threshold': COMPRESSION_THRESHOLD,
            'enable_prefetch': True,
            'enable_stats': True,
            'stats_interval': 60  # seconds
        }
    
    async def initialize(self):
        """Initialize async components"""
        # Create Redis connection pool
        self.redis_pool = ConnectionPool(
            host=self.config['redis_host'],
            port=self.config['redis_port'],
            db=self.config['redis_db'],
            password=self.config['redis_password'],
            max_connections=self.config['pool_size'],
            decode_responses=False  # Binary data
        )
        
        self.redis = redis.Redis(connection_pool=self.redis_pool)
        
        # Test connection
        await self.redis.ping()
        
        # Start background tasks
        if self.config['enable_prefetch']:
            self.tasks.append(asyncio.create_task(self._prefetch_worker()))
        
        if self.config['enable_stats']:
            self.tasks.append(asyncio.create_task(self._stats_reporter()))
        
        self.tasks.append(asyncio.create_task(self._cleanup_worker()))
        
        logger.info("Feature Cache initialized with Redis backend")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get feature from cache with multi-level lookup.
        Returns None if not found.
        """
        start_time = time.perf_counter()
        self.stats.total_requests += 1
        
        # L1 lookup (in-memory)
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats.l1_hits += 1
            self._update_latency(start_time)
            
            # Check if expired
            if isinstance(value, CacheEntry) and value.is_expired():
                self.l1_cache.delete(key)
                value = None
            else:
                return self._deserialize_value(value)
        
        self.stats.l1_misses += 1
        
        # L2 lookup (Redis)
        value = await self._get_from_redis(key)
        if value is not None:
            self.stats.l2_hits += 1
            
            # Promote to L1
            self.l1_cache.put(key, value)
            
            self._update_latency(start_time)
            return self._deserialize_value(value)
        
        self.stats.l2_misses += 1
        self._update_latency(start_time)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set feature in cache with write-through to both levels.
        """
        ttl = ttl or self.config['ttl']
        
        # Serialize and potentially compress
        entry = self._create_cache_entry(key, value, ttl)
        
        # Write to L1
        self.l1_cache.put(key, entry)
        
        # Write to L2 (async)
        asyncio.create_task(self._set_in_redis(key, entry, ttl))
        
        return True
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Batch get multiple keys efficiently.
        """
        results = {}
        redis_keys = []
        
        # Check L1 first
        for key in keys:
            value = self.l1_cache.get(key)
            if value is not None:
                if not (isinstance(value, CacheEntry) and value.is_expired()):
                    results[key] = self._deserialize_value(value)
                    self.stats.l1_hits += 1
                else:
                    redis_keys.append(key)
                    self.stats.l1_misses += 1
            else:
                redis_keys.append(key)
                self.stats.l1_misses += 1
        
        # Batch get from Redis
        if redis_keys:
            redis_results = await self._batch_get_from_redis(redis_keys)
            results.update(redis_results)
        
        return results
    
    async def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Batch set multiple key-value pairs.
        """
        ttl = ttl or self.config['ttl']
        
        # Prepare entries
        entries = {}
        for key, value in items.items():
            entry = self._create_cache_entry(key, value, ttl)
            
            # Write to L1
            self.l1_cache.put(key, entry)
            entries[key] = entry
        
        # Batch write to L2
        asyncio.create_task(self._batch_set_in_redis(entries, ttl))
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete from both cache levels"""
        # Delete from L1
        l1_deleted = self.l1_cache.delete(key)
        
        # Delete from L2
        l2_deleted = await self.redis.delete(key) > 0
        
        return l1_deleted or l2_deleted
    
    async def clear(self):
        """Clear all cache levels"""
        # Clear L1
        self.l1_cache.clear()
        
        # Clear L2 (pattern delete)
        async for key in self.redis.scan_iter("*"):
            await self.redis.delete(key)
        
        logger.info("Cache cleared")
    
    def _create_cache_entry(self, key: str, value: Any, ttl: int) -> CacheEntry:
        """Create cache entry with serialization and compression"""
        # Serialize value
        serialized = self._serialize_value(value)
        
        # Compress if needed
        compressed = False
        if (self.config['compression_enabled'] and 
            len(serialized) > self.config['compression_threshold']):
            serialized = lz4.frame.compress(serialized)
            compressed = True
        
        return CacheEntry(
            key=key,
            value=serialized,
            timestamp=time.time(),
            size_bytes=len(serialized),
            compressed=compressed,
            ttl=ttl
        )
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for caching"""
        if isinstance(value, np.ndarray):
            # Use msgpack for numpy arrays (fast)
            return msgpack.packb(value, default=m.encode)
        else:
            # Use pickle for general objects
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_value(self, entry: Any) -> Any:
        """Deserialize cached value"""
        if not isinstance(entry, CacheEntry):
            return entry
        
        value = entry.value
        
        # Decompress if needed
        if entry.compressed:
            value = lz4.frame.decompress(value)
        
        # Deserialize
        try:
            # Try msgpack first (for numpy arrays)
            return msgpack.unpackb(value, object_hook=m.decode)
        except:
            # Fall back to pickle
            return pickle.loads(value)
    
    async def _get_from_redis(self, key: str) -> Optional[CacheEntry]:
        """Get from Redis cache"""
        try:
            data = await self.redis.get(key)
            if data:
                # Deserialize entry
                entry = pickle.loads(data)
                return entry
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    async def _set_in_redis(self, key: str, entry: CacheEntry, ttl: int):
        """Set in Redis cache"""
        try:
            # Serialize entry
            data = pickle.dumps(entry)
            
            # Set with TTL
            await self.redis.set(key, data, ex=ttl)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def _batch_get_from_redis(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get from Redis"""
        results = {}
        
        try:
            # Use pipeline for efficiency
            pipe = self.redis.pipeline()
            for key in keys:
                pipe.get(key)
            
            values = await pipe.execute()
            
            for key, data in zip(keys, values):
                if data:
                    entry = pickle.loads(data)
                    if not entry.is_expired():
                        results[key] = self._deserialize_value(entry)
                        self.stats.l2_hits += 1
                        
                        # Promote to L1
                        self.l1_cache.put(key, entry)
                    else:
                        self.stats.l2_misses += 1
                else:
                    self.stats.l2_misses += 1
                    
        except Exception as e:
            logger.error(f"Redis batch get error: {e}")
        
        return results
    
    async def _batch_set_in_redis(self, entries: Dict[str, CacheEntry], ttl: int):
        """Batch set in Redis"""
        try:
            pipe = self.redis.pipeline()
            
            for key, entry in entries.items():
                data = pickle.dumps(entry)
                pipe.set(key, data, ex=ttl)
            
            await pipe.execute()
            
        except Exception as e:
            logger.error(f"Redis batch set error: {e}")
    
    def _update_latency(self, start_time: float):
        """Update latency statistics"""
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        self.stats.avg_latency_us = (
            self.stats.avg_latency_us * 0.95 + latency_us * 0.05
        )
    
    async def prefetch(self, keys: List[str]):
        """Prefetch keys into cache"""
        for key in keys:
            if key not in self.prefetch_keys:
                await self.prefetch_queue.put(key)
                self.prefetch_keys.add(key)
    
    async def _prefetch_worker(self):
        """Background prefetch worker"""
        while True:
            try:
                # Get keys to prefetch
                keys_to_fetch = []
                
                # Batch prefetch
                for _ in range(min(10, self.prefetch_queue.qsize())):
                    try:
                        key = await asyncio.wait_for(
                            self.prefetch_queue.get(), 
                            timeout=0.1
                        )
                        keys_to_fetch.append(key)
                        self.prefetch_keys.discard(key)
                    except asyncio.TimeoutError:
                        break
                
                if keys_to_fetch:
                    # Batch get from Redis
                    await self.batch_get(keys_to_fetch)
                
                await asyncio.sleep(0.01)  # 10ms interval
                
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup_worker(self):
        """Clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Clean every minute
                
                # Clean L1 cache
                expired_keys = []
                for key, value in self.l1_cache.cache.items():
                    if isinstance(value, CacheEntry) and value.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self.l1_cache.delete(key)
                    self.stats.evictions += 1
                
                if expired_keys:
                    logger.info(f"Evicted {len(expired_keys)} expired entries from L1")
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    async def _stats_reporter(self):
        """Report cache statistics"""
        while True:
            try:
                await asyncio.sleep(self.config['stats_interval'])
                
                # Calculate memory usage
                memory_usage = sum(
                    entry.size_bytes 
                    for entry in self.l1_cache.cache.values()
                    if isinstance(entry, CacheEntry)
                ) / 1_048_576  # Convert to MB
                
                self.stats.memory_usage_mb = memory_usage
                
                logger.info(
                    f"Cache Stats: "
                    f"L1 Hit Rate: {self.stats.l1_hit_rate:.2%}, "
                    f"L2 Hit Rate: {self.stats.l2_hit_rate:.2%}, "
                    f"Overall Hit Rate: {self.stats.overall_hit_rate:.2%}, "
                    f"Avg Latency: {self.stats.avg_latency_us:.2f}μs, "
                    f"Memory: {memory_usage:.2f}MB, "
                    f"Evictions: {self.stats.evictions}"
                )
                
            except Exception as e:
                logger.error(f"Stats reporter error: {e}")
    
    async def shutdown(self):
        """Clean shutdown"""
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        if self.redis_pool:
            await self.redis_pool.disconnect()
        
        logger.info("Feature Cache shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'l1_hit_rate': round(self.stats.l1_hit_rate, 4),
            'l2_hit_rate': round(self.stats.l2_hit_rate, 4),
            'overall_hit_rate': round(self.stats.overall_hit_rate, 4),
            'avg_latency_us': round(self.stats.avg_latency_us, 2),
            'memory_usage_mb': round(self.stats.memory_usage_mb, 2),
            'total_requests': self.stats.total_requests,
            'evictions': self.stats.evictions,
            'l1_size': self.l1_cache.size,
            'compression_ratio': round(self.stats.compression_ratio, 2)
        }


# Factory function
def create_feature_cache(config: Optional[Dict] = None) -> FeatureCache:
    """Create and return a feature cache instance"""
    return FeatureCache(config)