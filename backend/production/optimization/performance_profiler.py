"""
Performance Profiler and Monitoring System
==========================================

Real-time performance monitoring with sub-millisecond precision.
Tracks latency, throughput, and identifies bottlenecks.
"""

import time
import asyncio
import logging
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import numpy as np
from datetime import datetime, timedelta
import json
import tracemalloc
import gc

logger = logging.getLogger(__name__)


@dataclass
class LatencyBucket:
    """Latency histogram bucket"""
    min_ms: float
    max_ms: float
    count: int = 0
    
    def add(self, latency_ms: float) -> bool:
        """Add to bucket if within range"""
        if self.min_ms <= latency_ms < self.max_ms:
            self.count += 1
            return True
        return False


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Latency metrics (milliseconds)
    current_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    p999_latency_ms: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    transactions_per_second: float = 0.0
    messages_per_second: float = 0.0
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    open_connections: int = 0
    active_threads: int = 0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    circuit_breaker_trips: int = 0
    
    # Business metrics
    orders_per_second: float = 0.0
    fill_rate: float = 0.0
    slippage_bps: float = 0.0


@dataclass
class ComponentMetrics:
    """Metrics for individual system components"""
    name: str
    calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    error_count: int = 0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(1, self.calls)


class LatencyTracker:
    """High-precision latency tracking"""
    
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.latencies: Deque[float] = deque(maxlen=window_size)
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        
        # Histogram buckets (microseconds to seconds)
        self.buckets = [
            LatencyBucket(0, 0.1),      # < 100μs
            LatencyBucket(0.1, 0.5),    # 100μs - 500μs
            LatencyBucket(0.5, 1),      # 500μs - 1ms
            LatencyBucket(1, 5),        # 1ms - 5ms
            LatencyBucket(5, 10),       # 5ms - 10ms
            LatencyBucket(10, 50),      # 10ms - 50ms
            LatencyBucket(50, 100),     # 50ms - 100ms
            LatencyBucket(100, 500),    # 100ms - 500ms
            LatencyBucket(500, 1000),   # 500ms - 1s
            LatencyBucket(1000, float('inf'))  # > 1s
        ]
        
        self.lock = threading.Lock()
    
    def record(self, latency_ms: float):
        """Record a latency measurement"""
        with self.lock:
            self.latencies.append(latency_ms)
            self.timestamps.append(time.time())
            
            # Update histogram
            for bucket in self.buckets:
                if bucket.add(latency_ms):
                    break
    
    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not self.latencies:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
        
        with self.lock:
            sorted_latencies = np.array(sorted(self.latencies))
        
        return {
            'p50': np.percentile(sorted_latencies, 50),
            'p95': np.percentile(sorted_latencies, 95),
            'p99': np.percentile(sorted_latencies, 99),
            'p999': np.percentile(sorted_latencies, 99.9)
        }
    
    def get_histogram(self) -> List[Dict[str, Any]]:
        """Get latency histogram"""
        return [
            {
                'range': f"{b.min_ms}-{b.max_ms}ms",
                'count': b.count,
                'percentage': b.count / max(1, sum(b.count for b in self.buckets))
            }
            for b in self.buckets
        ]
    
    def get_throughput(self, window_seconds: float = 1.0) -> float:
        """Calculate throughput (requests per second)"""
        if not self.timestamps:
            return 0.0
        
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            recent_count = sum(1 for t in self.timestamps if t > cutoff_time)
            
        return recent_count / window_seconds


class PerformanceProfiler:
    """
    Main performance profiling system.
    Tracks all aspects of system performance with minimal overhead.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Metrics storage
        self.metrics = PerformanceMetrics()
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        
        # Latency tracking
        self.latency_tracker = LatencyTracker(self.config['window_size'])
        
        # Throughput tracking
        self.request_times: Deque[float] = deque(maxlen=10000)
        self.transaction_times: Deque[float] = deque(maxlen=10000)
        
        # Resource monitoring
        self.process = psutil.Process()
        
        # Profiling state
        self.profiling_enabled = self.config['enabled']
        self.memory_profiling = self.config['memory_profiling']
        
        # Start memory profiling if enabled
        if self.memory_profiling:
            tracemalloc.start()
        
        # Background monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Performance Profiler initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'enabled': True,
            'window_size': 10000,
            'monitoring_interval': 1.0,  # seconds
            'memory_profiling': False,
            'detailed_logging': False,
            'alert_thresholds': {
                'latency_p99_ms': 100,
                'error_rate': 0.01,
                'cpu_percent': 80,
                'memory_percent': 80
            }
        }
    
    async def start(self):
        """Start performance monitoring"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")
    
    async def stop(self):
        """Stop performance monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
        
        if self.memory_profiling:
            tracemalloc.stop()
        
        logger.info("Performance monitoring stopped")
    
    @contextmanager
    def measure(self, operation: str):
        """
        Context manager for measuring operation latency.
        
        Usage:
            with profiler.measure('database_query'):
                # perform operation
        """
        if not self.profiling_enabled:
            yield
            return
        
        start_time = time.perf_counter()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            raise
        finally:
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Record latency
            self.latency_tracker.record(latency_ms)
            
            # Update component metrics
            if operation not in self.component_metrics:
                self.component_metrics[operation] = ComponentMetrics(name=operation)
            
            component = self.component_metrics[operation]
            component.calls += 1
            component.total_time_ms += latency_ms
            component.avg_time_ms = component.total_time_ms / component.calls
            component.max_time_ms = max(component.max_time_ms, latency_ms)
            
            if error_occurred:
                component.error_count += 1
            
            # Update global metrics
            self.metrics.current_latency_ms = latency_ms
            self.metrics.avg_latency_ms = (
                self.metrics.avg_latency_ms * 0.95 + latency_ms * 0.05
            )
            self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency_ms)
            self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
    
    def record_request(self):
        """Record a request for throughput calculation"""
        self.request_times.append(time.time())
    
    def record_transaction(self):
        """Record a transaction"""
        self.transaction_times.append(time.time())
    
    def record_error(self, error_type: str = 'general'):
        """Record an error"""
        self.metrics.error_rate = self.metrics.error_rate * 0.99 + 0.01
    
    def record_timeout(self):
        """Record a timeout"""
        self.metrics.timeout_rate = self.metrics.timeout_rate * 0.99 + 0.01
    
    def record_circuit_breaker_trip(self):
        """Record circuit breaker trip"""
        self.metrics.circuit_breaker_trips += 1
    
    def record_order(self, filled: bool, slippage_bps: float = 0.0):
        """Record order execution metrics"""
        self.metrics.orders_per_second = self.latency_tracker.get_throughput()
        
        if filled:
            self.metrics.fill_rate = self.metrics.fill_rate * 0.99 + 0.01
        else:
            self.metrics.fill_rate = self.metrics.fill_rate * 0.99
        
        self.metrics.slippage_bps = (
            self.metrics.slippage_bps * 0.95 + slippage_bps * 0.05
        )
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.config['monitoring_interval'])
                
                # Update metrics
                await self._update_metrics()
                
                # Check alerts
                self._check_alerts()
                
                # Log if detailed logging enabled
                if self.config['detailed_logging']:
                    self._log_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _update_metrics(self):
        """Update all metrics"""
        # Latency percentiles
        percentiles = self.latency_tracker.get_percentiles()
        self.metrics.p50_latency_ms = percentiles['p50']
        self.metrics.p95_latency_ms = percentiles['p95']
        self.metrics.p99_latency_ms = percentiles['p99']
        self.metrics.p999_latency_ms = percentiles['p999']
        
        # Throughput
        self.metrics.requests_per_second = self._calculate_throughput(
            self.request_times
        )
        self.metrics.transactions_per_second = self._calculate_throughput(
            self.transaction_times
        )
        
        # Resource usage
        self.metrics.cpu_usage_percent = self.process.cpu_percent()
        
        memory_info = self.process.memory_info()
        self.metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
        self.metrics.memory_usage_percent = self.process.memory_percent()
        
        # Connection and thread count
        self.metrics.open_connections = len(self.process.connections())
        self.metrics.active_threads = threading.active_count()
    
    def _calculate_throughput(self, timestamps: Deque[float], 
                            window_seconds: float = 1.0) -> float:
        """Calculate throughput from timestamps"""
        if not timestamps:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_count = sum(1 for t in timestamps if t > cutoff_time)
        return recent_count / window_seconds
    
    def _check_alerts(self):
        """Check for performance alerts"""
        thresholds = self.config['alert_thresholds']
        
        # Latency alert
        if self.metrics.p99_latency_ms > thresholds['latency_p99_ms']:
            logger.warning(
                f"High latency detected: P99={self.metrics.p99_latency_ms:.2f}ms "
                f"(threshold={thresholds['latency_p99_ms']}ms)"
            )
        
        # Error rate alert
        if self.metrics.error_rate > thresholds['error_rate']:
            logger.warning(
                f"High error rate: {self.metrics.error_rate:.2%} "
                f"(threshold={thresholds['error_rate']:.2%})"
            )
        
        # CPU alert
        if self.metrics.cpu_usage_percent > thresholds['cpu_percent']:
            logger.warning(
                f"High CPU usage: {self.metrics.cpu_usage_percent:.1f}% "
                f"(threshold={thresholds['cpu_percent']}%)"
            )
        
        # Memory alert
        if self.metrics.memory_usage_percent > thresholds['memory_percent']:
            logger.warning(
                f"High memory usage: {self.metrics.memory_usage_percent:.1f}% "
                f"(threshold={thresholds['memory_percent']}%)"
            )
    
    def _log_metrics(self):
        """Log current metrics"""
        logger.info(
            f"Performance Metrics - "
            f"Latency: P50={self.metrics.p50_latency_ms:.2f}ms, "
            f"P99={self.metrics.p99_latency_ms:.2f}ms | "
            f"Throughput: {self.metrics.requests_per_second:.1f} RPS | "
            f"CPU: {self.metrics.cpu_usage_percent:.1f}% | "
            f"Memory: {self.metrics.memory_usage_mb:.1f}MB"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary"""
        return {
            'latency': {
                'current_ms': round(self.metrics.current_latency_ms, 2),
                'avg_ms': round(self.metrics.avg_latency_ms, 2),
                'min_ms': round(self.metrics.min_latency_ms, 2),
                'max_ms': round(self.metrics.max_latency_ms, 2),
                'p50_ms': round(self.metrics.p50_latency_ms, 2),
                'p95_ms': round(self.metrics.p95_latency_ms, 2),
                'p99_ms': round(self.metrics.p99_latency_ms, 2),
                'p999_ms': round(self.metrics.p999_latency_ms, 2),
                'histogram': self.latency_tracker.get_histogram()
            },
            'throughput': {
                'requests_per_second': round(self.metrics.requests_per_second, 2),
                'transactions_per_second': round(self.metrics.transactions_per_second, 2),
                'orders_per_second': round(self.metrics.orders_per_second, 2)
            },
            'resources': {
                'cpu_percent': round(self.metrics.cpu_usage_percent, 1),
                'memory_mb': round(self.metrics.memory_usage_mb, 1),
                'memory_percent': round(self.metrics.memory_usage_percent, 1),
                'connections': self.metrics.open_connections,
                'threads': self.metrics.active_threads
            },
            'errors': {
                'error_rate': round(self.metrics.error_rate, 4),
                'timeout_rate': round(self.metrics.timeout_rate, 4),
                'circuit_breaker_trips': self.metrics.circuit_breaker_trips
            },
            'business': {
                'fill_rate': round(self.metrics.fill_rate, 4),
                'slippage_bps': round(self.metrics.slippage_bps, 2)
            },
            'components': {
                name: {
                    'calls': comp.calls,
                    'avg_time_ms': round(comp.avg_time_ms, 2),
                    'max_time_ms': round(comp.max_time_ms, 2),
                    'error_rate': round(comp.error_rate, 4)
                }
                for name, comp in self.component_metrics.items()
            }
        }
    
    def get_memory_profile(self) -> Optional[Dict[str, Any]]:
        """Get memory profiling information"""
        if not self.memory_profiling:
            return None
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return {
            'top_allocations': [
                {
                    'file': stat.traceback.format()[0],
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                for stat in top_stats[:10]
            ],
            'total_allocated_mb': sum(stat.size for stat in top_stats) / 1024 / 1024
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = PerformanceMetrics()
        self.component_metrics.clear()
        self.latency_tracker = LatencyTracker(self.config['window_size'])
        self.request_times.clear()
        self.transaction_times.clear()
        
        logger.info("Performance metrics reset")
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        metrics = self.get_metrics()
        
        if format == 'json':
            return json.dumps(metrics, indent=2)
        elif format == 'prometheus':
            # Prometheus format
            lines = []
            
            # Latency metrics
            lines.append(f"trading_latency_p50_ms {metrics['latency']['p50_ms']}")
            lines.append(f"trading_latency_p99_ms {metrics['latency']['p99_ms']}")
            
            # Throughput metrics
            lines.append(f"trading_requests_per_second {metrics['throughput']['requests_per_second']}")
            
            # Resource metrics
            lines.append(f"trading_cpu_usage_percent {metrics['resources']['cpu_percent']}")
            lines.append(f"trading_memory_usage_mb {metrics['resources']['memory_mb']}")
            
            # Error metrics
            lines.append(f"trading_error_rate {metrics['errors']['error_rate']}")
            
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global profiler instance
_profiler: Optional[PerformanceProfiler] = None

def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler

def measure(operation: str):
    """Decorator for measuring function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with get_profiler().measure(operation):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with get_profiler().measure(operation):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator