"""
Performance Tracking and Metrics Collection System
Comprehensive monitoring and analytics for the AdaptiveThreshold system
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from contextlib import contextmanager
import json
import psutil
import traceback
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationEvent:
    """Records an adaptation event with full context"""
    user_id: str
    symbol: Optional[str]
    parameter_name: str
    old_value: float
    new_value: float
    performance_score: float
    confidence: float
    reason: str
    timestamp: datetime
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_connections: int
    request_count: int
    error_count: int
    avg_response_time_ms: float
    timestamp: datetime


@dataclass
class TradingPerformanceSnapshot:
    """Snapshot of trading performance metrics"""
    user_id: str
    symbol: Optional[str]
    total_return: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_return: float
    max_drawdown: float
    volatility: float
    trade_count: int
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    timestamp: datetime
    period_days: int


class MetricsCollector:
    """Thread-safe metrics collection with batching and persistence"""
    
    def __init__(self, batch_size: int = 100, flush_interval: int = 60):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._metrics_buffer: List[PerformanceMetric] = []
        self._adaptations_buffer: List[AdaptationEvent] = []
        self._system_metrics_buffer: List[SystemMetrics] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics")
        
        # Initialize local storage
        self._init_local_storage()
        
        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self._flush_thread.start()
    
    def _init_local_storage(self):
        """Initialize local SQLite storage for metrics"""
        storage_path = Path("metrics_storage.db")
        self._conn = sqlite3.connect(str(storage_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS adaptation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT,
                parameter_name TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                performance_score REAL NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                execution_time_ms REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT
            )
        """)
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                memory_used_mb REAL NOT NULL,
                disk_usage_percent REAL NOT NULL,
                active_connections INTEGER NOT NULL,
                request_count INTEGER NOT NULL,
                error_count INTEGER NOT NULL,
                avg_response_time_ms REAL NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """)
        
        self._conn.commit()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, 
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics_buffer.append(metric)
            if len(self._metrics_buffer) >= self.batch_size:
                self._flush_metrics()
    
    def record_adaptation(self, event: AdaptationEvent):
        """Record an adaptation event"""
        with self._lock:
            self._adaptations_buffer.append(event)
            if len(self._adaptations_buffer) >= self.batch_size:
                self._flush_adaptations()
    
    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system performance metrics"""
        with self._lock:
            self._system_metrics_buffer.append(metrics)
            if len(self._system_metrics_buffer) >= self.batch_size:
                self._flush_system_metrics()
    
    def _background_flush(self):
        """Background thread to periodically flush metrics"""
        while True:
            try:
                time.sleep(self.flush_interval)
                current_time = time.time()
                
                if current_time - self._last_flush >= self.flush_interval:
                    with self._lock:
                        if self._metrics_buffer:
                            self._flush_metrics()
                        if self._adaptations_buffer:
                            self._flush_adaptations()
                        if self._system_metrics_buffer:
                            self._flush_system_metrics()
            except Exception as e:
                logger.error(f"Error in background flush: {e}")
    
    def _flush_metrics(self):
        """Flush performance metrics to storage"""
        if not self._metrics_buffer:
            return
        
        try:
            metrics_data = [
                (m.name, m.value, m.timestamp.isoformat(), 
                 json.dumps(m.tags), json.dumps(m.metadata))
                for m in self._metrics_buffer
            ]
            
            self._conn.executemany(
                "INSERT INTO performance_metrics (name, value, timestamp, tags, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                metrics_data
            )
            self._conn.commit()
            
            logger.debug(f"Flushed {len(self._metrics_buffer)} performance metrics")
            self._metrics_buffer.clear()
            self._last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Error flushing performance metrics: {e}")
    
    def _flush_adaptations(self):
        """Flush adaptation events to storage"""
        if not self._adaptations_buffer:
            return
        
        try:
            adaptations_data = [
                (a.user_id, a.symbol, a.parameter_name, a.old_value, a.new_value,
                 a.performance_score, a.confidence, a.reason, a.timestamp.isoformat(),
                 a.execution_time_ms, a.success, a.error_message)
                for a in self._adaptations_buffer
            ]
            
            self._conn.executemany(
                "INSERT INTO adaptation_events "
                "(user_id, symbol, parameter_name, old_value, new_value, performance_score, "
                "confidence, reason, timestamp, execution_time_ms, success, error_message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                adaptations_data
            )
            self._conn.commit()
            
            logger.debug(f"Flushed {len(self._adaptations_buffer)} adaptation events")
            self._adaptations_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing adaptation events: {e}")
    
    def _flush_system_metrics(self):
        """Flush system metrics to storage"""
        if not self._system_metrics_buffer:
            return
        
        try:
            system_data = [
                (m.cpu_percent, m.memory_percent, m.memory_used_mb, m.disk_usage_percent,
                 m.active_connections, m.request_count, m.error_count, 
                 m.avg_response_time_ms, m.timestamp.isoformat())
                for m in self._system_metrics_buffer
            ]
            
            self._conn.executemany(
                "INSERT INTO system_metrics "
                "(cpu_percent, memory_percent, memory_used_mb, disk_usage_percent, "
                "active_connections, request_count, error_count, avg_response_time_ms, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                system_data
            )
            self._conn.commit()
            
            logger.debug(f"Flushed {len(self._system_metrics_buffer)} system metrics")
            self._system_metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing system metrics: {e}")
    
    def force_flush(self):
        """Force flush all buffered metrics"""
        with self._lock:
            self._flush_metrics()
            self._flush_adaptations()
            self._flush_system_metrics()
    
    def get_metrics_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of metrics from the last N hours"""
        cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        
        try:
            # Get performance metrics summary
            cursor = self._conn.execute("""
                SELECT name, COUNT(*) as count, AVG(value) as avg_value, 
                       MIN(value) as min_value, MAX(value) as max_value
                FROM performance_metrics 
                WHERE timestamp > ?
                GROUP BY name
            """, (cutoff_time,))
            
            metrics_summary = {
                row[0]: {
                    'count': row[1],
                    'avg_value': row[2],
                    'min_value': row[3],
                    'max_value': row[4]
                }
                for row in cursor.fetchall()
            }
            
            # Get adaptation events summary
            cursor = self._conn.execute("""
                SELECT COUNT(*) as total_adaptations, 
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_adaptations,
                       AVG(execution_time_ms) as avg_execution_time
                FROM adaptation_events 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            adaptation_summary = cursor.fetchone()
            
            # Get system metrics summary
            cursor = self._conn.execute("""
                SELECT AVG(cpu_percent) as avg_cpu, AVG(memory_percent) as avg_memory,
                       AVG(avg_response_time_ms) as avg_response_time,
                       SUM(request_count) as total_requests, SUM(error_count) as total_errors
                FROM system_metrics 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            system_summary = cursor.fetchone()
            
            return {
                'period_hours': hours_back,
                'metrics': metrics_summary,
                'adaptations': {
                    'total': adaptation_summary[0] if adaptation_summary[0] else 0,
                    'successful': adaptation_summary[1] if adaptation_summary[1] else 0,
                    'avg_execution_time_ms': adaptation_summary[2] if adaptation_summary[2] else 0
                },
                'system': {
                    'avg_cpu_percent': system_summary[0] if system_summary[0] else 0,
                    'avg_memory_percent': system_summary[1] if system_summary[1] else 0,
                    'avg_response_time_ms': system_summary[2] if system_summary[2] else 0,
                    'total_requests': system_summary[3] if system_summary[3] else 0,
                    'total_errors': system_summary[4] if system_summary[4] else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}


class PerformanceTracker:
    """Main performance tracking class with advanced analytics"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.request_times: deque = deque(maxlen=1000)
        self.error_count = 0
        self.request_count = 0
        self._start_time = time.time()
        
        # System monitoring
        self._system_monitor_thread = threading.Thread(
            target=self._monitor_system, daemon=True
        )
        self._system_monitor_thread.start()
    
    def _monitor_system(self):
        """Monitor system performance in background"""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                system_metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    active_connections=len(psutil.net_connections()),
                    request_count=self.request_count,
                    error_count=self.error_count,
                    avg_response_time_ms=np.mean(list(self.request_times)) * 1000 if self.request_times else 0,
                    timestamp=datetime.now()
                )
                
                self.metrics_collector.record_system_metrics(system_metrics)
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error monitoring system: {e}")
                time.sleep(60)
    
    @contextmanager
    def track_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for tracking operation performance"""
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            self.error_count += 1
            raise
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record timing metric
            self.metrics_collector.record_metric(
                name=f"{operation_name}_duration_seconds",
                value=execution_time,
                tags=tags,
                metadata={'success': success, 'error_message': error_message}
            )
            
            # Track for average response time calculation
            if operation_name.startswith('http_'):
                self.request_times.append(execution_time)
                self.request_count += 1
    
    def track_adaptation_performance(self, user_id: str, symbol: Optional[str], 
                                   parameter_updates: List, performance_score: float,
                                   execution_time_ms: float, success: bool, 
                                   error_message: Optional[str] = None):
        """Track detailed adaptation performance"""
        for update in parameter_updates:
            event = AdaptationEvent(
                user_id=user_id,
                symbol=symbol,
                parameter_name=update.parameter_name,
                old_value=update.old_value,
                new_value=update.new_value,
                performance_score=performance_score,
                confidence=update.confidence,
                reason=update.reason,
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message
            )
            
            self.metrics_collector.record_adaptation(event)
            
            # Record individual parameter change metrics
            self.metrics_collector.record_metric(
                name="parameter_adaptation",
                value=abs(update.new_value - update.old_value) / update.old_value,
                tags={
                    'user_id': user_id,
                    'symbol': symbol or 'global',
                    'parameter': update.parameter_name,
                    'success': str(success)
                },
                metadata={
                    'old_value': update.old_value,
                    'new_value': update.new_value,
                    'confidence': update.confidence
                }
            )
    
    def track_trading_performance(self, snapshot: TradingPerformanceSnapshot):
        """Track trading performance snapshot"""
        tags = {
            'user_id': snapshot.user_id,
            'symbol': snapshot.symbol or 'portfolio',
            'period_days': str(snapshot.period_days)
        }
        
        # Record all performance metrics
        metrics_to_record = [
            ('total_return', snapshot.total_return),
            ('sharpe_ratio', snapshot.sharpe_ratio),
            ('win_rate', snapshot.win_rate),
            ('avg_trade_return', snapshot.avg_trade_return),
            ('max_drawdown', snapshot.max_drawdown),
            ('volatility', snapshot.volatility),
            ('trade_count', snapshot.trade_count),
            ('profit_factor', snapshot.profit_factor),
            ('calmar_ratio', snapshot.calmar_ratio),
            ('sortino_ratio', snapshot.sortino_ratio)
        ]
        
        for metric_name, value in metrics_to_record:
            self.metrics_collector.record_metric(
                name=f"trading_{metric_name}",
                value=value,
                tags=tags,
                metadata={'snapshot_timestamp': snapshot.timestamp.isoformat()}
            )
    
    def get_performance_analytics(self, user_id: str, symbol: Optional[str] = None,
                                 hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            # Query adaptation performance
            cursor = self.metrics_collector._conn.execute("""
                SELECT parameter_name, COUNT(*) as adaptation_count,
                       AVG(ABS(new_value - old_value) / old_value) as avg_change_percent,
                       AVG(performance_score) as avg_performance_score,
                       AVG(confidence) as avg_confidence,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_adaptations
                FROM adaptation_events 
                WHERE user_id = ? AND (symbol = ? OR ? IS NULL) AND timestamp > ?
                GROUP BY parameter_name
            """, (user_id, symbol, symbol, cutoff_time))
            
            adaptation_analytics = {
                row[0]: {
                    'adaptation_count': row[1],
                    'avg_change_percent': row[2] * 100 if row[2] else 0,
                    'avg_performance_score': row[3] if row[3] else 0,
                    'avg_confidence': row[4] if row[4] else 0,
                    'success_rate': row[5] / row[1] if row[1] > 0 else 0
                }
                for row in cursor.fetchall()
            }
            
            # Query trading performance trends
            cursor = self.metrics_collector._conn.execute("""
                SELECT name, value, timestamp
                FROM performance_metrics 
                WHERE tags LIKE ? AND name LIKE 'trading_%' AND timestamp > ?
                ORDER BY timestamp
            """, (f'%"user_id": "{user_id}"%', cutoff_time))
            
            trading_metrics = defaultdict(list)
            for row in cursor.fetchall():
                metric_name = row[0].replace('trading_', '')
                trading_metrics[metric_name].append({
                    'value': row[1],
                    'timestamp': row[2]
                })
            
            # Calculate performance trends
            performance_trends = {}
            for metric_name, values in trading_metrics.items():
                if len(values) >= 2:
                    recent_values = [v['value'] for v in values[-10:]]  # Last 10 points
                    if len(recent_values) >= 2:
                        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                        performance_trends[metric_name] = {
                            'trend_slope': trend,
                            'latest_value': recent_values[-1],
                            'change_from_start': recent_values[-1] - recent_values[0] if recent_values else 0,
                            'data_points': len(recent_values)
                        }
            
            return {
                'user_id': user_id,
                'symbol': symbol,
                'period_hours': hours_back,
                'adaptation_analytics': adaptation_analytics,
                'performance_trends': performance_trends,
                'summary': {
                    'total_adaptations': sum(a['adaptation_count'] for a in adaptation_analytics.values()),
                    'avg_success_rate': np.mean([a['success_rate'] for a in adaptation_analytics.values()]) if adaptation_analytics else 0,
                    'trending_metrics': len(performance_trends)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {'error': str(e)}
    
    def export_metrics(self, file_path: str, format: str = 'csv', hours_back: int = 24):
        """Export metrics to file for external analysis"""
        cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        
        try:
            if format.lower() == 'csv':
                # Export performance metrics
                df_metrics = pd.read_sql_query("""
                    SELECT name, value, timestamp, tags, metadata
                    FROM performance_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp
                """, self.metrics_collector._conn, params=(cutoff_time,))
                
                df_metrics.to_csv(f"{file_path}_metrics.csv", index=False)
                
                # Export adaptation events
                df_adaptations = pd.read_sql_query("""
                    SELECT user_id, symbol, parameter_name, old_value, new_value,
                           performance_score, confidence, reason, timestamp, 
                           execution_time_ms, success, error_message
                    FROM adaptation_events 
                    WHERE timestamp > ?
                    ORDER BY timestamp
                """, self.metrics_collector._conn, params=(cutoff_time,))
                
                df_adaptations.to_csv(f"{file_path}_adaptations.csv", index=False)
                
                # Export system metrics
                df_system = pd.read_sql_query("""
                    SELECT cpu_percent, memory_percent, memory_used_mb, disk_usage_percent,
                           active_connections, request_count, error_count, 
                           avg_response_time_ms, timestamp
                    FROM system_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp
                """, self.metrics_collector._conn, params=(cutoff_time,))
                
                df_system.to_csv(f"{file_path}_system.csv", index=False)
                
                logger.info(f"Metrics exported to {file_path}_*.csv files")
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise


def performance_monitor(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for performance monitoring"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with performance_tracker.track_operation(operation_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global performance tracker instance
performance_tracker = PerformanceTracker()


if __name__ == "__main__":
    # Example usage
    tracker = PerformanceTracker()
    
    # Example: Track an operation
    with tracker.track_operation("test_operation", tags={"component": "test"}):
        time.sleep(0.1)  # Simulate work
    
    # Example: Track adaptation performance
    from adaptive_threshold import ThresholdUpdate
    
    updates = [
        ThresholdUpdate(
            parameter_name="rsi_threshold",
            old_value=70.0,
            new_value=72.5,
            reason="Performance improvement",
            confidence=0.85
        )
    ]
    
    tracker.track_adaptation_performance(
        user_id="test_user",
        symbol="BTCUSD",
        parameter_updates=updates,
        performance_score=0.75,
        execution_time_ms=15.5,
        success=True
    )
    
    # Get analytics
    analytics = tracker.get_performance_analytics("test_user", "BTCUSD", hours_back=1)
    print("Performance Analytics:", json.dumps(analytics, indent=2, default=str))
    
    # Force flush metrics
    tracker.metrics_collector.force_flush()
    
    # Export metrics
    tracker.export_metrics("test_export", format="csv", hours_back=1)