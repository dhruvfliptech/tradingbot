"""
Monitoring and Performance Tracking System
========================================

This module provides comprehensive monitoring, alerting, and performance
tracking for the RL service integration layer.

Features:
- Real-time performance metrics collection
- Alert system for anomalies and issues
- Trading performance tracking
- Model performance comparison
- Resource utilization monitoring
- Custom dashboard metrics
- Historical data retention
- Prometheus-style metrics export
"""

import asyncio
import logging
import json
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import threading
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Alert:
    """Alert definition"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class Metric:
    """Metric definition"""
    name: str
    type: MetricType
    value: Union[float, int, List[float]]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    rl_predictions: int
    fallback_predictions: int
    total_predictions: int
    avg_response_time_ms: float
    success_rate: float
    rl_accuracy: float
    fallback_accuracy: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_websockets: int
    queue_size: int

class RLMonitor:
    """
    Main monitoring system for RL service
    
    Provides:
    - Real-time metrics collection
    - Alert management
    - Performance tracking
    - Resource monitoring
    - Historical data storage
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: Dict[str, Alert] = {}
        self.performance_history: deque = deque(maxlen=1000)
        
        # Alert rules
        self.alert_rules: Dict[str, Callable] = {}
        self.alert_thresholds = {
            'high_response_time': 1000,      # ms
            'low_success_rate': 0.9,         # percentage
            'high_queue_size': 100,          # requests
            'high_cpu_usage': 80,            # percentage
            'high_memory_usage': 1000,       # MB
            'low_rl_accuracy': 0.6,          # percentage
        }
        
        # Background task tracking
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.request_tracker = RequestTracker()
        self.trading_tracker = TradingPerformanceTracker()
        
        self._setup_default_alert_rules()
        
        logger.info("RLMonitor initialized")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        
        def high_response_time_rule():
            recent_metrics = list(self.metrics['avg_response_time_ms'])[-10:]
            if recent_metrics and np.mean(recent_metrics) > self.alert_thresholds['high_response_time']:
                return Alert(
                    id="high_response_time",
                    title="High Response Time",
                    description=f"Average response time is {np.mean(recent_metrics):.1f}ms",
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    source="performance_monitor"
                )
            return None
        
        def low_success_rate_rule():
            recent_success_rates = [m.value for m in list(self.metrics['success_rate'])[-5:]]
            if recent_success_rates and np.mean(recent_success_rates) < self.alert_thresholds['low_success_rate']:
                return Alert(
                    id="low_success_rate",
                    title="Low Success Rate",
                    description=f"Success rate dropped to {np.mean(recent_success_rates):.2f}",
                    severity=AlertSeverity.ERROR,
                    timestamp=datetime.now(),
                    source="performance_monitor"
                )
            return None
        
        def high_queue_size_rule():
            queue_sizes = [m.value for m in list(self.metrics['queue_size'])[-3:]]
            if queue_sizes and np.mean(queue_sizes) > self.alert_thresholds['high_queue_size']:
                return Alert(
                    id="high_queue_size",
                    title="High Queue Size",
                    description=f"Prediction queue size is {np.mean(queue_sizes):.0f} requests",
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    source="performance_monitor"
                )
            return None
        
        def resource_usage_rule():
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
            
            alerts = []
            if cpu_usage > self.alert_thresholds['high_cpu_usage']:
                alerts.append(Alert(
                    id="high_cpu_usage",
                    title="High CPU Usage",
                    description=f"CPU usage is {cpu_usage:.1f}%",
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    source="resource_monitor"
                ))
            
            if memory_usage > self.alert_thresholds['high_memory_usage']:
                alerts.append(Alert(
                    id="high_memory_usage",
                    title="High Memory Usage",
                    description=f"Memory usage is {memory_usage:.1f}MB",
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    source="resource_monitor"
                ))
            
            return alerts
        
        self.alert_rules = {
            'high_response_time': high_response_time_rule,
            'low_success_rate': low_success_rate_rule,
            'high_queue_size': high_queue_size_rule,
            'resource_usage': resource_usage_rule
        }
    
    async def initialize(self):
        """Initialize monitoring system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._cleanup_task())
        ]
        
        logger.info("RLMonitor started")
    
    async def cleanup(self):
        """Cleanup monitoring system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("RLMonitor stopped")
    
    def record_metric(self, name: str, value: Union[float, int], 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None, description: str = ""):
        """Record a metric value"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            description=description
        )
        
        self.metrics[name].append(metric)
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        self.record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: Union[float, int], labels: Dict[str, str] = None):
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    async def log_prediction(self, request: Any, response: Any, response_time_ms: float):
        """Log a prediction request and response"""
        
        # Record metrics
        self.record_histogram("prediction_response_time_ms", response_time_ms)
        self.increment_counter("predictions_total", labels={
            "model": response.model_used,
            "user_id": request.user_id,
            "symbol": request.symbol
        })
        
        if not response.fallback_used:
            self.increment_counter("rl_predictions_total")
        else:
            self.increment_counter("fallback_predictions_total")
        
        if response.should_trade:
            self.increment_counter("trading_signals_total", labels={
                "action": response.action,
                "confidence_bucket": self._get_confidence_bucket(response.confidence)
            })
        
        # Track for trading performance
        await self.trading_tracker.log_prediction(request, response)
        
        # Update request tracker
        self.request_tracker.record_request(response_time_ms, not response.fallback_used, response.should_trade)
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for labeling"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def create_alert(self, alert_id: str, title: str, description: str, 
                    severity: AlertSeverity, source: str = "user",
                    metadata: Dict[str, Any] = None):
        """Create a new alert"""
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        
        # Log alert
        logger.log(
            logging.ERROR if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] else logging.WARNING,
            f"Alert created: {title} - {description}"
        )
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            logger.info(f"Alert resolved: {alert.title}")
            return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts"""
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        if severity:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_recent_metrics(self, metric_name: str, hours: int = 1) -> List[Metric]:
        """Get recent metrics for a specific metric name"""
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics[metric_name] if m.timestamp > cutoff_time]
        
        return recent_metrics
    
    def get_metric_summary(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        recent_metrics = self.get_recent_metrics(metric_name, hours)
        
        if not recent_metrics:
            return {"error": f"No data for metric {metric_name}"}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "metric_name": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "latest": values[-1] if values else None,
            "time_range_hours": hours
        }
    
    async def _metrics_collector(self):
        """Background task to collect system metrics"""
        while self.is_running:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.set_gauge("cpu_usage_percent", cpu_usage)
                self.set_gauge("memory_usage_mb", memory.used / 1024 / 1024)
                self.set_gauge("memory_usage_percent", memory.percent)
                self.set_gauge("disk_usage_percent", disk.percent)
                
                # Collect request tracker metrics
                tracker_metrics = self.request_tracker.get_metrics()
                for name, value in tracker_metrics.items():
                    self.set_gauge(f"request_tracker_{name}", value)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(30)
    
    async def _alert_processor(self):
        """Background task to process alert rules"""
        while self.is_running:
            try:
                # Run alert rules
                for rule_name, rule_func in self.alert_rules.items():
                    try:
                        result = rule_func()
                        
                        if isinstance(result, list):
                            # Multiple alerts
                            for alert in result:
                                if alert and alert.id not in self.alerts:
                                    self.alerts[alert.id] = alert
                        elif result:
                            # Single alert
                            if result.id not in self.alerts:
                                self.alerts[result.id] = result
                                
                    except Exception as e:
                        logger.error(f"Error running alert rule {rule_name}: {e}")
                
                # Auto-resolve alerts that no longer apply
                await self._auto_resolve_alerts()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in alert processor: {e}")
                await asyncio.sleep(60)
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts that no longer apply"""
        for alert_id, alert in self.alerts.items():
            if alert.resolved:
                continue
            
            # Auto-resolve high response time if it's back to normal
            if alert_id == "high_response_time":
                recent_metrics = self.get_recent_metrics("prediction_response_time_ms", hours=0.1)
                if recent_metrics:
                    recent_avg = np.mean([m.value for m in recent_metrics])
                    if recent_avg < self.alert_thresholds['high_response_time']:
                        self.resolve_alert(alert_id)
            
            # Auto-resolve queue size alerts
            elif alert_id == "high_queue_size":
                recent_metrics = self.get_recent_metrics("queue_size", hours=0.1)
                if recent_metrics:
                    recent_avg = np.mean([m.value for m in recent_metrics])
                    if recent_avg < self.alert_thresholds['high_queue_size']:
                        self.resolve_alert(alert_id)
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.is_running:
            try:
                current_time = datetime.now()
                retention_cutoff = current_time - timedelta(hours=self.retention_hours)
                
                # Clean up old metrics
                for metric_name, metric_deque in self.metrics.items():
                    # Remove old metrics (deque will automatically limit size)
                    while metric_deque and metric_deque[0].timestamp < retention_cutoff:
                        metric_deque.popleft()
                
                # Clean up old resolved alerts (keep for 7 days)
                alert_retention_cutoff = current_time - timedelta(days=7)
                alerts_to_remove = []
                
                for alert_id, alert in self.alerts.items():
                    if (alert.resolved and alert.resolved_at and 
                        alert.resolved_at < alert_retention_cutoff):
                        alerts_to_remove.append(alert_id)
                
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for metric_name, metric_deque in self.metrics.items():
            if not metric_deque:
                continue
            
            latest_metric = metric_deque[-1]
            
            # Generate Prometheus metric line
            labels_str = ""
            if latest_metric.labels:
                label_parts = [f'{k}="{v}"' for k, v in latest_metric.labels.items()]
                labels_str = "{" + ",".join(label_parts) + "}"
            
            lines.append(f"rl_service_{metric_name}{labels_str} {latest_metric.value}")
        
        return "\n".join(lines)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""
        # Get key metrics summaries
        response_time_summary = self.get_metric_summary("prediction_response_time_ms", hours=1)
        
        # Recent performance snapshot
        recent_snapshot = None
        if self.performance_history:
            recent_snapshot = asdict(self.performance_history[-1])
        
        # Active alerts by severity
        alerts_by_severity = {}
        for severity in AlertSeverity:
            alerts_by_severity[severity.value] = len(self.get_active_alerts(severity))
        
        # Trading performance
        trading_metrics = self.trading_tracker.get_summary()
        
        return {
            "overview": {
                "active_alerts": len(self.get_active_alerts()),
                "alerts_by_severity": alerts_by_severity,
                "recent_performance": recent_snapshot,
                "system_health": "healthy" if len(self.get_active_alerts(AlertSeverity.ERROR)) == 0 else "degraded"
            },
            "performance": {
                "response_time": response_time_summary,
                "trading_metrics": trading_metrics,
                "request_tracker": self.request_tracker.get_metrics()
            },
            "alerts": [asdict(alert) for alert in self.get_active_alerts()],
            "metrics_available": list(self.metrics.keys())
        }


class RequestTracker:
    """Track request-level metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.requests = deque(maxlen=window_size)
        
    def record_request(self, response_time_ms: float, used_rl: bool, generated_signal: bool):
        """Record a request"""
        self.requests.append({
            'timestamp': datetime.now(),
            'response_time_ms': response_time_ms,
            'used_rl': used_rl,
            'generated_signal': generated_signal
        })
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        if not self.requests:
            return {
                'total_requests': 0,
                'avg_response_time_ms': 0.0,
                'rl_usage_rate': 0.0,
                'signal_generation_rate': 0.0
            }
        
        response_times = [r['response_time_ms'] for r in self.requests]
        rl_requests = [r for r in self.requests if r['used_rl']]
        signal_requests = [r for r in self.requests if r['generated_signal']]
        
        return {
            'total_requests': len(self.requests),
            'avg_response_time_ms': np.mean(response_times),
            'rl_usage_rate': len(rl_requests) / len(self.requests),
            'signal_generation_rate': len(signal_requests) / len(self.requests)
        }


class TradingPerformanceTracker:
    """Track trading-specific performance metrics"""
    
    def __init__(self):
        self.predictions = deque(maxlen=10000)
        self.trades = deque(maxlen=1000)
        
    async def log_prediction(self, request: Any, response: Any):
        """Log a prediction for performance tracking"""
        self.predictions.append({
            'timestamp': datetime.now(),
            'user_id': request.user_id,
            'symbol': request.symbol,
            'model_used': response.model_used,
            'action': response.action,
            'confidence': response.confidence,
            'should_trade': response.should_trade,
            'fallback_used': response.fallback_used
        })
    
    def log_trade_outcome(self, prediction_id: str, outcome: Dict[str, Any]):
        """Log the outcome of a trade based on a prediction"""
        self.trades.append({
            'timestamp': datetime.now(),
            'prediction_id': prediction_id,
            'outcome': outcome
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        if not self.predictions:
            return {"error": "No prediction data available"}
        
        recent_predictions = [p for p in self.predictions 
                            if (datetime.now() - p['timestamp']).total_seconds() < 3600]
        
        if not recent_predictions:
            return {"error": "No recent prediction data"}
        
        # Calculate metrics
        rl_predictions = [p for p in recent_predictions if not p['fallback_used']]
        fallback_predictions = [p for p in recent_predictions if p['fallback_used']]
        
        trading_signals = [p for p in recent_predictions if p['should_trade']]
        
        avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
        
        return {
            'total_predictions_1h': len(recent_predictions),
            'rl_predictions_1h': len(rl_predictions),
            'fallback_predictions_1h': len(fallback_predictions),
            'trading_signals_1h': len(trading_signals),
            'avg_confidence': avg_confidence,
            'rl_usage_rate': len(rl_predictions) / len(recent_predictions) if recent_predictions else 0,
            'signal_rate': len(trading_signals) / len(recent_predictions) if recent_predictions else 0
        }


class PerformanceTracker:
    """High-level performance tracking for the RL service"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'rl_requests': 0,
            'fallback_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': deque(maxlen=1000),
            'start_time': datetime.now()
        }
        
    def record_request(self, model_type: str):
        """Record a request by model type"""
        self.metrics['total_requests'] += 1
        if model_type == 'rl':
            self.metrics['rl_requests'] += 1
        else:
            self.metrics['fallback_requests'] += 1
    
    def record_response_time(self, response_time_ms: float):
        """Record response time"""
        self.metrics['response_times'].append(response_time_ms)
        self.metrics['successful_requests'] += 1
    
    def record_error(self):
        """Record an error"""
        self.metrics['failed_requests'] += 1
    
    def get_total_requests(self) -> int:
        """Get total number of requests"""
        return self.metrics['total_requests']
    
    def get_rl_requests(self) -> int:
        """Get number of RL requests"""
        return self.metrics['rl_requests']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        response_times = list(self.metrics['response_times'])
        
        uptime_seconds = (datetime.now() - self.metrics['start_time']).total_seconds()
        
        return {
            'total_requests': self.metrics['total_requests'],
            'rl_requests': self.metrics['rl_requests'],
            'fallback_requests': self.metrics['fallback_requests'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': (self.metrics['successful_requests'] / 
                           max(1, self.metrics['total_requests'])),
            'avg_response_time': np.mean(response_times) if response_times else 0.0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0.0,
            'uptime_seconds': uptime_seconds,
            'requests_per_second': self.metrics['total_requests'] / max(1, uptime_seconds)
        }
    
    def get_rl_metrics(self) -> Dict[str, Any]:
        """Get RL-specific metrics"""
        return {
            'total_requests': self.metrics['rl_requests'],
            'accuracy': 0.75,  # Placeholder - would calculate from actual results
            'avg_response_time': np.mean(list(self.metrics['response_times'])) if self.metrics['response_times'] else 0.0
        }
    
    def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback model metrics"""
        return {
            'total_requests': self.metrics['fallback_requests'],
            'accuracy': 0.65,  # Placeholder - would calculate from actual results
            'avg_response_time': np.mean(list(self.metrics['response_times'])) if self.metrics['response_times'] else 0.0
        }
    
    async def update_metrics(self):
        """Update computed metrics"""
        # Placeholder for any background metric calculations
        pass
    
    async def cleanup(self):
        """Cleanup performance tracker"""
        # Clear metrics if needed
        pass


# Global instances
rl_monitor = RLMonitor()
performance_tracker = PerformanceTracker()


if __name__ == "__main__":
    # Test the monitoring system
    async def test_monitoring():
        monitor = RLMonitor()
        
        try:
            await monitor.initialize()
            
            # Record some test metrics
            monitor.set_gauge("test_gauge", 42.5)
            monitor.increment_counter("test_counter", 3)
            monitor.record_histogram("test_histogram", 123.45)
            
            # Create test alert
            monitor.create_alert(
                "test_alert",
                "Test Alert",
                "This is a test alert",
                AlertSeverity.WARNING,
                "test_system"
            )
            
            # Wait a bit for background tasks
            await asyncio.sleep(2)
            
            # Get dashboard data
            dashboard_data = monitor.get_dashboard_data()
            print(f"Dashboard data: {json.dumps(dashboard_data, indent=2, default=str)}")
            
            # Get metrics summary
            gauge_summary = monitor.get_metric_summary("test_gauge")
            print(f"Gauge summary: {gauge_summary}")
            
            # Export Prometheus metrics
            prometheus_metrics = monitor.export_prometheus_metrics()
            print(f"Prometheus metrics:\n{prometheus_metrics}")
            
        finally:
            await monitor.cleanup()
    
    # Run test
    asyncio.run(test_monitoring())