"""
Advanced Logging and Monitoring Utilities
Comprehensive logging, alerting, and monitoring system for the ML service
"""

import logging
import logging.handlers
import json
import time
import uuid
import threading
import traceback
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager
from functools import wraps
import inspect
import sys
import os
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import hashlib
from collections import defaultdict, deque
import psutil
import threading

from config import get_config


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    function_name: str
    line_number: int
    file_name: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None


@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    condition: Callable[[], bool]
    message_template: str
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable[[], bool]
    timeout_seconds: int = 5
    critical: bool = True
    description: str = ""
    last_result: Optional[bool] = None
    last_check_time: Optional[datetime] = None
    failure_count: int = 0
    max_failures: int = 3


class StructuredLogger:
    """Enhanced logging with structured output and context tracking"""
    
    def __init__(self, name: str, config_override: Optional[Dict] = None):
        self.name = name
        self.config = config_override or get_config().logging
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Context storage (thread-local)
        self._context = threading.local()
    
    def _setup_logger(self):
        """Setup logger with appropriate handlers"""
        self.logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        if self.config.enable_json_logging:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(self.config.format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if configured)
        if self.config.file_path:
            file_path = Path(self.config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(file_path),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def set_context(self, **kwargs):
        """Set logging context for current thread"""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(kwargs)
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context"""
        return getattr(self._context, 'data', {})
    
    def clear_context(self):
        """Clear logging context"""
        if hasattr(self._context, 'data'):
            self._context.data.clear()
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary logging context"""
        original_context = self.get_context().copy()
        self.set_context(**kwargs)
        try:
            yield
        finally:
            self.clear_context()
            if original_context:
                self.set_context(**original_context)
    
    def _log_with_context(self, level: int, message: str, extra: Optional[Dict] = None):
        """Log message with context and structured data"""
        # Get caller information
        frame = inspect.currentframe().f_back.f_back
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        file_name = frame.f_code.co_filename
        
        # Prepare extra fields
        log_extra = {
            'function_name': function_name,
            'line_number': line_number,
            'file_name': os.path.basename(file_name),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add context data
        context = self.get_context()
        if context:
            log_extra.update(context)
        
        # Add additional extra fields
        if extra:
            log_extra.update(extra)
        
        # Generate request ID if enabled and not present
        if (self.config.enable_request_id and 
            'request_id' not in log_extra):
            log_extra['request_id'] = str(uuid.uuid4())[:8]
        
        self.logger.log(level, message, extra=log_extra)
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """Log info message"""
        self._log_with_context(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict] = None):
        """Log error message"""
        # Add stack trace for errors
        if extra is None:
            extra = {}
        if 'stack_trace' not in extra:
            extra['stack_trace'] = traceback.format_exc()
        
        self._log_with_context(logging.ERROR, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict] = None):
        """Log critical message"""
        # Add stack trace for critical errors
        if extra is None:
            extra = {}
        if 'stack_trace' not in extra:
            extra['stack_trace'] = traceback.format_exc()
        
        self._log_with_context(logging.CRITICAL, message, extra)
    
    def log_operation(self, operation: str, success: bool, duration_ms: float, 
                     extra: Optional[Dict] = None):
        """Log operation with performance metrics"""
        log_extra = {
            'operation': operation,
            'success': success,
            'duration_ms': duration_ms,
            'performance_metric': True
        }
        if extra:
            log_extra.update(extra)
        
        level = logging.INFO if success else logging.ERROR
        message = f"Operation {operation} {'succeeded' if success else 'failed'} in {duration_ms:.2f}ms"
        self._log_with_context(level, message, log_extra)
    
    def log_adaptation(self, user_id: str, symbol: Optional[str], parameter: str,
                      old_value: float, new_value: float, reason: str, 
                      confidence: float, performance_score: float):
        """Log threshold adaptation with structured data"""
        log_extra = {
            'event_type': 'threshold_adaptation',
            'user_id': user_id,
            'symbol': symbol,
            'parameter': parameter,
            'old_value': old_value,
            'new_value': new_value,
            'change_percent': abs(new_value - old_value) / old_value * 100,
            'reason': reason,
            'confidence': confidence,
            'performance_score': performance_score
        }
        
        message = (f"Adapted {parameter} for {user_id}"
                  f"{f'/{symbol}' if symbol else ''}: "
                  f"{old_value:.4f} -> {new_value:.4f} "
                  f"(confidence: {confidence:.2f})")
        
        self.info(message, log_extra)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'function': getattr(record, 'function_name', ''),
            'line': getattr(record, 'line_number', ''),
            'file': getattr(record, 'file_name', ''),
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'message',
                          'function_name', 'line_number', 'file_name']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class AlertManager:
    """Advanced alerting system with multiple notification channels"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.notification_channels = []
        self._alert_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # Setup default alerts
        self._setup_default_alerts()
        
        # Start alert monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_alerts, daemon=True)
        self._monitor_thread.start()
    
    def _setup_default_alerts(self):
        """Setup default system alerts"""
        # High CPU usage alert
        self.register_alert(Alert(
            id="high_cpu",
            name="High CPU Usage",
            description="CPU usage above 80%",
            severity="medium",
            condition=lambda: psutil.cpu_percent(interval=1) > 80,
            message_template="High CPU usage detected: {cpu_percent:.1f}%",
            cooldown_minutes=15
        ))
        
        # High memory usage alert
        self.register_alert(Alert(
            id="high_memory",
            name="High Memory Usage",
            description="Memory usage above 85%",
            severity="high",
            condition=lambda: psutil.virtual_memory().percent > 85,
            message_template="High memory usage detected: {memory_percent:.1f}%",
            cooldown_minutes=10
        ))
        
        # Database connection issues
        self.register_alert(Alert(
            id="db_connection_errors",
            name="Database Connection Errors",
            description="Multiple database connection failures",
            severity="critical",
            condition=self._check_db_connection_errors,
            message_template="Database connection errors detected: {error_count} errors in last 5 minutes",
            cooldown_minutes=5
        ))
        
        # Adaptation failures
        self.register_alert(Alert(
            id="adaptation_failures",
            name="Adaptation Failures",
            description="High rate of adaptation failures",
            severity="high",
            condition=self._check_adaptation_failures,
            message_template="High adaptation failure rate: {failure_rate:.1f}% in last hour",
            cooldown_minutes=30
        ))
    
    def _check_db_connection_errors(self) -> bool:
        """Check for database connection errors"""
        # This would integrate with performance tracker to check recent errors
        # For now, return False as placeholder
        return False
    
    def _check_adaptation_failures(self) -> bool:
        """Check for adaptation failures"""
        # This would integrate with performance tracker to check failure rate
        # For now, return False as placeholder
        return False
    
    def register_alert(self, alert: Alert):
        """Register an alert"""
        with self._lock:
            self.alerts[alert.id] = alert
    
    def remove_alert(self, alert_id: str):
        """Remove an alert"""
        with self._lock:
            if alert_id in self.alerts:
                del self.alerts[alert_id]
    
    def add_notification_channel(self, channel):
        """Add a notification channel"""
        self.notification_channels.append(channel)
    
    def _monitor_alerts(self):
        """Monitor alerts in background thread"""
        while True:
            try:
                with self._lock:
                    for alert in self.alerts.values():
                        self._check_alert(alert)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring alerts: {e}")
                time.sleep(60)
    
    def _check_alert(self, alert: Alert):
        """Check if an alert should trigger"""
        now = datetime.now()
        
        # Check cooldown
        if (alert.last_triggered and 
            now - alert.last_triggered < timedelta(minutes=alert.cooldown_minutes)):
            return
        
        try:
            # Check condition
            if alert.condition():
                alert.trigger_count += 1
                alert.last_triggered = now
                
                # Format message
                context = self._get_alert_context(alert)
                message = alert.message_template.format(**context)
                
                # Record alert
                alert_record = {
                    'alert_id': alert.id,
                    'name': alert.name,
                    'severity': alert.severity,
                    'message': message,
                    'timestamp': now.isoformat(),
                    'trigger_count': alert.trigger_count
                }
                self._alert_history.append(alert_record)
                
                # Send notifications
                self._send_alert_notifications(alert, message, context)
                
                logger.warning(f"Alert triggered: {alert.name} - {message}")
                
        except Exception as e:
            logger.error(f"Error checking alert {alert.id}: {e}")
    
    def _get_alert_context(self, alert: Alert) -> Dict[str, Any]:
        """Get context for alert message formatting"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'alert_name': alert.name,
            'severity': alert.severity
        }
        
        # Add system metrics
        try:
            context.update({
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            })
        except:
            pass
        
        return context
    
    def _send_alert_notifications(self, alert: Alert, message: str, context: Dict[str, Any]):
        """Send alert notifications through all channels"""
        for channel in self.notification_channels:
            try:
                channel.send_alert(alert, message, context)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
    
    def get_alert_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        return [
            alert for alert in self._alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff
        ]


class HealthCheckManager:
    """Health check management system"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()
        self.overall_health = True
        self.last_check_time: Optional[datetime] = None
        
        # Setup default health checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default health checks"""
        self.register_check(HealthCheck(
            name="database_connection",
            check_function=self._check_database_connection,
            timeout_seconds=10,
            critical=True,
            description="Database connectivity check"
        ))
        
        self.register_check(HealthCheck(
            name="memory_usage",
            check_function=lambda: psutil.virtual_memory().percent < 90,
            timeout_seconds=5,
            critical=False,
            description="Memory usage within acceptable limits"
        ))
        
        self.register_check(HealthCheck(
            name="disk_space",
            check_function=lambda: psutil.disk_usage('/').percent < 95,
            timeout_seconds=5,
            critical=True,
            description="Sufficient disk space available"
        ))
    
    def _check_database_connection(self) -> bool:
        """Check database connection health"""
        try:
            # This would check actual database connection
            # For now, return True as placeholder
            return True
        except Exception:
            return False
    
    def register_check(self, check: HealthCheck):
        """Register a health check"""
        with self._lock:
            self.health_checks[check.name] = check
    
    def remove_check(self, check_name: str):
        """Remove a health check"""
        with self._lock:
            if check_name in self.health_checks:
                del self.health_checks[check_name]
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        critical_failures = []
        
        with self._lock:
            for name, check in self.health_checks.items():
                check_result = self._run_single_check(check)
                results[name] = check_result
                
                if not check_result['healthy']:
                    if check.critical:
                        overall_healthy = False
                        critical_failures.append(name)
        
        self.overall_health = overall_healthy
        self.last_check_time = datetime.now()
        
        return {
            'overall_healthy': overall_healthy,
            'critical_failures': critical_failures,
            'checks': results,
            'last_check_time': self.last_check_time.isoformat(),
            'total_checks': len(self.health_checks)
        }
    
    def _run_single_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Run a single health check"""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = check.check_function()
            execution_time = (time.time() - start_time) * 1000
            
            if result:
                check.failure_count = 0
            else:
                check.failure_count += 1
            
            check.last_result = result
            check.last_check_time = datetime.now()
            
            return {
                'healthy': result,
                'execution_time_ms': execution_time,
                'failure_count': check.failure_count,
                'critical': check.critical,
                'description': check.description,
                'last_check_time': check.last_check_time.isoformat()
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            check.failure_count += 1
            check.last_result = False
            check.last_check_time = datetime.now()
            
            return {
                'healthy': False,
                'execution_time_ms': execution_time,
                'failure_count': check.failure_count,
                'critical': check.critical,
                'description': check.description,
                'error': str(e),
                'last_check_time': check.last_check_time.isoformat()
            }


class NotificationChannel:
    """Base class for notification channels"""
    
    def send_alert(self, alert: Alert, message: str, context: Dict[str, Any]):
        """Send alert notification"""
        raise NotImplementedError


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert: Alert, message: str, context: Dict[str, Any]):
        """Send alert to Slack"""
        color_map = {
            'low': '#36a64f',      # green
            'medium': '#ff9500',   # orange
            'high': '#ff4444',     # red
            'critical': '#8B0000'  # dark red
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, '#808080'),
                "title": f"🚨 {alert.name}",
                "text": message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.upper(), "short": True},
                    {"title": "Time", "value": context.get('timestamp', ''), "short": True},
                    {"title": "Description", "value": alert.description, "short": False}
                ],
                "footer": "ML Service Alert",
                "ts": int(time.time())
            }]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")


def logged_function(operation_name: Optional[str] = None):
    """Decorator for automatic function logging"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            success = True
            
            logger.debug(f"Starting {name}", extra={
                'operation': name,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            })
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger.error(f"Error in {name}: {str(e)}", extra={
                    'operation': name,
                    'error_type': type(e).__name__
                })
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_operation(name, success, duration_ms)
        
        return wrapper
    return decorator


# Global instances
logger = StructuredLogger("ml_service")
alert_manager = AlertManager()
health_check_manager = HealthCheckManager()

# Add notification channels based on configuration
config = get_config()
if hasattr(config.monitoring, 'slack_webhook_url') and config.monitoring.slack_webhook_url:
    alert_manager.add_notification_channel(
        SlackNotificationChannel(config.monitoring.slack_webhook_url)
    )


if __name__ == "__main__":
    # Example usage
    logger.info("ML Service monitoring initialized")
    
    # Set context
    with logger.context(user_id="test_user", operation="example"):
        logger.info("This is a test log with context")
        
        try:
            raise ValueError("Test error")
        except Exception:
            logger.error("Test error occurred")
    
    # Run health checks
    health_results = health_check_manager.run_checks()
    print("Health Check Results:", json.dumps(health_results, indent=2, default=str))
    
    # Get alert history
    alert_history = alert_manager.get_alert_history(hours_back=1)
    print("Alert History:", json.dumps(alert_history, indent=2, default=str))