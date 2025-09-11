"""
Real-time Risk Monitoring and Alert System
Continuous monitoring of risk metrics with automated alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from collections import deque
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import websocket
import threading

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class MetricType(Enum):
    """Types of monitored metrics"""
    DRAWDOWN = "DRAWDOWN"
    VAR = "VAR"
    LEVERAGE = "LEVERAGE"
    CORRELATION = "CORRELATION"
    VOLATILITY = "VOLATILITY"
    LIQUIDITY = "LIQUIDITY"
    PNL = "PNL"
    EXPOSURE = "EXPOSURE"
    SHARPE = "SHARPE"
    WIN_RATE = "WIN_RATE"
    SYSTEM = "SYSTEM"


@dataclass
class MonitorConfig:
    """Risk monitoring configuration"""
    # Update frequencies (seconds)
    update_frequency: int = 5  # Main update cycle
    alert_frequency: int = 60  # Minimum time between similar alerts
    
    # Alert thresholds
    drawdown_warning: float = 0.08
    drawdown_critical: float = 0.12
    var_warning: float = 0.025
    var_critical: float = 0.04
    leverage_warning: float = 3.0
    leverage_critical: float = 4.5
    correlation_warning: float = 0.80
    correlation_critical: float = 0.95
    volatility_spike_warning: float = 2.0
    volatility_spike_critical: float = 3.0
    
    # Performance thresholds
    sharpe_warning: float = 1.0
    win_rate_warning: float = 0.40
    daily_loss_warning: float = 0.03
    daily_loss_critical: float = 0.05
    
    # System thresholds
    latency_warning: int = 1000  # milliseconds
    latency_critical: int = 5000
    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.10
    
    # Alert settings
    enable_email_alerts: bool = False
    enable_sms_alerts: bool = False
    enable_webhook_alerts: bool = True
    enable_sound_alerts: bool = True
    
    # Data retention
    metric_history_size: int = 10000
    alert_history_size: int = 1000


@dataclass
class Alert:
    """Risk alert"""
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    metric_value: float
    threshold: float
    message: str
    metadata: Dict = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class MetricSnapshot:
    """Snapshot of all risk metrics"""
    timestamp: datetime
    drawdown: float
    var_1d: float
    cvar_1d: float
    leverage: float
    max_correlation: float
    volatility: float
    liquidity_ratio: float
    daily_pnl: float
    total_exposure: float
    sharpe_ratio: float
    win_rate: float
    positions_count: int
    system_health: float
    latency_ms: int
    error_rate: float


class RiskMonitor:
    """
    Real-time risk monitoring system with alerts
    """
    
    def __init__(self,
                 config: Optional[MonitorConfig] = None,
                 alert_callback: Optional[Callable] = None):
        """
        Initialize Risk Monitor
        
        Args:
            config: Monitor configuration
            alert_callback: Callback for alerts
        """
        self.config = config or MonitorConfig()
        self.alert_callback = alert_callback
        
        # Monitoring state
        self.is_running = False
        self.monitor_task = None
        
        # Metrics storage
        self.current_metrics = None
        self.metric_history = deque(maxlen=self.config.metric_history_size)
        self.alert_history = deque(maxlen=self.config.alert_history_size)
        
        # Alert management
        self.active_alerts: Set[str] = set()
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_level': {level.value: 0 for level in AlertLevel},
            'alerts_by_type': {metric.value: 0 for metric in MetricType},
            'start_time': datetime.now(),
            'uptime_seconds': 0
        }
        
        # WebSocket for real-time updates
        self.websocket = None
        self.ws_clients = []
        
        logger.info("Risk Monitor initialized")
    
    async def start(self):
        """Start risk monitoring"""
        if self.is_running:
            logger.warning("Risk Monitor already running")
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Risk Monitor started")
    
    async def stop(self):
        """Stop risk monitoring"""
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Risk Monitor stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store snapshot
                self.current_metrics = metrics
                self.metric_history.append(metrics)
                
                # Check for alerts
                alerts = self._check_alerts(metrics)
                
                # Process alerts
                for alert in alerts:
                    await self._process_alert(alert)
                
                # Broadcast updates
                await self._broadcast_updates(metrics)
                
                # Update statistics
                self._update_statistics()
                
                # Sleep until next update
                await asyncio.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.update_frequency)
    
    async def _collect_metrics(self) -> MetricSnapshot:
        """Collect current risk metrics"""
        # This would connect to actual data sources
        # For now, returning simulated metrics
        
        return MetricSnapshot(
            timestamp=datetime.now(),
            drawdown=np.random.uniform(0, 0.15),
            var_1d=np.random.uniform(0.01, 0.05),
            cvar_1d=np.random.uniform(0.02, 0.07),
            leverage=np.random.uniform(1, 5),
            max_correlation=np.random.uniform(0.3, 0.95),
            volatility=np.random.uniform(0.01, 0.05),
            liquidity_ratio=np.random.uniform(0.1, 1.0),
            daily_pnl=np.random.uniform(-0.05, 0.05),
            total_exposure=np.random.uniform(0, 2000000),
            sharpe_ratio=np.random.uniform(0.5, 2.5),
            win_rate=np.random.uniform(0.3, 0.7),
            positions_count=np.random.randint(0, 20),
            system_health=np.random.uniform(0.8, 1.0),
            latency_ms=np.random.randint(10, 2000),
            error_rate=np.random.uniform(0, 0.1)
        )
    
    def _check_alerts(self, metrics: MetricSnapshot) -> List[Alert]:
        """Check metrics against alert thresholds"""
        alerts = []
        
        # Drawdown alerts
        if metrics.drawdown >= self.config.drawdown_critical:
            alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.DRAWDOWN,
                metrics.drawdown,
                self.config.drawdown_critical,
                f"CRITICAL: Drawdown at {metrics.drawdown:.2%}"
            ))
        elif metrics.drawdown >= self.config.drawdown_warning:
            alerts.append(self._create_alert(
                AlertLevel.WARNING,
                MetricType.DRAWDOWN,
                metrics.drawdown,
                self.config.drawdown_warning,
                f"Warning: Drawdown at {metrics.drawdown:.2%}"
            ))
        
        # VaR alerts
        if metrics.var_1d >= self.config.var_critical:
            alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.VAR,
                metrics.var_1d,
                self.config.var_critical,
                f"CRITICAL: VaR at {metrics.var_1d:.2%}"
            ))
        elif metrics.var_1d >= self.config.var_warning:
            alerts.append(self._create_alert(
                AlertLevel.WARNING,
                MetricType.VAR,
                metrics.var_1d,
                self.config.var_warning,
                f"Warning: VaR at {metrics.var_1d:.2%}"
            ))
        
        # Leverage alerts
        if metrics.leverage >= self.config.leverage_critical:
            alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.LEVERAGE,
                metrics.leverage,
                self.config.leverage_critical,
                f"CRITICAL: Leverage at {metrics.leverage:.2f}x"
            ))
        elif metrics.leverage >= self.config.leverage_warning:
            alerts.append(self._create_alert(
                AlertLevel.WARNING,
                MetricType.LEVERAGE,
                metrics.leverage,
                self.config.leverage_warning,
                f"Warning: Leverage at {metrics.leverage:.2f}x"
            ))
        
        # Correlation alerts
        if metrics.max_correlation >= self.config.correlation_critical:
            alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.CORRELATION,
                metrics.max_correlation,
                self.config.correlation_critical,
                f"CRITICAL: Max correlation at {metrics.max_correlation:.2%}"
            ))
        elif metrics.max_correlation >= self.config.correlation_warning:
            alerts.append(self._create_alert(
                AlertLevel.WARNING,
                MetricType.CORRELATION,
                metrics.max_correlation,
                self.config.correlation_warning,
                f"Warning: Max correlation at {metrics.max_correlation:.2%}"
            ))
        
        # Daily loss alerts
        if metrics.daily_pnl <= -self.config.daily_loss_critical:
            alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.PNL,
                metrics.daily_pnl,
                -self.config.daily_loss_critical,
                f"CRITICAL: Daily loss at {-metrics.daily_pnl:.2%}"
            ))
        elif metrics.daily_pnl <= -self.config.daily_loss_warning:
            alerts.append(self._create_alert(
                AlertLevel.WARNING,
                MetricType.PNL,
                metrics.daily_pnl,
                -self.config.daily_loss_warning,
                f"Warning: Daily loss at {-metrics.daily_pnl:.2%}"
            ))
        
        # System alerts
        if metrics.latency_ms >= self.config.latency_critical:
            alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.SYSTEM,
                metrics.latency_ms,
                self.config.latency_critical,
                f"CRITICAL: System latency at {metrics.latency_ms}ms"
            ))
        elif metrics.latency_ms >= self.config.latency_warning:
            alerts.append(self._create_alert(
                AlertLevel.WARNING,
                MetricType.SYSTEM,
                metrics.latency_ms,
                self.config.latency_warning,
                f"Warning: System latency at {metrics.latency_ms}ms"
            ))
        
        # Filter alerts by cooldown
        filtered_alerts = []
        for alert in alerts:
            alert_key = f"{alert.metric_type.value}_{alert.level.value}"
            
            if alert_key in self.alert_cooldowns:
                last_alert_time = self.alert_cooldowns[alert_key]
                if (datetime.now() - last_alert_time).seconds < self.config.alert_frequency:
                    continue
            
            filtered_alerts.append(alert)
            self.alert_cooldowns[alert_key] = datetime.now()
        
        return filtered_alerts
    
    def _create_alert(self,
                     level: AlertLevel,
                     metric_type: MetricType,
                     value: float,
                     threshold: float,
                     message: str) -> Alert:
        """Create alert object"""
        return Alert(
            timestamp=datetime.now(),
            level=level,
            metric_type=metric_type,
            metric_value=value,
            threshold=threshold,
            message=message,
            metadata={
                'monitor_id': id(self),
                'config': self.config.__dict__
            }
        )
    
    async def _process_alert(self, alert: Alert):
        """Process and distribute alert"""
        # Store alert
        self.alert_history.append(alert)
        self.active_alerts.add(f"{alert.metric_type.value}_{alert.level.value}")
        
        # Update statistics
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_level'][alert.level.value] += 1
        self.stats['alerts_by_type'][alert.metric_type.value] += 1
        
        # Log alert
        if alert.level == AlertLevel.EMERGENCY:
            logger.critical(alert.message)
        elif alert.level == AlertLevel.CRITICAL:
            logger.error(alert.message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(alert.message)
        else:
            logger.info(alert.message)
        
        # Execute callback
        if self.alert_callback:
            await self.alert_callback(alert)
        
        # Send notifications
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            await self._send_notifications(alert)
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        tasks = []
        
        if self.config.enable_email_alerts:
            tasks.append(self._send_email_alert(alert))
        
        if self.config.enable_webhook_alerts:
            tasks.append(self._send_webhook_alert(alert))
        
        if self.config.enable_sound_alerts:
            tasks.append(self._play_alert_sound(alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert (placeholder)"""
        # In production, implement actual email sending
        logger.info(f"Email alert would be sent: {alert.message}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        webhook_data = {
            "timestamp": alert.timestamp.isoformat(),
            "level": alert.level.value,
            "type": alert.metric_type.value,
            "message": alert.message,
            "value": alert.metric_value,
            "threshold": alert.threshold
        }
        
        # In production, send to actual webhook
        logger.info(f"Webhook alert: {json.dumps(webhook_data)}")
    
    async def _play_alert_sound(self, alert: Alert):
        """Play alert sound (placeholder)"""
        logger.info(f"Alert sound would play for: {alert.level.value}")
    
    async def _broadcast_updates(self, metrics: MetricSnapshot):
        """Broadcast metrics updates to connected clients"""
        update_data = {
            "timestamp": metrics.timestamp.isoformat(),
            "metrics": {
                "drawdown": metrics.drawdown,
                "var": metrics.var_1d,
                "leverage": metrics.leverage,
                "pnl": metrics.daily_pnl,
                "positions": metrics.positions_count,
                "sharpe": metrics.sharpe_ratio
            },
            "alerts": len(self.active_alerts)
        }
        
        # Broadcast to WebSocket clients
        for client in self.ws_clients:
            try:
                await client.send(json.dumps(update_data))
            except:
                self.ws_clients.remove(client)
    
    def _update_statistics(self):
        """Update monitoring statistics"""
        self.stats['uptime_seconds'] = (datetime.now() - self.stats['start_time']).total_seconds()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for risk dashboard"""
        if not self.current_metrics:
            return {}
        
        # Calculate trend indicators
        recent_metrics = list(self.metric_history)[-100:] if len(self.metric_history) > 100 else list(self.metric_history)
        
        drawdown_trend = "stable"
        if len(recent_metrics) > 10:
            recent_drawdowns = [m.drawdown for m in recent_metrics[-10:]]
            if recent_drawdowns[-1] > recent_drawdowns[0] * 1.2:
                drawdown_trend = "worsening"
            elif recent_drawdowns[-1] < recent_drawdowns[0] * 0.8:
                drawdown_trend = "improving"
        
        return {
            "current_metrics": {
                "timestamp": self.current_metrics.timestamp.isoformat(),
                "drawdown": f"{self.current_metrics.drawdown:.2%}",
                "var_1d": f"{self.current_metrics.var_1d:.2%}",
                "leverage": f"{self.current_metrics.leverage:.2f}x",
                "daily_pnl": f"{self.current_metrics.daily_pnl:.2%}",
                "sharpe_ratio": f"{self.current_metrics.sharpe_ratio:.2f}",
                "positions": self.current_metrics.positions_count,
                "system_health": f"{self.current_metrics.system_health:.1%}"
            },
            "trends": {
                "drawdown": drawdown_trend,
                "metrics_collected": len(self.metric_history)
            },
            "alerts": {
                "active": list(self.active_alerts),
                "total": self.stats['total_alerts'],
                "by_level": self.stats['alerts_by_level'],
                "recent": [
                    {
                        "time": alert.timestamp.strftime("%H:%M:%S"),
                        "level": alert.level.value,
                        "message": alert.message
                    }
                    for alert in list(self.alert_history)[-5:]
                ]
            },
            "statistics": self.stats
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alert_history:
            if f"{alert.metric_type.value}_{alert.timestamp.timestamp()}" == alert_id:
                alert.acknowledged = True
                self.active_alerts.discard(f"{alert.metric_type.value}_{alert.level.value}")
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def export_metrics(self, filepath: str, format: str = "csv"):
        """Export metrics history"""
        if not self.metric_history:
            logger.warning("No metrics to export")
            return
        
        data = []
        for metric in self.metric_history:
            data.append({
                "timestamp": metric.timestamp.isoformat(),
                "drawdown": metric.drawdown,
                "var_1d": metric.var_1d,
                "leverage": metric.leverage,
                "daily_pnl": metric.daily_pnl,
                "sharpe_ratio": metric.sharpe_ratio,
                "positions": metric.positions_count
            })
        
        df = pd.DataFrame(data)
        
        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        
        logger.info(f"Exported {len(data)} metrics to {filepath}")
    
    def get_risk_score(self) -> float:
        """Calculate overall risk score (0-100)"""
        if not self.current_metrics:
            return 0
        
        scores = []
        
        # Drawdown score (0-25)
        dd_score = min(25, (self.current_metrics.drawdown / 0.15) * 25)
        scores.append(dd_score)
        
        # VaR score (0-25)
        var_score = min(25, (self.current_metrics.var_1d / 0.05) * 25)
        scores.append(var_score)
        
        # Leverage score (0-25)
        lev_score = min(25, (self.current_metrics.leverage / 5.0) * 25)
        scores.append(lev_score)
        
        # System health score (0-25)
        health_score = (1 - self.current_metrics.system_health) * 25
        scores.append(health_score)
        
        return sum(scores)