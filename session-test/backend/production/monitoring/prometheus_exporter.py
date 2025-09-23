"""
Prometheus Metrics Exporter for Trading Bot
Exports comprehensive metrics for trading performance, system health, and risk management.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server
)
import psutil
import threading
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingMetrics:
    """Container for trading performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_return_per_trade: float = 0.0
    total_trades: int = 0
    open_positions: int = 0
    portfolio_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class SystemMetrics:
    """Container for system health metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    api_response_time: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0

@dataclass
class RiskMetrics:
    """Container for risk management metrics"""
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    position_concentration: float = 0.0
    leverage_ratio: float = 0.0
    correlation_risk: float = 0.0
    liquidity_risk: float = 0.0
    market_exposure: float = 0.0

class PrometheusExporter:
    """
    Comprehensive Prometheus metrics exporter for trading bot monitoring.
    
    Tracks:
    - Trading performance metrics (returns, Sharpe ratio, drawdown)
    - System health metrics (CPU, memory, latency)
    - Risk management metrics (VaR, concentration, leverage)
    - Agent performance metrics (RL agents, ensemble performance)
    """
    
    def __init__(self, port: int = 8000, registry: Optional[CollectorRegistry] = None):
        self.port = port
        self.registry = registry or CollectorRegistry()
        self._running = False
        self._metrics_cache = {}
        self._last_update = time.time()
        
        # Initialize Prometheus metrics
        self._init_trading_metrics()
        self._init_system_metrics()
        self._init_risk_metrics()
        self._init_agent_metrics()
        self._init_operational_metrics()
        
    def _init_trading_metrics(self):
        """Initialize trading performance metrics"""
        # Trading Performance Gauges
        self.total_return = Gauge(
            'trading_total_return',
            'Total portfolio return percentage',
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'trading_sharpe_ratio',
            'Portfolio Sharpe ratio',
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            'trading_max_drawdown',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        
        self.win_rate = Gauge(
            'trading_win_rate',
            'Win rate percentage',
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'trading_portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.unrealized_pnl = Gauge(
            'trading_unrealized_pnl_usd',
            'Unrealized P&L in USD',
            registry=self.registry
        )
        
        self.realized_pnl = Gauge(
            'trading_realized_pnl_usd',
            'Realized P&L in USD',
            registry=self.registry
        )
        
        self.open_positions = Gauge(
            'trading_open_positions',
            'Number of open positions',
            registry=self.registry
        )
        
        # Trading Activity Counters
        self.trades_total = Counter(
            'trading_trades_total',
            'Total number of trades executed',
            ['strategy', 'side', 'symbol'],
            registry=self.registry
        )
        
        self.trade_volume_usd = Counter(
            'trading_volume_usd_total',
            'Total trading volume in USD',
            ['symbol'],
            registry=self.registry
        )
        
        # Trade Execution Timing
        self.trade_execution_time = Histogram(
            'trading_execution_duration_seconds',
            'Trade execution time in seconds',
            ['strategy', 'order_type'],
            registry=self.registry
        )
        
    def _init_system_metrics(self):
        """Initialize system health metrics"""
        # System Resource Gauges
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        self.network_latency = Gauge(
            'system_network_latency_ms',
            'Network latency in milliseconds',
            ['endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'system_active_connections',
            'Number of active connections',
            ['service'],
            registry=self.registry
        )
        
        # API Performance
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        # Error Tracking
        self.errors_total = Counter(
            'errors_total',
            'Total errors by type',
            ['error_type', 'service'],
            registry=self.registry
        )
        
    def _init_risk_metrics(self):
        """Initialize risk management metrics"""
        # Value at Risk
        self.var_95 = Gauge(
            'risk_var_95_percent',
            'Value at Risk 95% confidence',
            registry=self.registry
        )
        
        self.var_99 = Gauge(
            'risk_var_99_percent',
            'Value at Risk 99% confidence',
            registry=self.registry
        )
        
        # Position Risk
        self.position_concentration = Gauge(
            'risk_position_concentration',
            'Position concentration risk (0-1)',
            registry=self.registry
        )
        
        self.leverage_ratio = Gauge(
            'risk_leverage_ratio',
            'Current leverage ratio',
            registry=self.registry
        )
        
        self.market_exposure = Gauge(
            'risk_market_exposure_percent',
            'Total market exposure percentage',
            registry=self.registry
        )
        
        # Correlation and Liquidity Risk
        self.correlation_risk = Gauge(
            'risk_correlation_score',
            'Portfolio correlation risk score',
            registry=self.registry
        )
        
        self.liquidity_risk = Gauge(
            'risk_liquidity_score',
            'Portfolio liquidity risk score',
            registry=self.registry
        )
        
        # Risk Events
        self.risk_breaches = Counter(
            'risk_breaches_total',
            'Risk limit breaches',
            ['limit_type', 'severity'],
            registry=self.registry
        )
        
    def _init_agent_metrics(self):
        """Initialize RL agent and ensemble performance metrics"""
        # RL Agent Performance
        self.agent_reward = Gauge(
            'agent_cumulative_reward',
            'Cumulative reward by agent',
            ['agent_type', 'strategy'],
            registry=self.registry
        )
        
        self.agent_actions = Counter(
            'agent_actions_total',
            'Total actions taken by agent',
            ['agent_type', 'action_type'],
            registry=self.registry
        )
        
        self.agent_accuracy = Gauge(
            'agent_prediction_accuracy',
            'Agent prediction accuracy',
            ['agent_type', 'timeframe'],
            registry=self.registry
        )
        
        # Ensemble Performance
        self.ensemble_consensus = Gauge(
            'ensemble_consensus_score',
            'Ensemble consensus score (0-1)',
            registry=self.registry
        )
        
        self.agent_weights = Gauge(
            'ensemble_agent_weight',
            'Individual agent weight in ensemble',
            ['agent_id'],
            registry=self.registry
        )
        
        # Model Training Metrics
        self.training_loss = Gauge(
            'model_training_loss',
            'Model training loss',
            ['model_type', 'epoch'],
            registry=self.registry
        )
        
        self.model_inference_time = Histogram(
            'model_inference_duration_seconds',
            'Model inference time',
            ['model_type'],
            registry=self.registry
        )
        
    def _init_operational_metrics(self):
        """Initialize operational and service metrics"""
        # Service Health
        self.service_uptime = Gauge(
            'service_uptime_seconds',
            'Service uptime in seconds',
            ['service'],
            registry=self.registry
        )
        
        self.service_health = Gauge(
            'service_health_status',
            'Service health status (1=healthy, 0=unhealthy)',
            ['service'],
            registry=self.registry
        )
        
        # Data Quality
        self.data_freshness = Gauge(
            'data_freshness_seconds',
            'Data age in seconds',
            ['data_source'],
            registry=self.registry
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-1)',
            ['data_source'],
            registry=self.registry
        )
        
        # Background Tasks
        self.background_tasks_active = Gauge(
            'background_tasks_active',
            'Number of active background tasks',
            ['task_type'],
            registry=self.registry
        )
        
        self.background_tasks_completed = Counter(
            'background_tasks_completed_total',
            'Total completed background tasks',
            ['task_type', 'status'],
            registry=self.registry
        )
        
    @contextmanager
    def time_operation(self, operation_name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations"""
        labels = labels or {}
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            # Record to appropriate histogram based on operation name
            if 'trade' in operation_name.lower():
                self.trade_execution_time.labels(**labels).observe(duration)
            elif 'api' in operation_name.lower():
                self.api_request_duration.labels(**labels).observe(duration)
            elif 'model' in operation_name.lower():
                self.model_inference_time.labels(**labels).observe(duration)
                
    def update_trading_metrics(self, metrics: TradingMetrics):
        """Update trading performance metrics"""
        self.total_return.set(metrics.total_return)
        self.sharpe_ratio.set(metrics.sharpe_ratio)
        self.max_drawdown.set(metrics.max_drawdown)
        self.win_rate.set(metrics.win_rate)
        self.portfolio_value.set(metrics.portfolio_value)
        self.unrealized_pnl.set(metrics.unrealized_pnl)
        self.realized_pnl.set(metrics.realized_pnl)
        self.open_positions.set(metrics.open_positions)
        
    def update_system_metrics(self, metrics: SystemMetrics):
        """Update system health metrics"""
        self.cpu_usage.set(metrics.cpu_usage)
        self.memory_usage.set(metrics.memory_usage)
        self.disk_usage.set(metrics.disk_usage)
        if metrics.network_latency > 0:
            self.network_latency.labels(endpoint='default').set(metrics.network_latency)
        self.active_connections.labels(service='trading').set(metrics.active_connections)
        
    def update_risk_metrics(self, metrics: RiskMetrics):
        """Update risk management metrics"""
        self.var_95.set(metrics.var_95)
        self.var_99.set(metrics.var_99)
        self.position_concentration.set(metrics.position_concentration)
        self.leverage_ratio.set(metrics.leverage_ratio)
        self.market_exposure.set(metrics.market_exposure)
        self.correlation_risk.set(metrics.correlation_risk)
        self.liquidity_risk.set(metrics.liquidity_risk)
        
    def record_trade(self, strategy: str, side: str, symbol: str, volume_usd: float):
        """Record a completed trade"""
        self.trades_total.labels(strategy=strategy, side=side, symbol=symbol).inc()
        self.trade_volume_usd.labels(symbol=symbol).inc(volume_usd)
        
    def record_api_request(self, endpoint: str, method: str, status: str, duration: float):
        """Record API request metrics"""
        self.api_requests_total.labels(endpoint=endpoint, method=method, status=status).inc()
        self.api_request_duration.labels(endpoint=endpoint, method=method).observe(duration)
        
    def record_error(self, error_type: str, service: str):
        """Record an error occurrence"""
        self.errors_total.labels(error_type=error_type, service=service).inc()
        
    def record_risk_breach(self, limit_type: str, severity: str):
        """Record a risk limit breach"""
        self.risk_breaches.labels(limit_type=limit_type, severity=severity).inc()
        
    def update_agent_metrics(self, agent_type: str, reward: float, accuracy: float, weight: float = None):
        """Update RL agent performance metrics"""
        self.agent_reward.labels(agent_type=agent_type, strategy='default').set(reward)
        self.agent_accuracy.labels(agent_type=agent_type, timeframe='1h').set(accuracy)
        if weight is not None:
            self.agent_weights.labels(agent_id=agent_type).set(weight)
            
    def record_agent_action(self, agent_type: str, action_type: str):
        """Record an agent action"""
        self.agent_actions.labels(agent_type=agent_type, action_type=action_type).inc()
        
    def update_service_health(self, service: str, is_healthy: bool, uptime_seconds: float):
        """Update service health status"""
        self.service_health.labels(service=service).set(1 if is_healthy else 0)
        self.service_uptime.labels(service=service).set(uptime_seconds)
        
    def update_data_quality(self, data_source: str, freshness_seconds: float, quality_score: float):
        """Update data quality metrics"""
        self.data_freshness.labels(data_source=data_source).set(freshness_seconds)
        self.data_quality_score.labels(data_source=data_source).set(quality_score)
        
    def collect_system_metrics(self):
        """Collect current system metrics automatically"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage for root partition
            disk = psutil.disk_usage('/')
            
            # Network connections
            connections = len(psutil.net_connections())
            
            system_metrics = SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                active_connections=connections
            )
            
            self.update_system_metrics(system_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            self.record_error("system_metrics_collection", "monitoring")
            
    def start_auto_collection(self, interval: int = 30):
        """Start automatic system metrics collection"""
        def collect_loop():
            while self._running:
                try:
                    self.collect_system_metrics()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in collection loop: {e}")
                    time.sleep(interval)
                    
        self._running = True
        collection_thread = threading.Thread(target=collect_loop, daemon=True)
        collection_thread.start()
        logger.info(f"Started automatic metrics collection with {interval}s interval")
        
    def stop_auto_collection(self):
        """Stop automatic metrics collection"""
        self._running = False
        logger.info("Stopped automatic metrics collection")
        
    def start_server(self, port: Optional[int] = None):
        """Start Prometheus HTTP server"""
        server_port = port or self.port
        start_http_server(server_port, registry=self.registry)
        logger.info(f"Prometheus metrics server started on port {server_port}")
        
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
        
    async def async_collect_metrics(self, trading_data: Dict[str, Any] = None, 
                                   risk_data: Dict[str, Any] = None,
                                   agent_data: Dict[str, Any] = None):
        """Asynchronously collect and update all metrics"""
        try:
            # Update trading metrics if provided
            if trading_data:
                trading_metrics = TradingMetrics(**trading_data)
                self.update_trading_metrics(trading_metrics)
                
            # Update risk metrics if provided
            if risk_data:
                risk_metrics = RiskMetrics(**risk_data)
                self.update_risk_metrics(risk_metrics)
                
            # Update agent metrics if provided
            if agent_data:
                for agent_id, data in agent_data.items():
                    self.update_agent_metrics(
                        agent_type=agent_id,
                        reward=data.get('reward', 0),
                        accuracy=data.get('accuracy', 0),
                        weight=data.get('weight')
                    )
                    
            # Always collect system metrics
            self.collect_system_metrics()
            
            self._last_update = time.time()
            
        except Exception as e:
            logger.error(f"Error in async metrics collection: {e}")
            self.record_error("async_collection", "monitoring")

# Global exporter instance
_exporter_instance = None

def get_exporter() -> PrometheusExporter:
    """Get global exporter instance (singleton pattern)"""
    global _exporter_instance
    if _exporter_instance is None:
        _exporter_instance = PrometheusExporter()
    return _exporter_instance

def initialize_monitoring(port: int = 8000, auto_collect: bool = True) -> PrometheusExporter:
    """Initialize monitoring system"""
    exporter = PrometheusExporter(port=port)
    
    if auto_collect:
        exporter.start_auto_collection()
        
    exporter.start_server()
    
    global _exporter_instance
    _exporter_instance = exporter
    
    logger.info("Trading bot monitoring initialized successfully")
    return exporter

if __name__ == "__main__":
    # Example usage
    exporter = initialize_monitoring(port=8000)
    
    # Example metric updates
    trading_metrics = TradingMetrics(
        total_return=15.5,
        sharpe_ratio=1.8,
        max_drawdown=-5.2,
        win_rate=65.0,
        portfolio_value=100000.0,
        open_positions=3
    )
    
    exporter.update_trading_metrics(trading_metrics)
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exporter.stop_auto_collection()
        logger.info("Monitoring stopped")